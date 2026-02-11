/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <sys/types.h>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "index/neighbor.hpp"
#include "simd/distance_ip.hpp"
#include "simd/distance_l2.hpp"
#include "space/quant/rabitq.hpp"
#include "space/space_concepts.hpp"
#include "storage/rocksdb_storage.hpp"
#include "storage/static_storage.hpp"
#include "utils/log.hpp"
#include "utils/math.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/metric_type.hpp"
#include "utils/prefetch.hpp"
#include "utils/rabitq_utils/fastscan.hpp"
#include "utils/rabitq_utils/lut.hpp"
#include "utils/rabitq_utils/rotator.hpp"

namespace alaya {
template <typename DataType = float,
          typename DistanceType = float,
          typename IDType = uint32_t,
          typename ScalarDataType = EmptyScalarData>
class RaBitQSpace {
 public:
  static constexpr bool has_scalar_data =  // NOLINT
      !std::is_same_v<ScalarDataType, EmptyScalarData>;

 private:
  IDType capacity_{0};                 ///< The maximum number of data points (nodes)
  uint32_t dim_{0};                    ///< Dimensionality of the data points
  MetricType metric_{MetricType::L2};  ///< Metric type
  RotatorType type_;                   ///< Rotator type
  IDType item_cnt_{0};                 ///< Number of data points (nodes)

  size_t quant_codes_offset_{0};
  size_t f_add_offset_{0};
  size_t f_rescale_offset_{0};
  size_t nei_id_offset_{0};
  size_t data_chunk_size_{0};  ///< Size of each node's data chunk

  DistFuncRaBitQ<DataType, DistanceType> distance_cal_func_;  ///< Distance calculation function

  StaticStorage<> storage_;  ///< Data Storage
  RocksDBConfig config_;     ///< Configuration for Scalar Data Storage
  std::unique_ptr<RocksDBStorage<IDType>>
      scalar_storage_;  ///< Scalar Data Storage (stores ScalarData)
  std::unique_ptr<RaBitQQuantizer<DataType>> quantizer_;  ///< Data Quantizer
  std::unique_ptr<Rotator<DataType>> rotator_;            ///< Data rotator

  IDType ep_;  ///< search entry point

  void initialize_offsets() {
    // data layout: (for each node, degree_bound defines their final outdegree)
    // 1. Its raw data vector
    // 2. Its neighbors' quantization codes
    // 3. F_add , F_rescale : please refer to
    // https://github.com/VectorDB-NTU/RaBitQ-Library/blob/main/docs/docs/rabitq/estimator.md
    // for detailed information
    // 4. Its neighbors' IDs
    size_t rvec_len = dim_ * sizeof(DataType);
    size_t nei_quant_code_len = get_padded_dim() * kDegreeBound / 8;  // 1 b/dim code
    size_t f_add_len = kDegreeBound * sizeof(DataType);
    size_t f_rescale_len = kDegreeBound * sizeof(DataType);
    size_t nei_id_len = kDegreeBound * sizeof(IDType);

    // byte
    quant_codes_offset_ = rvec_len;
    f_add_offset_ = quant_codes_offset_ + nei_quant_code_len;
    f_rescale_offset_ = f_add_offset_ + f_add_len;
    nei_id_offset_ = f_rescale_offset_ + f_rescale_len;
    data_chunk_size_ = nei_id_offset_ + nei_id_len;

    set_metric_function();
  }

  void save_scalar_config(std::ofstream &writer) {
    // Save db_path_ string
    size_t db_path_size = config_.db_path_.size();
    writer.write(reinterpret_cast<char *>(&db_path_size), sizeof(db_path_size));
    writer.write(config_.db_path_.data(), db_path_size);

    // Save POD fields
    writer.write(reinterpret_cast<char *>(&config_.write_buffer_size_),
                 sizeof(config_.write_buffer_size_));
    writer.write(reinterpret_cast<char *>(&config_.max_write_buffer_number_),
                 sizeof(config_.max_write_buffer_number_));
    writer.write(reinterpret_cast<char *>(&config_.target_file_size_base_),
                 sizeof(config_.target_file_size_base_));
    writer.write(reinterpret_cast<char *>(&config_.max_background_compactions_),
                 sizeof(config_.max_background_compactions_));
    writer.write(reinterpret_cast<char *>(&config_.max_background_flushes_),
                 sizeof(config_.max_background_flushes_));
    writer.write(reinterpret_cast<char *>(&config_.block_cache_size_mb_),
                 sizeof(config_.block_cache_size_mb_));

    // Save bool as uint8_t for cross-platform compatibility
    uint8_t enable_compression = config_.enable_compression_ ? 1 : 0;
    writer.write(reinterpret_cast<char *>(&enable_compression), sizeof(enable_compression));

    // Save indexed_fields_ for secondary index support
    size_t fields_count = config_.indexed_fields_.size();
    writer.write(reinterpret_cast<const char *>(&fields_count), sizeof(fields_count));
    for (const auto &field : config_.indexed_fields_) {
      size_t field_len = field.size();
      writer.write(reinterpret_cast<const char *>(&field_len), sizeof(field_len));
      writer.write(field.data(), field_len);
    }
  }

  void load_scalar_config(std::ifstream &reader) {
    config_.create_if_missing_ = false;  // db is missing means something's wrong
    config_.error_if_exists_ = false;    // Of course db exists
    // Load db_path_ string
    size_t db_path_size;
    reader.read(reinterpret_cast<char *>(&db_path_size), sizeof(db_path_size));
    config_.db_path_.resize(db_path_size);
    reader.read(config_.db_path_.data(), db_path_size);

    // Load POD fields
    reader.read(reinterpret_cast<char *>(&config_.write_buffer_size_),
                sizeof(config_.write_buffer_size_));
    reader.read(reinterpret_cast<char *>(&config_.max_write_buffer_number_),
                sizeof(config_.max_write_buffer_number_));
    reader.read(reinterpret_cast<char *>(&config_.target_file_size_base_),
                sizeof(config_.target_file_size_base_));
    reader.read(reinterpret_cast<char *>(&config_.max_background_compactions_),
                sizeof(config_.max_background_compactions_));
    reader.read(reinterpret_cast<char *>(&config_.max_background_flushes_),
                sizeof(config_.max_background_flushes_));
    reader.read(reinterpret_cast<char *>(&config_.block_cache_size_mb_),
                sizeof(config_.block_cache_size_mb_));

    // Load bool from uint8_t for cross-platform compatibility
    uint8_t enable_compression = 1;  // default to true
    reader.read(reinterpret_cast<char *>(&enable_compression), sizeof(enable_compression));
    config_.enable_compression_ = (enable_compression != 0);

    // Load indexed_fields_ for secondary index support
    size_t fields_count = 0;
    reader.read(reinterpret_cast<char *>(&fields_count), sizeof(fields_count));
    if (reader.good() && fields_count < 1000) {
      config_.indexed_fields_.clear();
      for (size_t i = 0; i < fields_count; i++) {
        size_t field_len = 0;
        reader.read(reinterpret_cast<char *>(&field_len), sizeof(field_len));
        std::string field(field_len, '\0');
        reader.read(field.data(), field_len);
        config_.indexed_fields_.push_back(std::move(field));
      }
    }
  }

 public:
  using DataTypeAlias = DataType;
  using DistanceTypeAlias = DistanceType;
  using IDTypeAlias = IDType;
  using DistDataType = DataType;

  // if you change degree bound , you should consider changing the layout too
  constexpr static size_t kDegreeBound = 32;  ///< Out degree of each node (in final graph)

  RaBitQSpace() = default;
  ~RaBitQSpace() = default;

  RaBitQSpace(RaBitQSpace &&other) = delete;
  RaBitQSpace(const RaBitQSpace &other) = delete;
  RaBitQSpace(IDType capacity,
              size_t dim,
              MetricType metric,
              RocksDBConfig config = RocksDBConfig::default_config(),
              RotatorType type = RotatorType::FhtKacRotator)
      : capacity_(capacity), dim_(dim), metric_(metric), type_(type), config_(std::move(config)) {
    if constexpr (!std::is_same_v<DataType, float> || !std::is_same_v<DistanceType, float>) {
      throw std::runtime_error("RaBitQSpace only supports float as DataType and DistanceType");
    }
    rotator_ = choose_rotator<DataType>(dim_, type_, alaya::math::round_up_pow2<size_t>(dim_, 64));
    quantizer_ = std::make_unique<RaBitQQuantizer<DataType>>(dim_, rotator_->size());
    initialize_offsets();
  }
  auto operator=(const RaBitQSpace &) -> RaBitQSpace & = delete;
  auto operator=(RaBitQSpace &&) -> RaBitQSpace & = delete;

  auto insert(DataType *data, const ScalarDataType *scalar_data = nullptr) -> IDType {
    throw std::runtime_error("Insert operation is not supported yet!");
  }
  auto remove(IDType id) -> IDType {
    throw std::runtime_error("Remove operation is not supported yet!");
  }

  /**
   * @brief Remove a data point by its item_id
   * @param item_id The item_id to remove
   * @return The internal ID that was removed
   * @throws std::runtime_error - not supported yet
   */
  auto remove(const std::string &item_id) -> IDType {
    throw std::runtime_error("Remove operation is not supported yet!");
  }

  void set_ep(IDType ep) { ep_ = ep; }
  auto get_ep() const -> IDType { return ep_; }

  void set_metric_function() {
    switch (metric_) {
      case MetricType::L2:
        distance_cal_func_ = simd::get_l2_sqr_func();
        break;
      case MetricType::COS:
      case MetricType::IP:
        distance_cal_func_ = simd::get_ip_sqr_func();
        break;
      default:
        throw std::runtime_error("invalid metric type.");
        break;
    }
  }

  void update_nei(IDType c, const std::vector<Neighbor<IDType, DistanceType>> &new_neighbors) {
    auto nei_ptr = get_edges(c);
    // update neighbors' IDs
    for (size_t i = 0; i < kDegreeBound; ++i) {
      *(nei_ptr + i) = new_neighbors[i].id_;
    }

    // rotate data before quantization
    std::vector<DataType> rotated_neighbors(kDegreeBound * get_padded_dim());
    std::vector<DataType> rotated_centroid(get_padded_dim());
    for (size_t i = 0; i < kDegreeBound; ++i) {
      const auto *neighbor_vec = get_data_by_id(new_neighbors[i].id_);
      this->rotator_->rotate(neighbor_vec, &rotated_neighbors[i * get_padded_dim()]);
    }
    this->rotator_->rotate(get_data_by_id(c), rotated_centroid.data());

    // quantize data and update batch data
    quantizer_->batch_quantize(rotated_neighbors.data(),
                               rotated_centroid.data(),
                               kDegreeBound,
                               get_nei_qc_ptr(c),
                               get_f_add_ptr(c),
                               get_f_rescale_ptr(c),
                               metric_);
  }

  void fit(const DataType *data, IDType item_cnt, const ScalarDataType *scalar_data = nullptr) {
    if (data == nullptr) {
      throw std::invalid_argument("Invalid or null vector data pointer.");
    }

    if constexpr (!std::is_floating_point_v<DataType>) {
      throw std::invalid_argument("Data type must be a floating point type!");
    }

    if constexpr (!(std::is_integral_v<IDType> && std::is_unsigned_v<IDType> &&
                    sizeof(IDType) == 4)) {
      throw std::invalid_argument("IDType must be a 32-bit unsigned integer!");
      // otherwise SearchBuffer and LinearPool won't function correctly.
    }

    if (item_cnt > capacity_) {
      throw std::length_error("The number of data points exceeds the capacity of the space");
    }
    item_cnt_ = item_cnt;

    // We don't fit after loading , so loaded storage_ would not be overwritten.
    storage_ = StaticStorage<>(std::vector<size_t>{item_cnt_, data_chunk_size_});
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < static_cast<int64_t>(item_cnt); i++) {
      const auto *src = data + (dim_ * i);
      auto *dst = get_data_by_id(i);
      std::copy(src, src + dim_, dst);
    }

    // Store ScalarData with synchronized IDs (0, 1, 2, ...)
    if constexpr (has_scalar_data) {
      if (scalar_data == nullptr) {
        throw std::invalid_argument("Invalid or null ScalarData pointer.");
      }
      if (scalar_storage_ == nullptr) {
        config_.error_if_exists_ = true;
        scalar_storage_ = std::make_unique<RocksDBStorage<IDType>>(config_);
      }
      // Batch insert with starting ID 0, ensuring sync with vector storage IDs
      if (!scalar_storage_->batch_insert(static_cast<IDType>(0),
                                         scalar_data,
                                         scalar_data + item_cnt)) {
        throw std::runtime_error("Failed to batch insert ScalarData");
      }
    }
  }

  auto get_distance(IDType i, IDType j) -> DistanceType {
    return distance_cal_func_(get_data_by_id(i), get_data_by_id(j), dim_);
  }

  [[nodiscard]] auto get_data_by_id(IDType id) const -> const DataType * {
    return reinterpret_cast<const DataType *>(&storage_.at(data_chunk_size_ * id));
  }

  [[nodiscard]] auto get_data_by_id(IDType id) -> DataType * {
    return reinterpret_cast<DataType *>(&storage_.at(data_chunk_size_ * id));
  }

  // get neighbors' quantization codes pointer
  [[nodiscard]] auto get_nei_qc_ptr(IDType id) const -> const uint8_t * {
    return reinterpret_cast<const uint8_t *>(
        &storage_.at((data_chunk_size_ * id) + quant_codes_offset_));
  }

  [[nodiscard]] auto get_nei_qc_ptr(IDType id) -> uint8_t * {
    return reinterpret_cast<uint8_t *>(&storage_.at((data_chunk_size_ * id) + quant_codes_offset_));
  }

  // get f_add pointer
  [[nodiscard]] auto get_f_add_ptr(IDType id) const -> const DataType * {
    return reinterpret_cast<const DataType *>(
        &storage_.at((data_chunk_size_ * id) + f_add_offset_));
  }

  [[nodiscard]] auto get_f_add_ptr(IDType id) -> DataType * {
    return reinterpret_cast<DataType *>(&storage_.at((data_chunk_size_ * id) + f_add_offset_));
  }

  // get f_rescale pointer
  [[nodiscard]] auto get_f_rescale_ptr(IDType id) const -> const DataType * {
    return reinterpret_cast<const DataType *>(
        &storage_.at((data_chunk_size_ * id) + f_rescale_offset_));
  }

  [[nodiscard]] auto get_f_rescale_ptr(IDType id) -> DataType * {
    return reinterpret_cast<DataType *>(&storage_.at((data_chunk_size_ * id) + f_rescale_offset_));
  }

  // get neighbors' IDs
  [[nodiscard]] auto get_edges(IDType id) const -> const IDType * {
    return reinterpret_cast<IDType *>(&storage_.at((data_chunk_size_ * id) + nei_id_offset_));
  }

  [[nodiscard]] auto get_edges(IDType id) -> IDType * {
    return reinterpret_cast<IDType *>(&storage_.at((data_chunk_size_ * id) + nei_id_offset_));
  }

  auto get_scalar_data(IDType id) const -> ScalarDataType {
    if constexpr (has_scalar_data) {
      return (*scalar_storage_)[id];
    }
    throw std::runtime_error("No ScalarData available.");
  }

  /**
   * @brief Get scalar data by item_id
   * @param item_id The item_id to look up
   * @return Pair of (internal_id, scalar_data)
   * @throws std::runtime_error if item_id not found or no scalar data available
   */
  auto get_scalar_data(const std::string &item_id) const -> std::pair<IDType, ScalarDataType> {
    if constexpr (has_scalar_data) {
      auto internal_id = scalar_storage_->find_by_item_id(item_id);
      if (!internal_id.has_value()) {
        throw std::runtime_error("Item ID not found: " + item_id);
      }
      return {internal_id.value(), (*scalar_storage_)[internal_id.value()]};
    }
    throw std::runtime_error("No ScalarData available.");
  }

  /**
   * @brief Get scalar data with metadata filter
   * @param filter MetadataFilter to apply
   * @param limit Maximum number of results
   * @return Vector of (internal_id, scalar_data) pairs
   */
  auto get_scalar_data(const MetadataFilter &filter, size_t limit) const
      -> std::vector<std::pair<IDType, ScalarDataType>> {
    if constexpr (has_scalar_data) {
      return scalar_storage_->scan_with_filter(
          [&filter](const ScalarData &sd) {
            return filter.evaluate(sd.metadata);
          },
          limit);
    }
    throw std::runtime_error("No ScalarData available.");
  }

  /**
   * @brief Get the scalar storage for direct index access
   * @return Pointer to RocksDBStorage (nullptr if no scalar data)
   */
  auto get_scalar_storage() const -> RocksDBStorage<IDType> * {
    if constexpr (has_scalar_data) {
      return scalar_storage_.get();
    }
    return nullptr;
  }

  /**
   * @brief Prefetch data into cache by ID to optimize memory access
   * @param id The ID of the data point to prefetch
   */
  auto prefetch_by_id(IDType id) -> void {  // for vertex
    // quant_codes_offset_ = rvec_len;
    mem_prefetch_l1(get_data_by_id(id), quant_codes_offset_ / 64);
  }

  /**
   * @brief Prefetch data into cache by address to optimize memory access
   * @param address The address of the data to prefetch
   */
  auto prefetch_by_address(DataType *address) -> void {  // for query
    // quant_codes_offset_ = rvec_len;
    mem_prefetch_l1(address, quant_codes_offset_ / 64);
  }

  auto rotate_vec(const DataType *src, DataType *dst) const { rotator_->rotate(src, dst); }

  auto get_padded_dim() const -> size_t { return rotator_->size(); }

  auto get_capacity() const -> size_t { return capacity_; }

  auto get_dim() const -> uint32_t { return dim_; }

  auto get_dist_func() const -> DistFuncRaBitQ<DataType, DistanceType> {
    return distance_cal_func_;
  }

  auto get_data_num() const -> IDType { return item_cnt_; }

  // no use
  auto get_data_size() const -> size_t { return data_chunk_size_; }

  auto get_query_computer(const DataType *query) const { return QueryComputer(*this, query); }

  auto get_query_computer(const IDType id) const {
    return QueryComputer(*this, get_data_by_id(id));
  }

  struct QueryComputer {
   private:
    // Cached hot-path data (avoid distance_space_ indirection chain)
    const char *storage_ptr_;                           ///< Raw storage data pointer
    size_t data_chunk_size_;                            ///< Node chunk size in bytes
    size_t qc_offset_;                                  ///< Quantization codes offset
    size_t f_add_offset_;                               ///< f_add offset
    size_t f_rescale_offset_;                           ///< f_rescale offset
    size_t nei_id_offset_;                              ///< Neighbor ID offset
    DistFuncRaBitQ<DataType, DistanceType> dist_func_;  ///< Distance function
    uint32_t dim_;                                      ///< Original dimension
    size_t padded_dim_;                                 ///< Padded dimension

    const DataType *query_;
    IDType c_;

    Lut<DataType> lookup_table_;

    DataType g_add_ = 0;
    DataType g_k1xsumq_ = 0;

    alignas(64) std::array<uint16_t, fastscan::kBatchSize> accu_res_{};
    alignas(64) std::array<DataType, kDegreeBound> est_dists_{};

    void batch_est_dist() {
      const char *base = storage_ptr_ + data_chunk_size_ * c_;
      const uint8_t *ALAYA_RESTRICT qc_ptr = reinterpret_cast<const uint8_t *>(base + qc_offset_);
      const DataType *ALAYA_RESTRICT f_add_ptr =
          reinterpret_cast<const DataType *>(base + f_add_offset_);
      const DataType *ALAYA_RESTRICT f_rescale_ptr =
          reinterpret_cast<const DataType *>(base + f_rescale_offset_);
      DataType *ALAYA_RESTRICT est_ptr = est_dists_.data();

      // look up, get sum(nth_segment)
      fastscan::accumulate(qc_ptr, lookup_table_.lut(), accu_res_.data(), padded_dim_);

      ConstRowMajorArrayMap<uint16_t> n_th_segment_arr(accu_res_.data(), 1, fastscan::kBatchSize);
      ConstRowMajorArrayMap<DataType> f_add_arr(f_add_ptr, 1, fastscan::kBatchSize);
      ConstRowMajorArrayMap<DataType> f_rescale_arr(f_rescale_ptr, 1, fastscan::kBatchSize);

      RowMajorArrayMap<DistDataType> est_dist_arr(est_ptr, 1, fastscan::kBatchSize);
      est_dist_arr =
          f_add_arr + g_add_ +
          (f_rescale_arr * (lookup_table_.delta() * (n_th_segment_arr.template cast<DataType>()) +
                            lookup_table_.sum_vl() + g_k1xsumq_));
    }

   public:
    QueryComputer() = default;
    ~QueryComputer() = default;

    // delete all since distance space is not allowed to be copied or moved.
    QueryComputer(QueryComputer &&) = delete;
    auto operator=(QueryComputer &&) -> QueryComputer & = delete;
    QueryComputer(const QueryComputer &) = delete;
    auto operator=(const QueryComputer &) -> QueryComputer & = delete;

    QueryComputer(const RaBitQSpace &distance_space, const DataType *query)
        : storage_ptr_(reinterpret_cast<const char *>(distance_space.storage_.data())),
          data_chunk_size_(distance_space.data_chunk_size_),
          qc_offset_(distance_space.quant_codes_offset_),
          f_add_offset_(distance_space.f_add_offset_),
          f_rescale_offset_(distance_space.f_rescale_offset_),
          nei_id_offset_(distance_space.nei_id_offset_),
          dist_func_(distance_space.distance_cal_func_),
          dim_(distance_space.dim_),
          padded_dim_(distance_space.get_padded_dim()),
          query_(query) {
      // rotate query vector
      std::vector<DataType> rotated_query(padded_dim_);
      distance_space.rotate_vec(query, rotated_query.data());

      lookup_table_ = std::move(Lut<DataType>(rotated_query.data(), padded_dim_));

      constexpr float c_1 = -((1 << 1) - 1) / 2.F;  // -0.5F NOLINT

      auto sumq = std::accumulate(rotated_query.begin(),
                                  rotated_query.begin() + padded_dim_,
                                  static_cast<DataType>(0));

      g_k1xsumq_ = sumq * c_1;
    }

    void load_centroid(IDType c) {
      c_ = c;

      const char *base = storage_ptr_ + data_chunk_size_ * c_;
      const auto *centroid_vec = reinterpret_cast<const DataType *>(base);
      g_add_ = dist_func_(query_, centroid_vec, dim_);

      batch_est_dist();
    }

    [[nodiscard]] auto get_edges() const -> const IDType * {
      const char *base = storage_ptr_ + data_chunk_size_ * c_;
      return reinterpret_cast<const IDType *>(base + nei_id_offset_);
    }

    auto get_exact_qr_c_dist() const -> DataType { return g_add_; }

    [[nodiscard]] auto est_data() const -> const DataType * { return est_dists_.data(); }

    /**
     * @brief Pass neighbors' index in centroid's edges instead of neighbors' id to avoid using
     * unordered_map
     *
     * @param i_th centroid's neighbors' index
     * @return DistanceType
     */
    auto operator()(size_t i_th) const -> DistanceType { return est_dists_[i_th]; }
  };

  auto save(std::string_view filename) -> void {
    std::ofstream writer(std::string(filename), std::ios::binary);
    if (!writer.is_open()) {
      throw std::runtime_error("Cannot open file " + std::string(filename));
    }

    writer.write(reinterpret_cast<char *>(&metric_), sizeof(metric_));
    writer.write(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    writer.write(reinterpret_cast<char *>(&item_cnt_), sizeof(item_cnt_));
    writer.write(reinterpret_cast<char *>(&capacity_), sizeof(capacity_));
    writer.write(reinterpret_cast<char *>(&type_), sizeof(type_));
    writer.write(reinterpret_cast<char *>(&ep_), sizeof(ep_));
    // no need to save offsets, we will take care of that in loading

    if constexpr (has_scalar_data) {
      save_scalar_config(writer);
    }

    rotator_->save(writer);

    storage_.save(writer);

    quantizer_->save(writer);

    LOG_INFO("RaBitQSpace is successfully saved to {}.", filename);
  }

  auto load(std::string_view filename) -> void {
    std::ifstream reader(std::string(filename), std::ios::binary);  // NOLINT

    if (!reader.is_open()) {
      throw std::runtime_error("Cannot open file " + std::string(filename));
    }

    reader.read(reinterpret_cast<char *>(&metric_), sizeof(metric_));
    reader.read(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    reader.read(reinterpret_cast<char *>(&item_cnt_), sizeof(item_cnt_));
    reader.read(reinterpret_cast<char *>(&capacity_), sizeof(capacity_));
    reader.read(reinterpret_cast<char *>(&type_), sizeof(type_));
    reader.read(reinterpret_cast<char *>(&ep_), sizeof(ep_));

    if constexpr (has_scalar_data) {
      load_scalar_config(reader);
      scalar_storage_ = std::make_unique<RocksDBStorage<IDType>>(config_);
    }

    rotator_ = choose_rotator<DataType>(dim_, type_, alaya::math::round_up_pow2<size_t>(dim_, 64));
    rotator_->load(reader);

    this->initialize_offsets();

    storage_ = StaticStorage<>(std::vector<size_t>{item_cnt_, data_chunk_size_});
    storage_.load(reader);

    quantizer_ = std::make_unique<RaBitQQuantizer<DataType>>();
    quantizer_->load(reader);

    LOG_INFO("RaBitQSpace is successfully loaded from {}", filename);
  }

  /**
   * @brief Close the RocksDB storage explicitly
   */
  void close_db() {
    if constexpr (has_scalar_data) {
      if (scalar_storage_ != nullptr) {
        scalar_storage_->flush();
        scalar_storage_.reset();
      }
    }
  }
};

template <typename T>
struct is_rabitq_space : std::false_type {};  // NOLINT

template <typename T, typename U, typename V, typename W>
struct is_rabitq_space<RaBitQSpace<T, U, V, W>> : std::true_type {};

template <typename T>
inline constexpr bool is_rabitq_space_v = is_rabitq_space<T>::value;  // NOLINT
}  // namespace alaya
