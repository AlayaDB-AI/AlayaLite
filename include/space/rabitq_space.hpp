#pragma once

#include <sys/types.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "space/distance/dist_l2.hpp"
#include "space/quant/rabitq.hpp"
#include "space/space_concepts.hpp"
#include "storage/sequential_storage.hpp"
#include "utils/log.hpp"
#include "utils/metric_type.hpp"
#include "utils/prefetch.hpp"
#include "utils/rbq_utils/fastscan.hpp"
#include "utils/rbq_utils/lut.hpp"
#include "utils/rbq_utils/rotator.hpp"

namespace alaya {
template <typename DataType = float, typename DistanceType = float, typename IDType = uint32_t,
          typename DataStorage = SequentialStorage<DataType, IDType>>
class RBQSpace {
 private:
  MetricType metric_{MetricType::L2};  ///< Metric type
  uint32_t dim_{0};                    ///< Dimensionality of the data points
  IDType item_cnt_{0};                 ///< Number of data points (nodes)
  IDType capacity_{0};                 ///< The maximum number of data points (nodes)

  size_t nei_quant_codes_offset_{0};
  size_t f_add_offset_{0};
  size_t f_rescale_offset_{0};
  size_t data_chunk_size_{0};  ///< Size of each node's data chunk

  DistFuncRBQ<DataType, DistanceType> distance_calu_func_;  ///< Distance calculation function
  DataStorage data_storage_;
  std::unique_ptr<RBQQuantizer<DataType>> quantizer_;

 public:
  using DataTypeAlias = DataType;
  using DistanceTypeAlias = DistanceType;
  using IDTypeAlias = IDType;
  using DistDataType = DataType;

  // if you change degree bound , you should consider changing the layout too
  constexpr static size_t kDegreeBound = 32;  ///< Out degree of each node (in final graph)

  RBQSpace() = default;
  ~RBQSpace() = default;

  RBQSpace(RBQSpace &&other) = delete;
  RBQSpace(const RBQSpace &other) = delete;
  auto operator=(const RBQSpace &) -> RBQSpace & = delete;
  auto operator=(RBQSpace &&) -> RBQSpace & = delete;

  auto insert(DataType *data) -> IDType {
    throw std::runtime_error("Insert operation is not supported yet!");
  }

  auto remove(IDType id) -> IDType {
    throw std::runtime_error("Remove operation is not supported yet!");
  }

  void set_metric_function() {
    switch (metric_) {
      case MetricType::L2:
        distance_calu_func_ = l2_sqr_rabitq;
        break;
      case MetricType::COS:
      case MetricType::IP:
        throw std::runtime_error("inner product or cosine is not supported yet!");
        break;
      default:
        throw std::runtime_error("invalid metric type.");
        break;
    }
  }

  RBQSpace(IDType capacity, size_t dim, MetricType metric,
           RotatorType type = RotatorType::FhtKacRotator)
      : capacity_(capacity), dim_(dim), metric_(metric) {
    quantizer_ = std::make_unique<RBQQuantizer<DataType>>(dim_, type);
    initialize();
    data_storage_.init(data_chunk_size_, capacity_);
  }

  void initialize() {
    // data layout: (for each node, degree_bound defines their final outdegree)
    // 1. its raw data vector
    // 2. its neighbors' quantization codes
    // 3. F_add , F_rescale : please refer to rbq.md for detailed information
    size_t rvec_len = dim_ * sizeof(DataType);
    size_t nei_quant_code_len = quantizer_->get_padded_dim() * kDegreeBound / 8;  // 1 b/dim code
    size_t f_add_len = kDegreeBound * sizeof(DataType);
    size_t f_rescale_len = kDegreeBound * sizeof(DataType);

    // byte
    nei_quant_codes_offset_ = rvec_len;
    f_add_offset_ = nei_quant_codes_offset_ + nei_quant_code_len;
    f_rescale_offset_ = f_add_offset_ + f_add_len;
    data_chunk_size_ = f_rescale_offset_ + f_rescale_len;

    set_metric_function();
  }

  /**
   * @brief Update quantization code and related data based on neighbors' id
   *
   * @param c Id of the centroid
   * @param c_edges centroid's neighbor
   */
  void update_batch_data(IDType c, const IDType *c_edges) {
    // get neighbors' data
    std::vector<DataType> neighbors;
    neighbors.reserve(kDegreeBound * dim_);
    for (int i = 0; i < kDegreeBound; ++i) {
      auto nei = get_data_by_id(*(c_edges + i));
      neighbors.insert(neighbors.end(), nei, nei + dim_);
    }

    quantizer_->batch_quantize(neighbors.data(), get_data_by_id(c), kDegreeBound, get_nei_qc_ptr(c),
                               get_f_add_ptr(c), get_f_rescale_ptr(c));
  }

  void fit(DataType *data, IDType item_cnt) {
    static_assert(std::is_floating_point_v<DataType>, "Data type must be an floating type!");
    if (item_cnt > capacity_) {
      throw std::runtime_error("The number of data points exceeds the capacity of the space");
    }
    item_cnt_ = item_cnt;
    for (int i = 0; i < item_cnt; i++) {
      data_storage_.point_insert(data + (i * dim_), dim_);
    }
  }

  auto get_distance(IDType i, IDType j) -> DistanceType {
    return distance_calu_func_(get_data_by_id(i), get_data_by_id(j), dim_);
  }

  // get raw data vector
  [[nodiscard]] auto get_data_by_id(IDType i) const -> const DataType * {
    return reinterpret_cast<DataType *>(data_storage_[i]);
  }

  // get neighbors' quantization codes pointer
  [[nodiscard]] auto get_nei_qc_ptr(IDType i) const -> const uint8_t * {
    return reinterpret_cast<const uint8_t *>(data_storage_[i]) + nei_quant_codes_offset_;
  }

  [[nodiscard]] auto get_nei_qc_ptr(IDType i) -> uint8_t * {
    return reinterpret_cast<uint8_t *>(data_storage_[i]) + nei_quant_codes_offset_;
  }

  // get f_add pointer
  [[nodiscard]] auto get_f_add_ptr(IDType i) const -> const DataType * {
    return reinterpret_cast<const DataType *>(reinterpret_cast<char *>(data_storage_[i]) +
                                              f_add_offset_);
  }

  [[nodiscard]] auto get_f_add_ptr(IDType i) -> DataType * {
    return reinterpret_cast<DataType *>(reinterpret_cast<char *>(data_storage_[i]) + f_add_offset_);
  }

  // get f_rescale pointer
  [[nodiscard]] auto get_f_rescale_ptr(IDType i) const -> const DataType * {
    return reinterpret_cast<const DataType *>(reinterpret_cast<char *>(data_storage_[i]) +
                                              f_rescale_offset_);
  }

  [[nodiscard]] auto get_f_rescale_ptr(IDType i) -> DataType * {
    return reinterpret_cast<DataType *>(reinterpret_cast<char *>(data_storage_[i]) +
                                        f_rescale_offset_);
  }

  /**
   * @brief Prefetch data into cache by ID to optimize memory access
   * @param id The ID of the data point to prefetch
   */
  auto prefetch_by_id(IDType id) -> void {  // for vertex
    // nei_quant_codes_offset_ = rvec_len;
    mem_prefetch_l1(get_data_by_id(id), nei_quant_codes_offset_ / 64);
  }

  /**
   * @brief Prefetch data into cache by address to optimize memory access
   * @param address The address of the data to prefetch
   */
  auto prefetch_by_address(DataType *address) -> void {  // for query
    // nei_quant_codes_offset_ = rvec_len;
    mem_prefetch_l1(address, nei_quant_codes_offset_ / 64);
  }

  auto rotate_vec(const DataType *src, DataType *dst) const { quantizer_->rotate_vec(src, dst); }

  auto get_padded_dim() const -> size_t { return quantizer_->get_padded_dim(); }

  auto get_capacity() const -> size_t { return capacity_; }

  auto get_dim() const -> uint32_t { return dim_; }

  auto get_dist_func() const -> DistFuncRBQ<DataType, DistanceType> { return distance_calu_func_; }

  auto get_data_num() const -> IDType { return item_cnt_; }

  // no use
  auto get_data_size() const -> size_t { return data_chunk_size_; }

  auto get_query_computer(const DataType *query) const { return QueryComputer(*this, query); }

  struct QueryComputer {
   private:
    const RBQSpace &distance_space_;
    const DataType *query_;
    const IDType *edges_;
    IDType c_;

    Lut<DataType> lookup_table_;

    DataType g_add_ = 0;
    DataType g_k1xsumq_ = 0;

    std::vector<DataType> est_dists_;

    void batch_est_dist() {
      std::vector<u_int16_t> accu_res(fastscan::kBatchSize);
      size_t padded_dim = distance_space_.get_padded_dim();
      fastscan::accumulate(distance_space_.get_nei_qc_ptr(c_), lookup_table_.lut(), accu_res.data(),
                           padded_dim);

      ConstRowMajorArrayMap<u_int16_t> ip_arr(accu_res.data(), 1, fastscan::kBatchSize);
      ConstRowMajorArrayMap<DataType> f_add_arr(distance_space_.get_f_add_ptr(c_), 1,
                                                fastscan::kBatchSize);
      ConstRowMajorArrayMap<DataType> f_rescale_arr(distance_space_.get_f_rescale_ptr(c_), 1,
                                                    fastscan::kBatchSize);

      
      RowMajorArrayMap<DistDataType> est_dist_arr(est_dists_.data(), 1, fastscan::kBatchSize);
      est_dist_arr = f_add_arr + g_add_ +
                     f_rescale_arr * (lookup_table_.delta() * (ip_arr.template cast<DataType>()) +
                                      lookup_table_.sum_vl() + g_k1xsumq_);
    }

   public:
    QueryComputer() = default;
    ~QueryComputer() = default;

    // delete all since distance space is not allowed to be copied or moved either
    QueryComputer(QueryComputer &&) = delete;
    auto operator=(QueryComputer &&) -> QueryComputer & = delete;
    QueryComputer(const QueryComputer &) = delete;
    auto operator=(const QueryComputer &) -> QueryComputer & = delete;

    /// todo: align?
    QueryComputer(const RBQSpace &distance_space, const DataType *query)
        : distance_space_(distance_space), query_(query) {
      // rotate query vector
      size_t padded_dim = distance_space_.get_padded_dim();
      std::vector<DataType> rotated_query(padded_dim);
      distance_space_.rotate_vec(query, rotated_query.data());

      lookup_table_ = std::move(Lut<DataType>(rotated_query.data(), padded_dim));

      float c_1 = -((1 << 1) - 1) / 2.F;
      auto sumq = std::accumulate(rotated_query.begin(), rotated_query.begin() + padded_dim,
                                  static_cast<DataType>(0));
      g_k1xsumq_ = sumq * c_1;

      est_dists_.resize(RBQSpace<>::kDegreeBound);
    }

    void load_centroid(IDType c, const IDType *edges) {
      c_ = c;
      edges_ = edges;

      auto centroid_vec = distance_space_.get_data_by_id(c_);  // len: dim, not padded_dim
      g_add_ = distance_space_.get_dist_func()(query_, centroid_vec, distance_space_.get_dim());

      batch_est_dist();
    }

    auto get_exact_qr_c_dist() const -> DataType { return g_add_; }

    /**
     * @brief Pass neighbors' index in centroid's edges instead of neighbors' id to avoid using unordered_map
     * 
     * @param i_th centroid's neighbors' index
     * @return DistanceType 
     */
    auto operator()(int i_th) const -> DistanceType { return est_dists_[i_th]; }
  };

  auto save(std::string_view &filename) -> void {
    std::ofstream writer(std::string(filename), std::ios::binary);
    if (!writer.is_open()) {
      throw std::runtime_error("Cannot open file " + std::string(filename));
    }

    writer.write(reinterpret_cast<char *>(&metric_), sizeof(metric_));
    writer.write(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    writer.write(reinterpret_cast<char *>(&item_cnt_), sizeof(item_cnt_));
    writer.write(reinterpret_cast<char *>(&capacity_), sizeof(capacity_));

    data_storage_.save(writer);

    quantizer_->save(writer);

    LOG_INFO("RBQSpace is successfully saved to {}.", filename);
  }

  auto load(std::string_view &filename) -> void {
    std::ifstream reader(filename.data(), std::ios::binary);  // NOLINT

    if (!reader.is_open()) {
      throw std::runtime_error("Cannot open file " + std::string(filename));
    }

    reader.read(reinterpret_cast<char *>(&metric_), sizeof(metric_));
    reader.read(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    reader.read(reinterpret_cast<char *>(&item_cnt_), sizeof(item_cnt_));
    reader.read(reinterpret_cast<char *>(&capacity_), sizeof(capacity_));

    // no need for init() after loading
    data_storage_.load(reader);

    quantizer_ = std::make_unique<RBQQuantizer<DataType>>();
    quantizer_->load(reader);

    this->initialize();
    LOG_INFO("RBQSpace is successfully loaded from {}", filename);
  }
};
}  // namespace alaya