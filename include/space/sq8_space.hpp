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
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include "simd/distance_ip.hpp"
#include "simd/distance_l2.hpp"
#include "space/quant/sq8.hpp"
#include "space_concepts.hpp"
#include "storage/rocksdb_storage.hpp"
#include "storage/sequential_storage.hpp"
#include "utils/log.hpp"
#include "utils/math.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/metric_type.hpp"
#include "utils/platform.hpp"
#include "utils/prefetch.hpp"

namespace alaya {

/**
 * @brief The SQ8Space class for managing distance calculations on 8-bit quantized data.
 *
 * This class provides functionality for storing and managing 8-bit quantized data points,
 * as well as computing distances between points.
 *
 * @tparam DataType The data type for storing raw data points, with the default being float.
 * @tparam DistanceType The data type for storing distances, with the default being float.
 * @tparam IDType The data type for storing IDs, with the default being uint32_t.
 * @tparam DataStorage The storage backend for vector data, with the default being
 * SequentialStorage.
 * @tparam ScalarDataType The data type for ScalarData, with the default being EmptyScalarData.
 */
template <typename DataType = float,
          typename DistanceType = float,
          typename IDType = uint32_t,
          typename DataStorage = SequentialStorage<uint8_t, IDType>,
          typename ScalarDataType = EmptyScalarData>
class SQ8Space {
 public:
  static constexpr bool has_scalar_data =
      !std::is_same_v<ScalarDataType, EmptyScalarData>;  // NOLINT

  using DataTypeAlias = DataType;
  using IDTypeAlias = IDType;
  using DistanceTypeAlias = DistanceType;

  using DistDataType = DataType;

 public:
  /**
   * @brief Construct an empty SQ8Space object without parameter for loading.
   *
   */
  SQ8Space() = default;

  /**
   * @brief Construct a new SQ8Space object.
   *
   * @param capacity The maximum number of data points (nodes)
   * @param dim Dimensionality of each data point
   * @param metric Metric type
   * @param config RocksDB configuration for scalardata storage
   */
  SQ8Space(IDType capacity,
           size_t dim,
           MetricType metric,
           RocksDBConfig config = RocksDBConfig::default_config())
      : capacity_(capacity),
        dim_(dim),
        metric_(metric),
        config_(std::move(config)),
        quantizer_(dim) {
    data_size_ = dim_ * sizeof(uint8_t);
    data_storage_.init(data_size_, capacity);
    set_metric_function();
  }

  ~SQ8Space() = default;

  SQ8Space(SQ8Space &&other) = delete;
  SQ8Space(const SQ8Space &other) = delete;
  auto operator=(const SQ8Space &) -> SQ8Space & = delete;
  auto operator=(SQ8Space &&) -> SQ8Space & = delete;

  /**
   * @brief Set the distance calculation function based on the metric type
   */
  void set_metric_function() {
    switch (metric_) {
      case MetricType::L2:
        distance_calu_func_ = simd::l2_sqr_sq8<DataType, DistanceType>;
        break;
      case MetricType::COS:
      case MetricType::IP:
        distance_calu_func_ = simd::ip_sqr_sq8<DataType, DistanceType>;
        break;
      default:
        break;
    }
  }

  /**
   * @brief Get the capacity of the space
   * @return The capacity
   */
  auto get_capacity() -> IDType { return capacity_; }

  /**
   * @brief Fit the data into the space
   * @param data Pointer to the input data array
   * @param item_cnt Number of data points
   * @param scalar_data Pointer to ScalarData array (optional)
   */
  void fit(const DataType *data, IDType item_cnt, const ScalarDataType *scalar_data = nullptr) {
    if (data == nullptr) {
      throw std::invalid_argument("Invalid or null vector data pointer.");
    }

    if (item_cnt > capacity_) {
      throw std::length_error("The number of data points exceeds the capacity of the space");
    }
    item_cnt_ = item_cnt;

    quantizer_.fit(data, item_cnt);
    for (IDType i = 0; i < item_cnt; i++) {
      auto id = data_storage_.reserve();
      quantizer_.encode(data + (i * dim_), data_storage_[id]);
    }

    // Store ScalarData with synchronized IDs (0, 1, 2, ...)
    if constexpr (has_scalar_data) {  // NOLINT
      if (scalar_data == nullptr) {
        throw std::invalid_argument("Invalid or null ScalarData pointer.");
      }
      if (scalar_storage_ == nullptr) {
        // otherwise existing ScalarData will lack corresponding vector data.
        // if you want to open a existing ScalarData db, try load() and then insert() your new data
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

  /**
   * @brief Get the encoded data pointer for a specific ID
   * @param id The ID of the data point
   * @return Pointer to the data for the given ID
   */
  auto get_data_by_id(IDType id) const -> uint8_t * { return data_storage_[id]; }

  /**
   * @brief Calculate the distance between two data points
   * @param i ID of the first data point
   * @param j ID of the second data point
   * @return The calculated distance
   */
  auto get_distance(IDType i, IDType j) -> DistanceType {
    return distance_calu_func_(get_data_by_id(i),
                               get_data_by_id(j),
                               dim_,
                               quantizer_.get_min(),
                               quantizer_.get_max());
  }

  /**
   * @brief Get the number of the vector data
   * @return The number of vector data.
   */
  auto get_data_num() -> IDType { return item_cnt_; }

  /**
   * @brief Get the size of each data point in bytes
   * @return The size of each data point
   */
  auto get_data_size() const -> size_t { return data_size_; }

  /**
   * @brief Get the distance calculation function
   * @return The distance calculation function
   */
  auto get_dist_func() -> DistFuncSQ<DataType, DistanceType> { return distance_calu_func_; }

  /**
   * @brief Get scalar data for a specific ID
   * @param id The ID of the data point
   * @return The scalar data for the given ID
   */
  auto get_scalar_data(IDType id) const -> ScalarDataType {
    if constexpr (has_scalar_data) {  // NOLINT
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
    if constexpr (has_scalar_data) {  // NOLINT
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
    if constexpr (has_scalar_data) {  // NOLINT
      return scalar_storage_->scan_with_filter(
          [&filter](const ScalarData &sd) {
            return filter.evaluate(sd.metadata);
          },
          limit);
    }
    throw std::runtime_error("No ScalarData available.");
  }

  /**
   * @brief Get the dimensionality of the data points
   * @return The dimensionality
   */
  auto get_dim() const -> uint32_t { return dim_; }

  /**
   * @brief Get the quantizer
   * @return quantizer
   */
  auto get_quantizer() const -> SQ8Quantizer<DataType> { return quantizer_; }

  /**
   * @brief Insert a data point into the space. The data point will be quantized and stored in the
   * space. The ID of the inserted data point will be returned.
   *
   * @param data Pointer to the data point to be inserted
   * @param scalar_data Pointer to ScalarData (optional, only used when ScalarDataType is not
   * EmptyScalardata)
   * @return IDType The ID of the inserted data point (-1 for failure)
   */
  auto insert(DataType *data, const ScalarDataType *scalar_data = nullptr) -> IDType {
    auto id = data_storage_.reserve();
    if (id == static_cast<IDType>(-1)) {
      return static_cast<IDType>(-1);
    }
    item_cnt_++;
    quantizer_.encode(data, data_storage_[id]);

    // Insert ScalarData with the same ID as vector
    if constexpr (has_scalar_data) {  // NOLINT
      if (scalar_data != nullptr && scalar_storage_ != nullptr) {
        if (!scalar_storage_->insert(id, *scalar_data)) {
          LOG_ERROR("Failed to insert ScalarData for ID {}", id);
          // Don't fail the whole operation, just log the error
        }
      }
    }

    return id;
  }

  /**
   * @brief Delete a data point by its ID. Currently, the data point will be marked as deleted, but
   * not exactly removed from the storage.
   *
   * @param id the ID of the data point to delete
   * @return IDType The ID of the deleted data point
   */
  auto remove(IDType id) -> IDType {
    delete_cnt_++;

    // Remove ScalarData if present
    if constexpr (has_scalar_data) {  // NOLINT
      if (scalar_storage_ != nullptr) {
        scalar_storage_->remove(id);
      }
    }

    return data_storage_.remove(id);
  }

  /**
   * @brief Remove a data point by its item_id
   * @param item_id The item_id to remove
   * @return The internal ID that was removed
   * @throws std::runtime_error if item_id not found
   */
  auto remove(const std::string &item_id) -> IDType {
    if constexpr (has_scalar_data) {  // NOLINT
      auto internal_id_opt = scalar_storage_->find_by_item_id(item_id);
      if (!internal_id_opt.has_value()) {
        throw std::runtime_error("Item ID not found: " + item_id);
      }
      return remove(internal_id_opt.value());  // Calls remove(IDType) above
    }
    throw std::runtime_error("No ScalarData available.");
  }

  /**
   * @brief Load the space from a file
   * @param filename The name of the file to load
   */
  auto load(std::string_view filename) -> void {
    std::ifstream reader(std::string(filename), std::ios::binary);

    if (!reader.is_open()) {
      throw std::runtime_error("Cannot open file " + std::string(filename));
    }

    reader.read(reinterpret_cast<char *>(&metric_), sizeof(metric_));
    reader.read(reinterpret_cast<char *>(&data_size_), sizeof(data_size_));
    reader.read(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    reader.read(reinterpret_cast<char *>(&item_cnt_), sizeof(item_cnt_));
    reader.read(reinterpret_cast<char *>(&delete_cnt_), sizeof(delete_cnt_));
    reader.read(reinterpret_cast<char *>(&capacity_), sizeof(capacity_));

    if constexpr (has_scalar_data) {  // NOLINT
      load_scalar_config(reader);
      scalar_storage_ = std::make_unique<RocksDBStorage<IDType>>(config_);
    }

    data_storage_.load(reader);
    quantizer_.load(reader);
    LOG_INFO("SQ8Space is loaded from {}", filename);
  }

  /**
   * @brief Save the space to a file
   * @param filename The name of the file to save
   */
  auto save(std::string_view filename) -> void {
    std::ofstream writer(std::string(filename), std::ios::binary);
    if (!writer.is_open()) {
      throw std::runtime_error("Cannot open file " + std::string(filename));
    }

    writer.write(reinterpret_cast<char *>(&metric_), sizeof(metric_));
    writer.write(reinterpret_cast<char *>(&data_size_), sizeof(data_size_));
    writer.write(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    writer.write(reinterpret_cast<char *>(&item_cnt_), sizeof(item_cnt_));
    writer.write(reinterpret_cast<char *>(&delete_cnt_), sizeof(delete_cnt_));
    writer.write(reinterpret_cast<char *>(&capacity_), sizeof(capacity_));

    if constexpr (has_scalar_data) {  // NOLINT
      save_scalar_config(writer);
    }

    data_storage_.save(writer);
    quantizer_.save(writer);
    LOG_INFO("SQ8Space is saved to {}", filename);
  }
  /**
   * @brief Nested structure for efficient query computation
   */
  struct QueryComputer {
    const SQ8Space &distance_space_;
    uint8_t *query_ = nullptr;

    /**
     * @brief Construct a new QueryComputer object
     * @param distance_space Reference to the RawSpace
     * @param query Pointer to the query data
     */
    QueryComputer(const SQ8Space &distance_space, const DataType *query)
        : distance_space_(distance_space) {
      size_t aligned_size = math::round_up_pow2(distance_space_.get_data_size(), 64);
      query_ = static_cast<uint8_t *>(alaya_aligned_alloc_impl(aligned_size, 64));
      distance_space.get_quantizer().encode(query, query_);
    }

    QueryComputer(const SQ8Space &distance_space, const IDType id)
        : distance_space_(distance_space) {
      size_t aligned_size = math::round_up_pow2(distance_space_.get_data_size(), 64);
      query_ = static_cast<uint8_t *>(alaya_aligned_alloc_impl(aligned_size, 64));
      std::memcpy(query_, distance_space_.get_data_by_id(id), distance_space_.get_data_size());
    }
    /**
     * @brief Destructor
     */
    ~QueryComputer() {
      if (query_ != nullptr) {
        alaya_aligned_free_impl(query_);
      }
    }

    /**
     * @brief Compute the distance between the query and a data point
     * @param u ID of the data point to compare with the query
     * @return The calculated distance
     */
    auto operator()(IDType u) const -> DistanceType {
      return distance_space_.distance_calu_func_(query_,
                                                 distance_space_.get_data_by_id(u),
                                                 distance_space_.get_dim(),
                                                 distance_space_.get_quantizer().get_min(),
                                                 distance_space_.get_quantizer().get_max());
    }
  };

  /**
   * @brief Prefetch data into cache by ID to optimize memory access
   * @param id The ID of the data point to prefetch
   */
  auto prefetch_by_id(IDType id) -> void { mem_prefetch_l1(get_data_by_id(id), data_size_ / 64); }

  /**
   * @brief Prefetch data into cache by address to optimize memory access
   * @param address The address of the data to prefetch
   */
  auto prefetch_by_address(DataType *address) -> void { mem_prefetch_l1(address, data_size_ / 64); }

  auto get_query_computer(const DataType *query) { return QueryComputer(*this, query); }

  auto get_query_computer(const IDType id) { return QueryComputer(*this, id); }

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

 private:
  IDType capacity_{0};                 ///< The maximum number of data points (nodes)
  uint32_t dim_{0};                    ///< Dimensionality of the data points
  MetricType metric_{MetricType::L2};  ///< Metric type

  DistFuncSQ<DataType, DistanceType> distance_calu_func_;  ///< Distance calculation function
  uint32_t data_size_{0};                                  ///< Size of each data point in bytes
  IDType item_cnt_{0};                                     ///< Number of data points (nodes)
  IDType delete_cnt_{0};              ///< Number of deleted data points (nodes)
  DataStorage data_storage_;          ///< Data storage for encoded data
  SQ8Quantizer<DataType> quantizer_;  ///< The quantizer used to quantize the data

  RocksDBConfig config_;  ///< Configuration for Scalar Data Storage
  std::unique_ptr<RocksDBStorage<IDType>>
      scalar_storage_;  ///< Scalar Data Storage (stores ScalarData)

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
  }

  void load_scalar_config(std::ifstream &reader) {
    config_.create_if_missing_ = false;  // db is missing means something went wrong
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
  }
};
}  // namespace alaya
