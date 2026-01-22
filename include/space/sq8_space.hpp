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
#include <string>
#include "distance/dist_ip.hpp"
#include "distance/dist_l2.hpp"
#include "space/quant/sq8.hpp"
#include "space_concepts.hpp"
#include "storage/sequential_storage.hpp"
#include "utils/metric_type.hpp"
#include "utils/prefetch.hpp"

#ifdef _MSC_VER
  #include <malloc.h>
#endif

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
 */
template <typename DataType = float,
          typename DistanceType = float,
          typename IDType = uint32_t,
          typename DataStorage = SequentialStorage<uint8_t, IDType>>
class SQ8Space {
 public:
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
   * @brief Construct a new RawSpace object.
   *
   * @param capacity The maximum number of data points (nodes)
   * @param dim Dimensionality of each data point
   * @param metirc Metric type
   */
  SQ8Space(IDType capacity, size_t dim, MetricType metric)
      : capacity_(capacity), dim_(dim), metric_(metric), quantizer_(dim) {
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
        distance_calu_func_ = l2_sqr_sq8;
        break;
      case MetricType::COS:
      case MetricType::IP:
        distance_calu_func_ = ip_sqr_sq8;
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
   */
  void fit(const DataType *data, IDType item_cnt) {
    if (item_cnt > capacity_) {
      throw std::runtime_error("The number of data points exceeds the capacity of the space");
    }
    item_cnt_ = item_cnt;

    quantizer_.fit(data, item_cnt);
    for (IDType i = 0; i < item_cnt; i++) {
      auto id = data_storage_.reserve();
      quantizer_.encode(data + (i * dim_), data_storage_[id]);
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
   * @return IDType The ID of the inserted data point (-1 for failure)
   */
  auto insert(DataType *data) -> IDType {
    auto id = data_storage_.reserve();
    if (id == static_cast<IDType>(-1)) {
      return static_cast<IDType>(-1);
    }
    item_cnt_++;
    quantizer_.encode(data, data_storage_[id]);
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
    return data_storage_.remove(id);
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
      size_t aligned_size = do_align(distance_space_.get_data_size(), 64);
#ifdef _MSC_VER
      query_ = static_cast<uint8_t *>(_aligned_malloc(aligned_size, 64));
#else
      query_ = static_cast<uint8_t *>(std::aligned_alloc(64, aligned_size));
#endif
      distance_space.get_quantizer().encode(query, query_);
    }

    QueryComputer(const SQ8Space &distance_space, const IDType id)
        : distance_space_(distance_space) {
      size_t aligned_size = do_align(distance_space_.get_data_size(), 64);
#ifdef _MSC_VER
      query_ = static_cast<uint8_t *>(_aligned_malloc(aligned_size, 64));
#else
      query_ = static_cast<uint8_t *>(std::aligned_alloc(64, aligned_size));
#endif
      std::memcpy(query_, distance_space_.get_data_by_id(id), distance_space_.get_data_size());
    }
    /**
     * @brief Destructor
     */
    ~QueryComputer() {
#ifdef _MSC_VER
      _aligned_free(query_);
#else
      std::free(query_);
#endif
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
};
}  // namespace alaya
