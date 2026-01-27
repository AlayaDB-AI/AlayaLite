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
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include "../utils/prefetch.hpp"
#include "simd/distance_ip.hpp"
#include "simd/distance_l2.hpp"
#include "space_concepts.hpp"
#include "storage/sequential_storage.hpp"
#include "utils/data_utils.hpp"
#include "utils/log.hpp"
#include "utils/math.hpp"
#include "utils/metric_type.hpp"
#include "utils/platform.hpp"
#include "utils/prefetch.hpp"

namespace alaya {

/**
 * @brief The RawSpace class for managing vector search, insert, delete and distance calculation.
 *
 * This class provides functionality for storing and managing data points in a space,
 * as well as computing distances between points.
 *
 * @tparam DataType The data type for storing data points, with the default being float.
 * @tparam DistanceType The data type for storing distances, with the default being float.
 * @tparam IDType The data type for storing IDs, with the default being uint32_t.
 */
template <typename DataType = float,
          typename DistanceType = float,
          typename IDType = uint32_t,
          typename DataStorage = SequentialStorage<DataType, IDType>>
class RawSpace {
 public:
  using DistDataType = DataType;  ///< Type alias for the data type used in distance calculations
  using DataTypeAlias = DataType;
  using IDTypeAlias = IDType;
  using DistanceTypeAlias = DistanceType;

  DistFunc<DistDataType, DistanceType> distance_calu_func_;  ///< Distance calculation function

  IDType capacity_{0};                 ///< The maximum number of data points (nodes)
  uint32_t dim_{0};                    ///< Dimensionality of the data points
  MetricType metric_{MetricType::L2};  ///< Metric type
  uint32_t data_size_{0};              ///< Size of each data point in bytes
  IDType item_cnt_{0};    ///< Number of data points (nodes), can be either, available or deleted
  IDType delete_cnt_{0};  ///< Number of deleted data points
  DataStorage data_storage_;  ///< Data storage

 public:
  RawSpace() = default;

  /**
   * @brief Construct a new RawSpace object.
   *
   * @param data Pointer to the input data array
   * @param node_num Number of data points
   * @param dim Dimensionality of each data point
   */
  RawSpace(IDType capacity, size_t dim, MetricType metric)
      : capacity_(capacity), dim_(dim), metric_(metric) {
    data_size_ = dim * sizeof(DataType);
    distance_calu_func_ = simd::l2_sqr<DataType, DistanceType>;  // Assign the distance function

    data_storage_.init(data_size_, capacity);

    if constexpr (!(std::is_same_v<DataType, float> || std::is_same_v<DataType, double>)) {
      if (metric_ == MetricType::COS) {
        LOG_ERROR("COS metric only support float or double");
        exit(-1);
      }
    }

    set_metric_function();
  }

  /**
   * @brief Move constructor
   */
  RawSpace(RawSpace &&other) = delete;
  RawSpace(const RawSpace &other) = delete;

  /**
   * @brief Destructor
   */
  ~RawSpace() = default;

  /**
   * @brief Set the distance calculation function based on the metric type
   */
  void set_metric_function() {
    switch (metric_) {
      case MetricType::L2:
        distance_calu_func_ = simd::l2_sqr<DataType, DistanceType>;
        break;
      case MetricType::IP:
      case MetricType::COS:
        distance_calu_func_ = simd::ip_sqr<DataType, DistanceType>;
        break;
      default:
        break;
    }
  }

  /**
   * @brief Fit the data into the space
   * @param data Pointer to the input data array, no padding between data points
   * @param item_cnt Number of data points
   */
  void fit(const DataType *data, IDType item_cnt) {
    item_cnt_ = item_cnt;
    for (IDType i = 0; i < item_cnt_; ++i) {
      // if the metric is cosine, normalize the query
      if (metric_ == MetricType::COS) {
        normalize(const_cast<DataType *>(data + (i * dim_)), dim_);
      }
      data_storage_.insert(data + (i * dim_));
    }
  }

  /**
   * @brief Insert a data point into the space
   * @param data Pointer to the data point
   */
  auto insert(const DataType *data) -> IDType {
    // if the metric is cosine, normalize the query
    if (metric_ == MetricType::COS) {
      normalize(const_cast<DataType *>(data), dim_);
    }
    item_cnt_++;
    return data_storage_.insert(data);
  }

  /**
   * @brief Delete a data point by its ID
   *
   * @param id the id of the data point to delete
   */
  auto remove(IDType id) -> IDType {
    delete_cnt_++;
    return data_storage_.remove(id);
  }

  /**
   * @brief Get the data pointer for a specific ID
   * @param id The ID of the data point
   * @return Pointer to the data for the given ID
   */
  auto get_data_by_id(IDType id) const -> DataType * { return data_storage_[id]; }

  /**
   * @brief Calculate the distance between two data points
   * @param i ID of the first data point
   * @param j ID of the second data point
   * @return The calculated distance
   */
  auto get_distance(IDType i, IDType j) -> DistanceType {
    return distance_calu_func_(get_data_by_id(i), get_data_by_id(j), dim_);
  }

  /**
   * @brief Get the number of the vector data
   * @return The number of vector data.
   */
  auto get_data_num() -> IDType { return item_cnt_; }

  /**
   * @brief Get the number of the available vector data
   * @return The number of vector data.
   */
  auto get_avl_data_num() -> IDType { return item_cnt_ - delete_cnt_; }

  /**
   * @brief Get the capacity object
   *
   * @return IDType The capacity of the space.
   */
  auto get_capacity() -> IDType { return capacity_; }

  /**
   * @brief Get the size of each data point in bytes
   * @return The size of each data point
   */
  auto get_data_size() -> size_t { return data_size_; }

  /**
   * @brief Get the distance calculation function
   * @return The distance calculation function
   */
  auto get_dist_func() -> DistFunc<DataType, DistanceType> { return distance_calu_func_; }

  /**
   * @brief Get the dimensionality of the data points
   * @return The dimensionality
   */
  auto get_dim() -> uint32_t { return dim_; }

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
    LOG_INFO("RawSpace is loaded from {}", filename);
  }

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
    LOG_INFO("RawSpace is saved to {}", filename);
  }

  /**
   * @brief Nested structure for efficient query computation
   */
  struct QueryComputer {
    const RawSpace &distance_space_;
    DataType *query_ = nullptr;

    /**
     * @brief Construct a new QueryComputer object
     * @param distance_space Reference to the RawSpace
     * @param query Pointer to the query data
     */
    QueryComputer(const RawSpace &distance_space, const DataType *query)
        : distance_space_(distance_space) {
      // if the metric is cosine, normalize the query
      if (distance_space_.metric_ == MetricType::COS) {
        normalize(const_cast<DataType *>(query), distance_space_.dim_);
      }

      auto aligned_size = math::round_up_pow2(distance_space_.data_size_, kAlignment);
      query_ = static_cast<DataType *>(alaya_aligned_alloc_impl(aligned_size, kAlignment));
      std::memcpy(query_, query, distance_space.data_size_);
    }

    QueryComputer(const RawSpace &distance_space, const IDType id)
        : distance_space_(distance_space) {
      size_t aligned_size = math::round_up_pow2(distance_space_.data_size_, kAlignment);
      query_ = static_cast<DataType *>(alaya_aligned_alloc_impl(aligned_size, kAlignment));
      std::memcpy(query_, distance_space.get_data_by_id(id), distance_space.data_size_);
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
      if (!distance_space_.data_storage_.is_valid(u)) {
        return std::numeric_limits<float>::max();
      }
      return distance_space_.distance_calu_func_(query_,
                                                 distance_space_.get_data_by_id(u),
                                                 distance_space_.dim_);
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

  auto get_query_computer(IDType id) { return QueryComputer(*this, id); }
};

// use static assert to check if RawSpace satisfies the Space concept
//                           data_type distance_type id_type data_storage
static_assert(Space<RawSpace<uint32_t, float, uint32_t>>);
static_assert(Space<RawSpace<uint32_t, float, uint64_t>>);
static_assert(Space<RawSpace<uint64_t, float, uint32_t>>);
static_assert(Space<RawSpace<uint64_t, float, uint64_t>>);
static_assert(Space<RawSpace<float, float, uint32_t>>);
static_assert(Space<RawSpace<float, float, uint64_t>>);
static_assert(Space<RawSpace<double, float, uint32_t>>);
static_assert(Space<RawSpace<double, float, uint64_t>>);

}  // namespace alaya
