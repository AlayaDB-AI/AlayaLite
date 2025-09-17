/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <sys/types.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace alaya {

const uint32_t kAlignment = 64;  ///< Constant for alignment (cache line / SIMD)


template <typename DataType, typename DistanceType>
using DistFuncRBQ = DistanceType (*)(const DataType *,const DataType *, size_t);

/**
 * @brief Type alias for a distance function pointer.
 *
 * This alias defines a function pointer type for distance calculation functions.
 *
 * @tparam DataType The data type for the input data points.
 * @tparam DistanceType The data type for the calculated distance.
 */
template <typename DataType, typename DistanceType>
using DistFunc = DistanceType (*)(DataType *, DataType *, size_t);

/**
 * @brief Type alias for a distance function pointer for SQ4 or SQ8.
 *
 * This alias defines a function pointer type for distance calculation functions.
 *
 * @tparam DataType The data type for the input data points.
 * @tparam DistanceType The data type for the calculated distance.
 */
template <typename DataType, typename DistanceType>
using DistFuncSQ = DistanceType (*)(const uint8_t *, const uint8_t *, size_t, const DataType *min,
                                    const DataType *max);

/**
 * @brief Concept to check if type T has a set_metric_function method.
 *
 * This concept ensures that T provides a method to set the metric function.
 *
 * @tparam T The type to be checked.
 */
template <typename T>
concept HasMertricFunc = (requires(T t) {
  { t.set_metric_function() } -> std::same_as<void>;
});

/**
 * @brief Concept to check if type T has a fit method.
 *
 * This concept ensures that T provides a method to fit data points into the space.
 *
 * @tparam T The type to be checked.
 * @tparam DataType The data type for the input data points.
 * @tparam IDType The data type for the IDs of the data points.
 */
template <typename T, typename DataType, typename IDType>
concept HasFitFn = (requires(T t, DataType *data, IDType item_cnt) {
  { t.fit(data, item_cnt) } -> std::same_as<void>;
});

/**
 * @brief Concept to check if type T has a member function get_data_size.
 *
 * This concept ensures that T provides a method to retrieve the size of its data,
 * returning a value of type size_t.
 */
template <typename T>
concept HasGetDataSize = (requires(T t) {
  { t.get_data_size() } -> std::same_as<size_t>;
});

/**
 * @brief Concept to check if type T has a member function get_dist_func.
 *
 * This concept ensures that T provides a method to retrieve a distance function,
 * which should match the DistFunc type for the specified DistanceType.
 */
template <typename T, typename DataType, typename DistanceType>
concept HasGetDistFunc = (requires(T t) {
  { t.get_dist_func() } -> std::same_as<DistFunc<DataType, DistanceType>>;
});

/**
 * @brief Concept to check if type T has a member function get_dist_func.
 *
 * This concept ensures that T provides a method to retrieve a distance function,
 * which should match the DistFuncSQ (for SQ4 or SQ8) type for the specified DistanceType.
 */
template <typename T, typename DataType, typename DistanceType>
concept HasGetDistFuncSQ = (requires(T t) {
  // { t.get_dist_func() } -> std::same_as<DistFuncSQ<DataType, DistanceType>>;
  { t.get_dist_func() } -> std::same_as<DistFuncSQ<DataType, DistanceType>>;
  // true;
});

template <typename T, typename DataType, typename DistanceType>
concept HasGetDistFuncRBQ = (requires(T t) {
  { t.get_dist_func() } -> std::same_as<DistFuncRBQ<DataType, DistanceType>>;
});

/**
 * @brief Concept to check if type T has a member function get_distance.
 *
 * This concept ensures that T implements a method to calculate the distance
 * between two elements, taking parameters of type IDType and returning a floating-point value.
 */
template <typename T, typename IDType>
concept HasGetDistance = (requires(T t, IDType i, IDType j) {
  { t.get_distance(i, j) } -> std::floating_point;
} || (requires(T t, IDType i, IDType j) {
                            { t.get_distance(i, j) } -> std::integral;
                          }));

/**
 * @brief Comprehensive interface concept for a distance space.
 *
 * This concept requires type T to implement methods for obtaining data size,
 * calculating distances, and retrieving the distance function, ensuring
 * unified distance computation across different types.
 */
template <typename T, typename DataType, typename DistanceType, typename IDType>
concept Space = HasGetDataSize<T> && HasGetDistance<T, IDType> &&
                (HasGetDistFunc<T, typename T::DistDataType, DistanceType> ||
                 HasGetDistFuncSQ<T, typename T::DistDataType, DistanceType>||
                HasGetDistFuncRBQ<T, typename T::DistDataType, DistanceType>) &&
                HasFitFn<T, DataType, IDType> && HasMertricFunc<T>;

}  // namespace alaya
