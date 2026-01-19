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

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace alaya {

// ==========================================
// 1. Type Definition (Basic Components)
// ==========================================

const uint32_t kAlignment = 64;

template <typename DataType, typename DistanceType>
using DistFunc = DistanceType (*)(const DataType *, const DataType *, size_t);

template <typename DataType, typename DistanceType>
using DistFuncSQ =
    DistanceType (*)(const uint8_t *, const uint8_t *, size_t, const DataType *, const DataType *);

template <typename DataType, typename DistanceType>
using DistFuncRaBitQ = DistanceType (*)(const DataType *, const DataType *, size_t);

// ==========================================
// 2. core Vector Space Concept
// ==========================================

/**
 * @brief Core concept of a vector space
 * * This is an aggregate concept that describes all the behaviors a complete vector space must
 * have.
 */
template <typename T>
concept Space = requires(T t,
                         const T ct,
                         const typename T::DataTypeAlias *data,
                         typename T::IDTypeAlias id) {
  { t.get_dim() } -> std::convertible_to<size_t>;
  { t.get_data_size() } -> std::convertible_to<size_t>;
  { t.get_capacity() } -> std::convertible_to<typename T::IDTypeAlias>;
  { t.get_data_num() } -> std::convertible_to<typename T::IDTypeAlias>;
  { t.get_distance(id, id) } -> std::common_with<typename T::DistanceTypeAlias>;
  { t.fit(data, id) } -> std::same_as<void>;
  { t.set_metric_function() } -> std::same_as<void>;

  // --- 3. Internal function pointer exposure (check if it supports a certain distance function
  // interface) --- Here we use disjunction (||) logic to check if it returns one of the function
  // pointers we support
  requires std::same_as<decltype(t.get_dist_func()),
                        DistFunc<typename T::DataTypeAlias, typename T::DistanceTypeAlias>> ||
               std::same_as<decltype(t.get_dist_func()),
                            DistFuncSQ<typename T::DataTypeAlias, typename T::DistanceTypeAlias>> ||
               std::same_as<
                   decltype(t.get_dist_func()),
                   DistFuncRaBitQ<typename T::DataTypeAlias, typename T::DistanceTypeAlias>>;
};

}  // namespace alaya
