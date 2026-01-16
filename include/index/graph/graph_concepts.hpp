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
#include <concepts>
#include <cstdint>
#include <memory>
#include "graph.hpp"

namespace alaya {

/**
 * @brief Concept that checks if a type T can build a graph.
 *
 * This concept ensures that any type T that is used with it has a member function
 * named `build_graph` which takes two parameters:
 * - A pointer to an array of data of type DataType (e.g., float* for vector data).
 * - An identifier of type IDType (e.g., an integer representing the number of vectors).
 *
 * @tparam T The type that is being constrained by this concept. It should have a
 *           member function `build_graph`.
 * @tparam IDType The type of the first parameter for the `build_graph` function,
 *                typically representing some identifier or size.
 */
template <class T, typename DataType = float, typename IDType = uint32_t>
concept HasBuildGraph = (requires(T t) {
  // Check that the member function build_graph exists and has the correct signature
  { t.build_graph() } -> std::same_as<std::unique_ptr<Graph<DataType, IDType>>>;
});

template <typename T>
concept GraphBuilder = HasBuildGraph<T,
                                     typename T::DistanceSpaceTypeAlias::DataTypeAlias,
                                     typename T::DistanceSpaceTypeAlias::IDTypeAlias>;

}  // namespace alaya
