// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

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
