// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

#include "utils/query_utils.hpp"

namespace alaya {

/** @brief RocksDB-independent set operations used while compiling scalar filter plans. */
template <typename IDType>
struct ScalarIdSetAlgebra {
  /** @brief Intersect two sorted, unique ID sets. */
  [[nodiscard]] static auto intersect(const std::vector<IDType> &lhs,
                                      const std::vector<IDType> &rhs) -> std::vector<IDType> {
    std::vector<IDType> result;
    result.reserve(std::min(lhs.size(), rhs.size()));
    std::set_intersection(lhs.begin(),
                          lhs.end(),
                          rhs.begin(),
                          rhs.end(),
                          std::back_inserter(result));
    return result;
  }

  /** @brief Union two sorted, unique ID sets. */
  [[nodiscard]] static auto unite(const std::vector<IDType> &lhs, const std::vector<IDType> &rhs)
      -> std::vector<IDType> {
    std::vector<IDType> result;
    result.reserve(lhs.size() + rhs.size());
    std::set_union(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(result));
    return result;
  }

  /** @brief Build the complement of an ID set within [0, universe_size). */
  [[nodiscard]] static auto complement(const std::vector<IDType> &ids, size_t universe_size)
      -> std::pair<DynamicBitset, size_t> {
    DynamicBitset result(universe_size);
    result.set_all();
    size_t excluded_count = 0;
    for (auto id : ids) {
      auto raw_id = static_cast<size_t>(id);
      if (raw_id < universe_size && result.get(raw_id)) {
        result.reset(raw_id);
        ++excluded_count;
      }
    }
    return {std::move(result), universe_size - excluded_count};
  }

  /** @brief Build the complement of a bitset over its existing universe. */
  [[nodiscard]] static auto complement(const DynamicBitset &ids, size_t universe_size)
      -> std::pair<DynamicBitset, size_t> {
    DynamicBitset result = ids;
    result.flip_all();
    size_t excluded_count = 0;
    for (size_t raw_id = 0; raw_id < universe_size; ++raw_id) {
      excluded_count += static_cast<size_t>(ids.get(raw_id));
    }
    return {std::move(result), universe_size - excluded_count};
  }
};

}  // namespace alaya
