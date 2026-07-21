// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "utils/metadata_filter.hpp"

namespace alaya {

/**
 * @brief Query-facing scalar secondary-index contract.
 *
 * Implementations translate one indexable condition into internal IDs. They do not evaluate
 * residual predicates or know anything about vector indexes.
 */
template <typename IDType>
class ScalarIndex {
 public:
  using IdVisitor = std::function<void(IDType)>;

  /** @brief Destroy the scalar-index implementation. */
  virtual ~ScalarIndex() = default;

  /** @brief Return whether the field has a secondary index. */
  [[nodiscard]] virtual auto is_indexed_field(const std::string &field) const -> bool = 0;

  /**
   * @brief Resolve an indexable condition to IDs.
   * @return Matching IDs, or std::nullopt when the condition cannot use this index.
   */
  [[nodiscard]] virtual auto lookup(const FilterCondition &condition) const
      -> std::optional<std::vector<IDType>> = 0;

  /**
   * @brief Visit IDs for an indexable condition without requiring callers to retain the set.
   * @return false when the condition cannot use this index.
   */
  virtual auto visit(const FilterCondition &condition, const IdVisitor &visitor) const -> bool {
    auto ids = lookup(condition);
    if (!ids.has_value()) {
      return false;
    }
    for (auto id : *ids) {
      visitor(id);
    }
    return true;
  }
};

}  // namespace alaya
