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

#include <algorithm>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "utils/scalar_data.hpp"

namespace alaya {
// NOLINTBEGIN
/// Comparison operators for filter conditions
enum class FilterOp : uint8_t {
  EQ,       ///< Equal (==)
  NE,       ///< Not equal (!=)
  GT,       ///< Greater than (>)
  GE,       ///< Greater than or equal (>=)
  LT,       ///< Less than (<)
  LE,       ///< Less than or equal (<=)
  IN,       ///< Value in list
  NOT_IN,   ///< Value not in list
  CONTAINS  ///< String contains substring
};

/// Logical operators for combining filter conditions
enum class LogicOp : uint8_t {
  AND,  ///< All conditions must be true
  OR,   ///< At least one condition must be true
  NOT   ///< Negate the condition
};

/**
 * @brief A single filter condition for metadata filtering
 *
 * Represents a condition like: field $op value
 * Example: "category" EQ "tech"
 */
struct FilterCondition {
  std::string field;                  ///< Field name in metadata
  FilterOp op;                        ///< Comparison operator
  MetadataValue value;                ///< Value to compare against
  std::vector<MetadataValue> values;  ///< Values for IN/NOT_IN operators

  /**
   * @brief Evaluate this condition against a metadata map
   * @param metadata The metadata to check
   * @return true if the condition is satisfied
   */
  [[nodiscard]] auto evaluate(const MetadataMap &metadata) const -> bool {
    auto it = metadata.find(field);
    if (it == metadata.end()) {
      return false;  // Field not found, condition not satisfied
    }

    const auto &actual = it->second;

    switch (op) {
      case FilterOp::EQ:
        return actual == value;
      case FilterOp::NE:
        return actual != value;
      case FilterOp::GT:
        return compare(actual, value) > 0;
      case FilterOp::GE:
        return compare(actual, value) >= 0;
      case FilterOp::LT:
        return compare(actual, value) < 0;
      case FilterOp::LE:
        return compare(actual, value) <= 0;
      case FilterOp::IN:
        return std::find(values.begin(), values.end(), actual) != values.end();
      case FilterOp::NOT_IN:
        return std::find(values.begin(), values.end(), actual) == values.end();
      case FilterOp::CONTAINS:
        return contains_string(actual, value);
      default:
        return false;
    }
  }

 private:
  /**
   * @brief Compare two MetadataValue objects
   * @return -1 if a < b, 0 if a == b, 1 if a > b
   */
  static auto compare(const MetadataValue &a, const MetadataValue &b) -> int {
    // Type mismatch: cannot compare
    if (a.index() != b.index()) {
      return 0;
    }

    return std::visit(
        [&b](const auto &va) -> int {
          using T = std::decay_t<decltype(va)>;
          const auto &vb = std::get<T>(b);
          if (va < vb) {
            return -1;
          }
          if (va > vb) {
            return 1;
          }
          return 0;
        },
        a);
  }

  /**
   * @brief Check if actual string contains pattern string
   */
  static auto contains_string(const MetadataValue &actual, const MetadataValue &pattern) -> bool {
    if (!std::holds_alternative<std::string>(actual) ||
        !std::holds_alternative<std::string>(pattern)) {
      return false;
    }
    return std::get<std::string>(actual).find(std::get<std::string>(pattern)) != std::string::npos;
  }
};

/**
 * @brief Composite filter expression supporting nested conditions
 *
 * Supports logical combinations of FilterCondition and nested MetadataFilter.
 * Example filter structure:
 * {
 *   "$and": [
 *     {"category": {"$eq": "technology"}},
 *     {"score": {"$gt": 100}},
 *     {"$or": [
 *       {"tag": {"$in": ["alaya", "lite", "dbgroup"]}},
 *       {"featured": {"$eq": true}}
 *     ]}
 *   ]
 * }
 */
struct MetadataFilter {
  LogicOp logic_op = LogicOp::AND;                           ///< Logic operator
  std::vector<FilterCondition> conditions;                   ///< Direct conditions
  std::vector<std::shared_ptr<MetadataFilter>> sub_filters;  ///< Nested filters

  /**
   * @brief Evaluate this filter against a metadata map
   * @param metadata The metadata to check
   * @return true if the filter is satisfied
   */
  [[nodiscard]] auto evaluate(const MetadataMap &metadata) const -> bool {
    if (is_empty()) {
      return true;  // Empty filter matches everything
    }

    std::vector<bool> results;
    results.reserve(conditions.size() + sub_filters.size());

    // Evaluate all direct conditions
    for (const auto &cond : conditions) {
      results.push_back(cond.evaluate(metadata));
    }

    // Evaluate all nested filters
    for (const auto &sub : sub_filters) {
      results.push_back(sub->evaluate(metadata));
    }

    if (results.empty()) {
      return true;
    }

    switch (logic_op) {
      case LogicOp::AND:
        return std::all_of(results.begin(), results.end(), [](bool v) {
          return v;
        });
      case LogicOp::OR:
        return std::any_of(results.begin(), results.end(), [](bool v) {
          return v;
        });
      case LogicOp::NOT:
        // NOT only applies to the first result
        return !results[0];
      default:
        return true;
    }
  }

  /**
   * @brief Create an empty filter that matches all records
   * @return Empty MetadataFilter
   */
  static auto empty() -> MetadataFilter { return MetadataFilter{}; }

  /**
   * @brief Check if this filter is empty (matches all)
   * @return true if empty
   */
  [[nodiscard]] auto is_empty() const -> bool { return conditions.empty() && sub_filters.empty(); }

  /**
   * @brief Add a simple equality condition
   * @param field Field name
   * @param value Value to compare
   * @return Reference to this filter for chaining
   */
  auto add_eq(const std::string &field, const MetadataValue &value) -> MetadataFilter & {
    FilterCondition cond;
    cond.field = field;
    cond.op = FilterOp::EQ;
    cond.value = value;
    conditions.push_back(std::move(cond));
    return *this;
  }

  /**
   * @brief Add a greater-than condition
   * @param field Field name
   * @param value Value to compare
   * @return Reference to this filter for chaining
   */
  auto add_gt(const std::string &field, const MetadataValue &value) -> MetadataFilter & {
    FilterCondition cond;
    cond.field = field;
    cond.op = FilterOp::GT;
    cond.value = value;
    conditions.push_back(std::move(cond));
    return *this;
  }

  /**
   * @brief Add a less-than condition
   * @param field Field name
   * @param value Value to compare
   * @return Reference to this filter for chaining
   */
  auto add_lt(const std::string &field, const MetadataValue &value) -> MetadataFilter & {
    FilterCondition cond;
    cond.field = field;
    cond.op = FilterOp::LT;
    cond.value = value;
    conditions.push_back(std::move(cond));
    return *this;
  }

  /**
   * @brief Add an IN condition
   * @param field Field name
   * @param values List of acceptable values
   * @return Reference to this filter for chaining
   */
  auto add_in(const std::string &field, std::vector<MetadataValue> values) -> MetadataFilter & {
    FilterCondition cond;
    cond.field = field;
    cond.op = FilterOp::IN;
    cond.values = std::move(values);
    conditions.push_back(std::move(cond));
    return *this;
  }

  /**
   * @brief Add a nested sub-filter
   * @param sub_filter The nested filter
   * @return Reference to this filter for chaining
   */
  auto add_sub_filter(MetadataFilter sub_filter) -> MetadataFilter & {
    sub_filters.push_back(std::make_shared<MetadataFilter>(std::move(sub_filter)));
    return *this;
  }
};
// NOLINTEND
}  // namespace alaya
