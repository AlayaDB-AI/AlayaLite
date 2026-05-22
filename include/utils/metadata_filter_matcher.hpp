// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "storage/rocksdb_storage.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/query_utils.hpp"
#include "utils/scalar_data.hpp"

namespace alaya {

template <typename IDType>
class MetadataFilterExecutor {
 public:
  enum class IndexBuildMode : uint8_t { kEager, kSkip };

  struct BlockedBitsetResult {
    explicit BlockedBitsetResult(size_t data_num) : blocked_(data_num) {}

    DynamicBitset blocked_;
    size_t matched_count_ = 0;
  };

  struct IndexedFilterPlan {
    std::vector<IDType> ids_;
    bool exact_ = false;
  };

  MetadataFilterExecutor(const MetadataFilter &filter,
                         const RocksDBStorage<IDType> *storage,
                         size_t data_num,
                         IndexBuildMode index_build_mode = IndexBuildMode::kEager)
      : filter_(filter), storage_(storage), data_num_(data_num), allow_ids_(data_num) {
    if (storage_ == nullptr) {
      throw std::invalid_argument("Storage cannot be null");
    }

    collect_required_fields(filter_, required_fields_);
    if (index_build_mode == IndexBuildMode::kEager) {
      build_index_fast_path();
    }
  }

  [[nodiscard]] auto filter() const -> const MetadataFilter & { return filter_; }
  [[nodiscard]] auto is_trivially_true() const -> bool { return filter_.is_empty(); }
  [[nodiscard]] auto has_index_fast_path() const -> bool { return has_index_fast_path_; }
  [[nodiscard]] auto index_fast_path_is_exact() const -> bool { return index_fast_path_exact_; }
  [[nodiscard]] auto indexed_ids() const -> const std::vector<IDType> & { return indexed_ids_; }
  [[nodiscard]] auto indexed_count() const -> size_t { return indexed_ids_.size(); }
  [[nodiscard]] auto data_num() const -> size_t { return data_num_; }

  void materialize_index_fast_path() {
    if (!has_index_fast_path_) {
      build_index_fast_path();
    }
  }

  [[nodiscard]] auto match(IDType id) const -> bool {
    if (filter_.is_empty()) {
      return true;
    }

    if (has_index_fast_path_) {
      if (static_cast<size_t>(id) >= data_num_) {
        return false;
      }
      if (!allow_ids_.get(id)) {
        return false;
      }
      if (index_fast_path_exact_) {
        return true;
      }
    }

    std::string raw_value;
    if (!storage_->get_raw_value(id, raw_value)) {
      return false;
    }
    return evaluate_raw_value(raw_value);
  }

  void eval_offsets(const std::vector<IDType> &ids, std::vector<uint8_t> &matches) const {
    auto blocked_result = build_blocked_bitset(ids);
    matches.assign(ids.size(), 0);
    for (size_t i = 0; i < ids.size(); ++i) {
      matches[i] = static_cast<uint8_t>(!blocked_result.blocked_.get(i));
    }
  }

  [[nodiscard]] auto build_blocked_bitset(const std::vector<IDType> &ids) const
      -> BlockedBitsetResult {
    BlockedBitsetResult result(ids.size());

    if (filter_.is_empty()) {
      result.matched_count_ = ids.size();
      return result;
    }

    if (has_index_fast_path_) {
      build_indexed_subset_bitset(ids, result);
      return result;
    }

    auto raw_values = storage_->batch_get_raw_values(ids);
    for (size_t i = 0; i < ids.size(); ++i) {
      if (raw_values[i].empty()) {
        result.blocked_.set(i);
        continue;
      }
      if (evaluate_raw_value(raw_values[i])) {
        ++result.matched_count_;
      } else {
        result.blocked_.set(i);
      }
    }

    return result;
  }

  [[nodiscard]] auto build_blocked_bitset() const -> BlockedBitsetResult {
    BlockedBitsetResult result(data_num_);

    if (filter_.is_empty()) {
      result.matched_count_ = data_num_;
      return result;
    }

    if (has_index_fast_path_) {
      result.blocked_.set_all();
      for (auto id : indexed_ids_) {
        if (index_fast_path_exact_) {
          result.blocked_.reset(id);
          ++result.matched_count_;
          continue;
        }

        if (match(id)) {
          result.blocked_.reset(id);
          ++result.matched_count_;
        }
      }
      return result;
    }

    // TODO(P0): This path performs O(N) RocksDB reads when no indexed sub-plan is
    // available. For large datasets, this degrades to a full table scan per query.
    if (data_num_ > 10000) {
      LOG_WARN(
          "metadata filter: O(N) full-scan fallback for {} records; "
          "consider indexing the filter fields",
          data_num_);
    }

    std::vector<IDType> ids;
    constexpr size_t kBatchSize = 1024;
    ids.reserve(kBatchSize);

    for (size_t begin = 0; begin < data_num_; begin += kBatchSize) {
      ids.clear();
      auto end = std::min(data_num_, begin + kBatchSize);
      for (size_t id = begin; id < end; ++id) {
        ids.push_back(static_cast<IDType>(id));
      }

      auto batch_result = build_blocked_bitset(ids);
      result.matched_count_ += batch_result.matched_count_;
      for (size_t i = 0; i < ids.size(); ++i) {
        if (batch_result.blocked_.get(i)) {
          result.blocked_.set(ids[i]);
        }
      }
    }

    return result;
  }

  [[nodiscard]] auto build_direct_indexed_blocked_bitset() const
      -> std::optional<BlockedBitsetResult> {
    auto condition = simple_direct_index_condition(filter_);
    if (condition == nullptr) {
      return std::nullopt;
    }

    if (auto int_range_result = build_direct_int_range_blocked_bitset(*condition)) {
      return int_range_result;
    }

    BlockedBitsetResult result(data_num_);
    result.blocked_.set_all();

    auto mark_allowed = [&result, this](IDType id) {
      if (static_cast<size_t>(id) >= data_num_) {
        return;
      }
      if (result.blocked_.get(id)) {
        result.blocked_.reset(id);
        ++result.matched_count_;
      }
    };

    if (!visit_indexed_ids(*condition, mark_allowed)) {
      return std::nullopt;
    }
    return result;
  }

 private:
  [[nodiscard]] static auto int_range_bounds(const FilterCondition &cond)
      -> std::optional<std::pair<int64_t, int64_t>> {
    switch (cond.op) {
      case FilterOp::GE:
        if (std::holds_alternative<int64_t>(cond.value)) {
          return std::pair{std::get<int64_t>(cond.value), std::numeric_limits<int64_t>::max()};
        }
        return std::nullopt;
      case FilterOp::GT:
        if (std::holds_alternative<int64_t>(cond.value)) {
          auto value = std::get<int64_t>(cond.value);
          if (value == std::numeric_limits<int64_t>::max()) {
            return std::pair{int64_t{1}, int64_t{0}};
          }
          return std::pair{value + 1, std::numeric_limits<int64_t>::max()};
        }
        return std::nullopt;
      case FilterOp::LE:
        if (std::holds_alternative<int64_t>(cond.value)) {
          return std::pair{std::numeric_limits<int64_t>::min(), std::get<int64_t>(cond.value)};
        }
        return std::nullopt;
      case FilterOp::LT:
        if (std::holds_alternative<int64_t>(cond.value)) {
          auto value = std::get<int64_t>(cond.value);
          if (value == std::numeric_limits<int64_t>::min()) {
            return std::pair{int64_t{1}, int64_t{0}};
          }
          return std::pair{std::numeric_limits<int64_t>::min(), value - 1};
        }
        return std::nullopt;
      default:
        return std::nullopt;
    }
  }

  [[nodiscard]] auto build_direct_int_range_blocked_bitset(const FilterCondition &cond) const
      -> std::optional<BlockedBitsetResult> {
    if (!is_indexed_field(cond.field)) {
      return std::nullopt;
    }

    auto bounds = int_range_bounds(cond);
    if (!bounds.has_value()) {
      return std::nullopt;
    }

    auto cached = storage_->get_int_range_blocked_bitset(cond.field,
                                                         bounds->first,
                                                         bounds->second,
                                                         data_num_);
    if (!cached.has_value() || cached->blocked_ == nullptr) {
      return std::nullopt;
    }

    BlockedBitsetResult result(data_num_);
    result.blocked_ = *cached->blocked_;
    result.matched_count_ = cached->matched_count_;
    return result;
  }

  [[nodiscard]] auto is_indexed_field(const std::string &field) const -> bool {
    const auto &indexed_fields = storage_->config().indexed_fields_;
    return std::find(indexed_fields.begin(), indexed_fields.end(), field) != indexed_fields.end();
  }

  [[nodiscard]] auto lookup_indexed_int_range_ids(const FilterCondition &cond) const
      -> std::optional<std::vector<IDType>> {
    auto bounds = int_range_bounds(cond);
    if (!bounds.has_value()) {
      return std::nullopt;
    }

    auto range = storage_->get_int_range_index_range(cond.field, bounds->first, bounds->second);
    if (!range.has_value()) {
      return std::nullopt;
    }

    std::vector<IDType> ids;
    ids.reserve(range->end_ - range->begin_);
    for (size_t i = range->begin_; i < range->end_; ++i) {
      ids.push_back((*range->entries_)[i].id_);
    }
    return ids;
  }

  [[nodiscard]] auto lookup_indexed_ids(const FilterCondition &cond) const
      -> std::optional<std::vector<IDType>> {
    if (!is_indexed_field(cond.field)) {
      return std::nullopt;
    }

    std::vector<IDType> ids;
    switch (cond.op) {
      case FilterOp::EQ:
        return storage_->get_ids_by_field_value(cond.field, cond.value);
      case FilterOp::IN_SET:
        ids.reserve(cond.values.size());
        for (const auto &value : cond.values) {
          auto partial = storage_->get_ids_by_field_value(cond.field, value);
          ids.insert(ids.end(), partial.begin(), partial.end());
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        return ids;
      case FilterOp::GE:
        if (std::holds_alternative<int64_t>(cond.value)) {
          return lookup_indexed_int_range_ids(cond);
        }
        if (std::holds_alternative<double>(cond.value)) {
          return storage_->get_ids_by_double_range(cond.field,
                                                   std::get<double>(cond.value),
                                                   std::numeric_limits<double>::max());
        }
        return std::nullopt;
      case FilterOp::GT:
        if (std::holds_alternative<int64_t>(cond.value)) {
          return lookup_indexed_int_range_ids(cond);
        }
        if (std::holds_alternative<double>(cond.value)) {
          auto value = std::get<double>(cond.value);
          return storage_
              ->get_ids_by_double_range(cond.field,
                                        std::nextafter(value, std::numeric_limits<double>::max()),
                                        std::numeric_limits<double>::max());
        }
        return std::nullopt;
      case FilterOp::LE:
        if (std::holds_alternative<int64_t>(cond.value)) {
          return lookup_indexed_int_range_ids(cond);
        }
        if (std::holds_alternative<double>(cond.value)) {
          return storage_->get_ids_by_double_range(cond.field,
                                                   std::numeric_limits<double>::lowest(),
                                                   std::get<double>(cond.value));
        }
        return std::nullopt;
      case FilterOp::LT:
        if (std::holds_alternative<int64_t>(cond.value)) {
          return lookup_indexed_int_range_ids(cond);
        }
        if (std::holds_alternative<double>(cond.value)) {
          auto value = std::get<double>(cond.value);
          return storage_
              ->get_ids_by_double_range(cond.field,
                                        std::numeric_limits<double>::lowest(),
                                        std::nextafter(value,
                                                       std::numeric_limits<double>::lowest()));
        }
        return std::nullopt;
      default:
        return std::nullopt;
    }
  }

  [[nodiscard]] auto simple_direct_index_condition(const MetadataFilter &filter) const
      -> const FilterCondition * {
    if (filter.logic_op != LogicOp::AND || filter.conditions.size() != 1 ||
        !filter.sub_filters.empty()) {
      return nullptr;
    }
    const auto &condition = filter.conditions.front();
    if (!is_indexed_field(condition.field)) {
      return nullptr;
    }
    return &condition;
  }

  template <typename Visitor>
  [[nodiscard]] auto visit_indexed_ids(const FilterCondition &cond, Visitor &&visitor) const
      -> bool {
    if (!is_indexed_field(cond.field)) {
      return false;
    }

    switch (cond.op) {
      case FilterOp::EQ:
        storage_->visit_ids_by_field_value(cond.field, cond.value, visitor);
        return true;
      case FilterOp::IN_SET:
        for (const auto &value : cond.values) {
          storage_->visit_ids_by_field_value(cond.field, value, visitor);
        }
        return true;
      case FilterOp::GE:
        if (std::holds_alternative<int64_t>(cond.value)) {
          storage_->visit_ids_by_int_range(cond.field,
                                           std::get<int64_t>(cond.value),
                                           std::numeric_limits<int64_t>::max(),
                                           visitor);
          return true;
        }
        if (std::holds_alternative<double>(cond.value)) {
          storage_->visit_ids_by_double_range(cond.field,
                                              std::get<double>(cond.value),
                                              std::numeric_limits<double>::max(),
                                              visitor);
          return true;
        }
        return false;
      case FilterOp::GT:
        if (std::holds_alternative<int64_t>(cond.value)) {
          auto value = std::get<int64_t>(cond.value);
          if (value == std::numeric_limits<int64_t>::max()) {
            return true;
          }
          storage_->visit_ids_by_int_range(cond.field,
                                           value + 1,
                                           std::numeric_limits<int64_t>::max(),
                                           visitor);
          return true;
        }
        if (std::holds_alternative<double>(cond.value)) {
          auto value = std::get<double>(cond.value);
          storage_->visit_ids_by_double_range(cond.field,
                                              std::nextafter(value,
                                                             std::numeric_limits<double>::max()),
                                              std::numeric_limits<double>::max(),
                                              visitor);
          return true;
        }
        return false;
      case FilterOp::LE:
        if (std::holds_alternative<int64_t>(cond.value)) {
          storage_->visit_ids_by_int_range(cond.field,
                                           std::numeric_limits<int64_t>::min(),
                                           std::get<int64_t>(cond.value),
                                           visitor);
          return true;
        }
        if (std::holds_alternative<double>(cond.value)) {
          storage_->visit_ids_by_double_range(cond.field,
                                              std::numeric_limits<double>::lowest(),
                                              std::get<double>(cond.value),
                                              visitor);
          return true;
        }
        return false;
      case FilterOp::LT:
        if (std::holds_alternative<int64_t>(cond.value)) {
          auto value = std::get<int64_t>(cond.value);
          if (value == std::numeric_limits<int64_t>::min()) {
            return true;
          }
          storage_->visit_ids_by_int_range(cond.field,
                                           std::numeric_limits<int64_t>::min(),
                                           value - 1,
                                           visitor);
          return true;
        }
        if (std::holds_alternative<double>(cond.value)) {
          auto value = std::get<double>(cond.value);
          storage_->visit_ids_by_double_range(cond.field,
                                              std::numeric_limits<double>::lowest(),
                                              std::nextafter(value,
                                                             std::numeric_limits<double>::lowest()),
                                              visitor);
          return true;
        }
        return false;
      default:
        return false;
    }
  }

  void normalize_indexed_ids(std::vector<IDType> &ids) const {
    ids.erase(std::remove_if(ids.begin(),
                             ids.end(),
                             [this](IDType id) {
                               return static_cast<size_t>(id) >= data_num_;
                             }),
              ids.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  }

  void discard_out_of_range_ids(std::vector<IDType> &ids) const {
    ids.erase(std::remove_if(ids.begin(),
                             ids.end(),
                             [this](IDType id) {
                               return static_cast<size_t>(id) >= data_num_;
                             }),
              ids.end());
  }

  [[nodiscard]] static auto simple_condition_ids_are_unique(FilterOp op) -> bool {
    switch (op) {
      case FilterOp::EQ:
      case FilterOp::GE:
      case FilterOp::GT:
      case FilterOp::LE:
      case FilterOp::LT:
        return true;
      default:
        return false;
    }
  }

  [[nodiscard]] auto intersect_ids(const std::vector<IDType> &lhs,
                                   const std::vector<IDType> &rhs) const -> std::vector<IDType> {
    std::vector<IDType> result;
    result.reserve(std::min(lhs.size(), rhs.size()));
    std::set_intersection(lhs.begin(),
                          lhs.end(),
                          rhs.begin(),
                          rhs.end(),
                          std::back_inserter(result));
    return result;
  }

  [[nodiscard]] auto union_ids(const std::vector<IDType> &lhs, const std::vector<IDType> &rhs) const
      -> std::vector<IDType> {
    std::vector<IDType> result;
    result.reserve(lhs.size() + rhs.size());
    std::set_union(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), std::back_inserter(result));
    return result;
  }

  [[nodiscard]] auto complement_ids(const std::vector<IDType> &ids) const -> std::vector<IDType> {
    std::vector<IDType> result;
    result.reserve(data_num_ > ids.size() ? data_num_ - ids.size() : 0);

    size_t cursor = 0;
    for (size_t raw_id = 0; raw_id < data_num_; ++raw_id) {
      auto id = static_cast<IDType>(raw_id);
      while (cursor < ids.size() && ids[cursor] < id) {
        ++cursor;
      }
      if (cursor < ids.size() && ids[cursor] == id) {
        continue;
      }
      result.push_back(id);
    }
    return result;
  }

  [[nodiscard]] auto build_condition_index_plan(const FilterCondition &condition) const
      -> std::optional<IndexedFilterPlan> {
    auto ids = lookup_indexed_ids(condition);
    if (!ids.has_value()) {
      return std::nullopt;
    }

    normalize_indexed_ids(*ids);
    return IndexedFilterPlan{std::move(*ids), true};
  }

  [[nodiscard]] auto build_simple_condition_index_plan(const FilterCondition &condition) const
      -> std::optional<IndexedFilterPlan> {
    auto ids = lookup_indexed_ids(condition);
    if (!ids.has_value()) {
      return std::nullopt;
    }

    if (simple_condition_ids_are_unique(condition.op)) {
      discard_out_of_range_ids(*ids);
    } else {
      normalize_indexed_ids(*ids);
    }
    return IndexedFilterPlan{std::move(*ids), true};
  }

  [[nodiscard]] auto build_index_plan_for_first_not_child(const MetadataFilter &filter) const
      -> std::optional<IndexedFilterPlan> {
    if (!filter.conditions.empty()) {
      return build_condition_index_plan(filter.conditions.front());
    }
    if (!filter.sub_filters.empty() && filter.sub_filters.front() != nullptr) {
      return build_index_plan(*filter.sub_filters.front());
    }
    return std::nullopt;
  }

  [[nodiscard]] auto build_and_index_plan(const MetadataFilter &filter) const
      -> std::optional<IndexedFilterPlan> {
    std::optional<std::vector<IDType>> current_ids;
    bool exact = true;

    auto consume_plan = [&](const std::optional<IndexedFilterPlan> &plan) {
      if (!plan.has_value()) {
        exact = false;
        return;
      }
      if (!current_ids.has_value()) {
        current_ids = plan->ids_;
      } else {
        current_ids = intersect_ids(*current_ids, plan->ids_);
      }
      exact = exact && plan->exact_;
    };

    for (const auto &condition : filter.conditions) {
      consume_plan(build_condition_index_plan(condition));
    }
    for (const auto &sub_filter : filter.sub_filters) {
      if (sub_filter == nullptr) {
        consume_plan(std::nullopt);
      } else {
        consume_plan(build_index_plan(*sub_filter));
      }
    }

    if (!current_ids.has_value()) {
      return std::nullopt;
    }
    return IndexedFilterPlan{std::move(*current_ids), exact};
  }

  [[nodiscard]] auto build_or_index_plan(const MetadataFilter &filter) const
      -> std::optional<IndexedFilterPlan> {
    std::vector<IDType> current_ids;
    bool have_plan = false;
    bool exact = true;

    auto consume_plan = [&](const std::optional<IndexedFilterPlan> &plan) -> bool {
      if (!plan.has_value()) {
        return false;
      }
      if (!have_plan) {
        current_ids = plan->ids_;
        have_plan = true;
      } else {
        current_ids = union_ids(current_ids, plan->ids_);
      }
      exact = exact && plan->exact_;
      return true;
    };

    for (const auto &condition : filter.conditions) {
      if (!consume_plan(build_condition_index_plan(condition))) {
        return std::nullopt;
      }
    }
    for (const auto &sub_filter : filter.sub_filters) {
      if (sub_filter == nullptr || !consume_plan(build_index_plan(*sub_filter))) {
        return std::nullopt;
      }
    }

    if (!have_plan) {
      return std::nullopt;
    }
    return IndexedFilterPlan{std::move(current_ids), exact};
  }

  [[nodiscard]] auto build_not_index_plan(const MetadataFilter &filter) const
      -> std::optional<IndexedFilterPlan> {
    auto child_plan = build_index_plan_for_first_not_child(filter);
    if (!child_plan.has_value() || !child_plan->exact_) {
      return std::nullopt;
    }
    return IndexedFilterPlan{complement_ids(child_plan->ids_), true};
  }

  [[nodiscard]] auto build_index_plan(const MetadataFilter &filter) const
      -> std::optional<IndexedFilterPlan> {
    if (filter.is_empty()) {
      return std::nullopt;
    }

    if (filter.logic_op == LogicOp::AND && filter.conditions.size() == 1 &&
        filter.sub_filters.empty()) {
      return build_simple_condition_index_plan(filter.conditions.front());
    }

    switch (filter.logic_op) {
      case LogicOp::AND:
        return build_and_index_plan(filter);
      case LogicOp::OR:
        return build_or_index_plan(filter);
      case LogicOp::NOT:
        return build_not_index_plan(filter);
    }
    return std::nullopt;
  }

  void build_indexed_subset_bitset(const std::vector<IDType> &ids,
                                   BlockedBitsetResult &result) const {
    if (index_fast_path_exact_) {
      for (size_t i = 0; i < ids.size(); ++i) {
        if (static_cast<size_t>(ids[i]) >= data_num_) {
          result.blocked_.set(i);
          continue;
        }
        if (allow_ids_.get(ids[i])) {
          ++result.matched_count_;
        } else {
          result.blocked_.set(i);
        }
      }
      return;
    }

    std::vector<IDType> candidate_ids;
    std::vector<size_t> candidate_positions;
    candidate_ids.reserve(ids.size());
    candidate_positions.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      if (static_cast<size_t>(ids[i]) >= data_num_) {
        result.blocked_.set(i);
        continue;
      }
      if (!allow_ids_.get(ids[i])) {
        result.blocked_.set(i);
        continue;
      }
      candidate_ids.push_back(ids[i]);
      candidate_positions.push_back(i);
    }

    auto raw_values = storage_->batch_get_raw_values(candidate_ids);
    for (size_t i = 0; i < candidate_ids.size(); ++i) {
      if (!raw_values[i].empty() && evaluate_raw_value(raw_values[i])) {
        ++result.matched_count_;
      } else {
        result.blocked_.set(candidate_positions[i]);
      }
    }
  }

  void build_index_fast_path() {
    auto indexed_plan = build_index_plan(filter_);
    if (!indexed_plan.has_value()) {
      return;
    }

    indexed_ids_ = std::move(indexed_plan->ids_);
    for (auto id : indexed_ids_) {
      allow_ids_.set(id);
    }
    has_index_fast_path_ = true;
    index_fast_path_exact_ = indexed_plan->exact_;
  }

  [[nodiscard]] auto evaluate_raw_value(const std::string &raw_value) const -> bool {
    auto metadata = ScalarData::deserialize_selected_metadata(raw_value.data(),
                                                              raw_value.size(),
                                                              required_fields_);
    return filter_.evaluate(metadata);
  }

  static void collect_required_fields(const MetadataFilter &filter,
                                      std::unordered_set<std::string> &fields) {
    for (const auto &cond : filter.conditions) {
      fields.insert(cond.field);
    }
    for (const auto &sub_filter : filter.sub_filters) {
      collect_required_fields(*sub_filter, fields);
    }
  }

  const MetadataFilter &filter_;
  const RocksDBStorage<IDType> *storage_ = nullptr;
  size_t data_num_ = 0;
  std::unordered_set<std::string> required_fields_;
  DynamicBitset allow_ids_;
  std::vector<IDType> indexed_ids_;
  bool has_index_fast_path_ = false;
  bool index_fast_path_exact_ = false;
};

}  // namespace alaya
