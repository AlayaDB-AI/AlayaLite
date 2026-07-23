// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "scalar/id_set_algebra.hpp"
#include "scalar/scalar_index.hpp"
#include "storage/legacy_rocksdb_adapters.hpp"
#include "storage/record_store.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/query_utils.hpp"
#include "utils/scalar_data.hpp"

namespace alaya {

template <typename IDType>
class MetadataFilterExecutor {
 public:
  struct BlockedBitsetResult {
    explicit BlockedBitsetResult(size_t data_num) : blocked_(data_num) {}

    DynamicBitset blocked_;
    size_t matched_count_ = 0;
  };

  struct IndexedFilterPlan {
    std::vector<IDType> ids_;
    std::optional<DynamicBitset> allow_ids_;
    size_t matched_count_ = 0;
    bool exact_ = false;
  };

  enum class IndexBuildMode {
    kBuild,
    kSkip,
  };

  /** @brief Build an executor through compatibility adapters over legacy RocksDB storage. */
  MetadataFilterExecutor(const MetadataFilter &filter,
                         const RocksDBStorage<IDType> *storage,
                         size_t data_num,
                         IndexBuildMode index_build_mode = IndexBuildMode::kBuild)
      : filter_(filter), data_num_(data_num), allow_ids_(data_num) {
    if (storage == nullptr) {
      throw std::invalid_argument("Storage cannot be null");
    }

    owned_scalar_index_ = std::make_shared<LegacyRocksDBScalarIndex<IDType>>(storage);
    owned_record_store_ = std::make_shared<LegacyRocksDBRecordStore<IDType>>(storage);
    scalar_index_ = owned_scalar_index_.get();
    record_store_ = owned_record_store_.get();

    initialize(index_build_mode);
  }

  /** @brief Build an executor from storage-engine-neutral scalar query providers. */
  MetadataFilterExecutor(const MetadataFilter &filter,
                         const ScalarIndex<IDType> *scalar_index,
                         const RecordStore<IDType> *record_store,
                         size_t data_num,
                         IndexBuildMode index_build_mode = IndexBuildMode::kBuild)
      : filter_(filter),
        scalar_index_(scalar_index),
        record_store_(record_store),
        data_num_(data_num),
        allow_ids_(data_num) {
    if (scalar_index_ == nullptr || record_store_ == nullptr) {
      throw std::invalid_argument("ScalarIndex and RecordStore cannot be null");
    }
    if (scalar_index_->generation() != record_store_->generation()) {
      throw std::invalid_argument("ScalarIndex and RecordStore generations do not match");
    }

    initialize(index_build_mode);
  }

  /** @brief Build an executor over a generation-stable provider view with explicit live IDs. */
  MetadataFilterExecutor(const MetadataFilter &filter,
                         const ScalarIndex<IDType> *scalar_index,
                         const RecordStore<IDType> *record_store,
                         size_t data_num,
                         const DynamicBitset *live_mask,
                         size_t live_count,
                         IndexBuildMode index_build_mode = IndexBuildMode::kBuild)
      : filter_(filter),
        scalar_index_(scalar_index),
        record_store_(record_store),
        data_num_(data_num),
        live_mask_(live_mask),
        live_count_(live_count),
        allow_ids_(data_num) {
    if (scalar_index_ == nullptr || record_store_ == nullptr || live_mask_ == nullptr) {
      throw std::invalid_argument("ScalarIndex, RecordStore and live mask cannot be null");
    }
    if (scalar_index_->generation() != record_store_->generation()) {
      throw std::invalid_argument("ScalarIndex and RecordStore generations do not match");
    }
    if (live_mask_->size() != data_num_ || live_count_ > data_num_) {
      throw std::invalid_argument("Live mask does not match the scalar ID universe");
    }

    initialize(index_build_mode);
  }

 private:
  /** @brief Collect residual fields and optionally compile the scalar-index fast path. */
  void initialize(IndexBuildMode index_build_mode) {
    collect_required_fields(filter_, required_fields_);
    if (index_build_mode == IndexBuildMode::kBuild) {
      build_index_fast_path();
    }
  }

 public:
  [[nodiscard]] auto filter() const -> const MetadataFilter & { return filter_; }

  /**
   * @brief Returns true only when the filter expression is empty and therefore matches every row.
   *
   * This says nothing about scalar-index availability. A non-empty unindexed filter returns false
   * and must still be evaluated through the residual/full-scan path.
   */
  [[nodiscard]] auto is_trivially_true() const -> bool {
    return filter_.is_empty() && live_count_ == data_num_;
  }
  [[nodiscard]] auto has_index_fast_path() const -> bool { return has_index_fast_path_; }
  [[nodiscard]] auto index_fast_path_is_exact() const -> bool { return index_fast_path_exact_; }
  [[nodiscard]] auto index_fast_path_uses_materialized_ids() const -> bool {
    return index_fast_path_uses_materialized_ids_;
  }
  [[nodiscard]] auto indexed_ids() const -> const std::vector<IDType> & { return indexed_ids_; }
  [[nodiscard]] auto indexed_count() const -> size_t { return indexed_count_; }
  [[nodiscard]] auto data_num() const -> size_t { return data_num_; }

  template <typename Visitor>
  void visit_index_fast_path_ids(Visitor &&visitor) const {
    if (!has_index_fast_path_) {
      return;
    }

    if (index_fast_path_uses_materialized_ids_) {
      for (auto id : indexed_ids_) {
        visitor(id);
      }
      return;
    }

    for (size_t raw_id = 0; raw_id < data_num_; ++raw_id) {
      if (allow_ids_.get(raw_id)) {
        visitor(static_cast<IDType>(raw_id));
      }
    }
  }

  void materialize_index_fast_path() {
    if (!has_index_fast_path_) {
      build_index_fast_path();
    }
  }

  [[nodiscard]] auto build_direct_indexed_blocked_bitset() const
      -> std::optional<BlockedBitsetResult> {
    auto condition = simple_direct_index_condition(filter_);
    if (condition == nullptr) {
      return std::nullopt;
    }

    BlockedBitsetResult result(data_num_);
    result.blocked_.set_all();
    bool indexed = visit_indexed_ids(*condition, [&](IDType id) {
      auto raw_id = static_cast<size_t>(id);
      if (!is_live(id) || !result.blocked_.get(raw_id)) {
        return;
      }
      result.blocked_.reset(raw_id);
      ++result.matched_count_;
    });
    if (!indexed) {
      return std::nullopt;
    }
    return result;
  }

  [[nodiscard]] auto match(IDType id) const -> bool {
    if (!is_live(id)) {
      return false;
    }
    if (filter_.is_empty()) {
      return true;
    }

    if (has_index_fast_path_) {
      if (!allow_ids_.get(id)) {
        return false;
      }
      return index_fast_path_exact_ || match_raw_value(id);
    }

    return match_raw_value(id);
  }

  [[nodiscard]] auto match_raw_value(IDType id) const -> bool {
    if (!is_live(id)) {
      return false;
    }
    std::string raw_value;
    if (!record_store_->get_raw_scalar(id, raw_value)) {
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
      for (size_t i = 0; i < ids.size(); ++i) {
        if (is_live(ids[i])) {
          ++result.matched_count_;
        } else {
          result.blocked_.set(i);
        }
      }
      return result;
    }

    if (has_index_fast_path_) {
      for (size_t i = 0; i < ids.size(); ++i) {
        if (!is_live(ids[i]) || !allow_ids_.get(ids[i])) {
          result.blocked_.set(i);
        } else if (index_fast_path_exact_ || match_raw_value(ids[i])) {
          ++result.matched_count_;
        } else {
          result.blocked_.set(i);
        }
      }
      return result;
    }

    auto raw_values = record_store_->batch_get_raw_scalars(ids);
    for (size_t i = 0; i < ids.size(); ++i) {
      if (!is_live(ids[i]) || raw_values[i].empty()) {
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
      result.matched_count_ = live_count_;
      if (live_mask_ != nullptr) {
        for (size_t raw_id = 0; raw_id < data_num_; ++raw_id) {
          if (!live_mask_->get(raw_id)) {
            result.blocked_.set(raw_id);
          }
        }
      }
      return result;
    }

    if (has_index_fast_path_) {
      result.blocked_.set_all();
      visit_index_fast_path_ids([&](IDType id) {
        if (!is_live(id)) {
          return;
        }
        if (index_fast_path_exact_) {
          result.blocked_.reset(id);
          ++result.matched_count_;
          return;
        }

        if (match(id)) {
          result.blocked_.reset(id);
          ++result.matched_count_;
        }
      });
      return result;
    }

    // TODO(P0): This path performs O(N) RocksDB reads when the index fast path
    // is not available (any filter beyond single-condition AND on an indexed
    // field). For large datasets, this degrades to a full table scan per query.
    // Future work: support multi-condition index intersection to avoid this.
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

 private:
  /** @brief Return whether an ID belongs to the provider generation rather than a deleted hole. */
  [[nodiscard]] auto is_live(IDType id) const -> bool {
    auto raw_id = static_cast<size_t>(id);
    return raw_id < data_num_ && (live_mask_ == nullptr || live_mask_->get(raw_id));
  }

  [[nodiscard]] auto is_indexed_field(const std::string &field) const -> bool {
    return scalar_index_->is_indexed_field(field);
  }

  [[nodiscard]] auto lookup_indexed_ids(const FilterCondition &cond) const
      -> std::optional<std::vector<IDType>> {
    return scalar_index_->lookup(cond);
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
    return scalar_index_->visit(cond, std::forward<Visitor>(visitor));
  }

  void normalize_indexed_ids(std::vector<IDType> &ids) const {
    ids.erase(std::remove_if(ids.begin(),
                             ids.end(),
                             [this](IDType id) {
                               return !is_live(id);
                             }),
              ids.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  }

  void discard_out_of_range_ids(std::vector<IDType> &ids) const {
    ids.erase(std::remove_if(ids.begin(),
                             ids.end(),
                             [this](IDType id) {
                               return !is_live(id);
                             }),
              ids.end());
  }

  /** @brief Intersect an allow bitset with the live IDs and refresh its cardinality. */
  void restrict_to_live_ids(DynamicBitset &allow_ids, size_t &matched_count) const {
    matched_count = 0;
    for (size_t raw_id = 0; raw_id < data_num_; ++raw_id) {
      if (!is_live(static_cast<IDType>(raw_id))) {
        allow_ids.reset(raw_id);
      } else if (allow_ids.get(raw_id)) {
        ++matched_count;
      }
    }
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

  [[nodiscard]] auto materialize_plan_ids(const IndexedFilterPlan &plan) const
      -> std::vector<IDType> {
    if (!plan.allow_ids_.has_value()) {
      return plan.ids_;
    }

    std::vector<IDType> ids;
    ids.reserve(plan.matched_count_);
    for (size_t raw_id = 0; raw_id < data_num_; ++raw_id) {
      if (plan.allow_ids_->get(raw_id)) {
        ids.push_back(static_cast<IDType>(raw_id));
      }
    }
    return ids;
  }

  [[nodiscard]] auto build_condition_index_plan(const FilterCondition &condition) const
      -> std::optional<IndexedFilterPlan> {
    auto ids = lookup_indexed_ids(condition);
    if (!ids.has_value()) {
      return std::nullopt;
    }

    normalize_indexed_ids(*ids);
    auto matched_count = ids->size();
    return IndexedFilterPlan{std::move(*ids), std::nullopt, matched_count, true};
  }

  void apply_index_plan(IndexedFilterPlan &&indexed_plan) {
    indexed_ids_ = std::move(indexed_plan.ids_);
    index_fast_path_exact_ = indexed_plan.exact_;
    allow_ids_.clear();
    if (indexed_plan.allow_ids_.has_value()) {
      restrict_to_live_ids(*indexed_plan.allow_ids_, indexed_plan.matched_count_);
      allow_ids_ = std::move(*indexed_plan.allow_ids_);
      index_fast_path_uses_materialized_ids_ = false;
    } else {
      for (auto id : indexed_ids_) {
        if (static_cast<size_t>(id) < data_num_) {
          allow_ids_.set(id);
        }
      }
      index_fast_path_uses_materialized_ids_ = true;
    }
    indexed_count_ = indexed_plan.matched_count_;
    has_index_fast_path_ = true;
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
    auto matched_count = ids->size();
    return IndexedFilterPlan{std::move(*ids), std::nullopt, matched_count, true};
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
      auto plan_ids = materialize_plan_ids(*plan);
      if (!current_ids.has_value()) {
        current_ids = std::move(plan_ids);
      } else {
        current_ids = ScalarIdSetAlgebra<IDType>::intersect(*current_ids, plan_ids);
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
    auto matched_count = current_ids->size();
    return IndexedFilterPlan{std::move(*current_ids), std::nullopt, matched_count, exact};
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
      auto plan_ids = materialize_plan_ids(*plan);
      if (!have_plan) {
        current_ids = std::move(plan_ids);
        have_plan = true;
      } else {
        current_ids = ScalarIdSetAlgebra<IDType>::unite(current_ids, plan_ids);
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
    auto matched_count = current_ids.size();
    return IndexedFilterPlan{std::move(current_ids), std::nullopt, matched_count, exact};
  }

  [[nodiscard]] auto build_not_index_plan(const MetadataFilter &filter) const
      -> std::optional<IndexedFilterPlan> {
    auto child_plan = build_index_plan_for_first_not_child(filter);
    if (!child_plan.has_value() || !child_plan->exact_) {
      return std::nullopt;
    }
    if (child_plan->allow_ids_.has_value()) {
      auto [allow_ids, matched_count] =
          ScalarIdSetAlgebra<IDType>::complement(*child_plan->allow_ids_, data_num_);
      restrict_to_live_ids(allow_ids, matched_count);
      return IndexedFilterPlan{std::vector<IDType>{}, std::move(allow_ids), matched_count, true};
    }
    auto [allow_ids, matched_count] =
        ScalarIdSetAlgebra<IDType>::complement(child_plan->ids_, data_num_);
    restrict_to_live_ids(allow_ids, matched_count);
    return IndexedFilterPlan{std::vector<IDType>{}, std::move(allow_ids), matched_count, true};
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

  void build_index_fast_path() {
    auto indexed_plan = build_index_plan(filter_);
    if (!indexed_plan.has_value()) {
      return;
    }
    apply_index_plan(std::move(*indexed_plan));
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

  const MetadataFilter &filter_;  ///< Parsed predicate tree whose lifetime exceeds this executor.
  std::shared_ptr<ScalarIndex<IDType>> owned_scalar_index_;  ///< Legacy compatibility adapter.
  std::shared_ptr<RecordStore<IDType>> owned_record_store_;  ///< Legacy compatibility adapter.
  const ScalarIndex<IDType> *scalar_index_ = nullptr;  ///< Non-owning secondary-index provider.
  const RecordStore<IDType> *record_store_ = nullptr;  ///< Non-owning canonical-record provider.
  size_t data_num_ = 0;                       ///< Valid internal-ID universe [0, data_num_).
  const DynamicBitset *live_mask_ = nullptr;  ///< Non-owning immutable mask; null means all live.
  size_t live_count_ = data_num_;             ///< Live IDs represented by live_mask_.
  std::unordered_set<std::string> required_fields_;  ///< Fields needed for residual evaluation.
  DynamicBitset allow_ids_;             ///< Materialized candidates for the current indexed plan.
  std::vector<IDType> indexed_ids_;     ///< Sorted candidates when vector form is cheaper.
  size_t indexed_count_ = 0;            ///< Candidate count before residual evaluation.
  bool has_index_fast_path_ = false;    ///< At least one predicate branch used ScalarIndex.
  bool index_fast_path_exact_ = false;  ///< Indexed candidates fully decide the predicate.
  bool index_fast_path_uses_materialized_ids_ = true;  ///< indexed_ids_ is authoritative.
};

}  // namespace alaya
