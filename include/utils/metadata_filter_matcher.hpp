// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
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
  struct BlockedBitsetResult {
    explicit BlockedBitsetResult(size_t data_num) : blocked_(data_num) {}

    DynamicBitset blocked_;
    size_t matched_count_ = 0;
  };

  MetadataFilterExecutor(const MetadataFilter &filter,
                         const RocksDBStorage<IDType> *storage,
                         size_t data_num)
      : filter_(filter), storage_(storage), data_num_(data_num), allow_ids_(data_num) {
    if (storage_ == nullptr) {
      throw std::invalid_argument("Storage cannot be null");
    }

    collect_required_fields(filter_, required_fields_);
    build_index_fast_path();
  }

  [[nodiscard]] auto filter() const -> const MetadataFilter & { return filter_; }
  [[nodiscard]] auto is_trivially_true() const -> bool { return filter_.is_empty(); }
  [[nodiscard]] auto has_index_fast_path() const -> bool { return has_index_fast_path_; }
  [[nodiscard]] auto indexed_ids() const -> const std::vector<IDType> & { return indexed_ids_; }
  [[nodiscard]] auto indexed_count() const -> size_t { return indexed_ids_.size(); }
  [[nodiscard]] auto data_num() const -> size_t { return data_num_; }

  [[nodiscard]] auto match(IDType id) const -> bool {
    if (filter_.is_empty()) {
      return true;
    }

    if (has_index_fast_path_) {
      return allow_ids_.get(id);
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
      for (size_t i = 0; i < ids.size(); ++i) {
        if (allow_ids_.get(ids[i])) {
          ++result.matched_count_;
        } else {
          result.blocked_.set(i);
        }
      }
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
        result.blocked_.reset(id);
      }
      result.matched_count_ = indexed_ids_.size();
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
  [[nodiscard]] auto is_indexed_field(const std::string &field) const -> bool {
    const auto &indexed_fields = storage_->config().indexed_fields_;
    return std::find(indexed_fields.begin(), indexed_fields.end(), field) != indexed_fields.end();
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
      case FilterOp::IN:
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
          return storage_->get_ids_by_int_range(cond.field,
                                                std::get<int64_t>(cond.value),
                                                std::numeric_limits<int64_t>::max());
        }
        if (std::holds_alternative<double>(cond.value)) {
          return storage_->get_ids_by_double_range(cond.field,
                                                   std::get<double>(cond.value),
                                                   std::numeric_limits<double>::max());
        }
        return std::nullopt;
      case FilterOp::GT:
        if (std::holds_alternative<int64_t>(cond.value)) {
          auto value = std::get<int64_t>(cond.value);
          if (value == std::numeric_limits<int64_t>::max()) {
            return std::vector<IDType>{};
          }
          return storage_->get_ids_by_int_range(cond.field,
                                                value + 1,
                                                std::numeric_limits<int64_t>::max());
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
          return storage_->get_ids_by_int_range(cond.field,
                                                std::numeric_limits<int64_t>::min(),
                                                std::get<int64_t>(cond.value));
        }
        if (std::holds_alternative<double>(cond.value)) {
          return storage_->get_ids_by_double_range(cond.field,
                                                   std::numeric_limits<double>::lowest(),
                                                   std::get<double>(cond.value));
        }
        return std::nullopt;
      case FilterOp::LT:
        if (std::holds_alternative<int64_t>(cond.value)) {
          auto value = std::get<int64_t>(cond.value);
          if (value == std::numeric_limits<int64_t>::min()) {
            return std::vector<IDType>{};
          }
          return storage_->get_ids_by_int_range(cond.field,
                                                std::numeric_limits<int64_t>::min(),
                                                value - 1);
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

  // TODO(P2): Extend to support multi-condition AND by intersecting indexed ID
  // sets, and OR by unioning them. Currently only single-condition AND filters
  // on an indexed field can use the fast path; all other filters fall through
  // to the O(N) full-scan in build_blocked_bitset().
  void build_index_fast_path() {
    if (filter_.logic_op != LogicOp::AND || filter_.conditions.size() != 1 ||
        !filter_.sub_filters.empty()) {
      return;
    }

    auto indexed_ids = lookup_indexed_ids(filter_.conditions.front());
    if (!indexed_ids.has_value()) {
      return;
    }

    indexed_ids_ = std::move(*indexed_ids);
    for (auto id : indexed_ids_) {
      allow_ids_.set(id);
    }
    has_index_fast_path_ = true;
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
};

}  // namespace alaya
