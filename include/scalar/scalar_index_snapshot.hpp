// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "scalar/id_set_algebra.hpp"
#include "scalar/scalar_index.hpp"
#include "utils/index_encoding.hpp"
#include "utils/query_utils.hpp"
#include "utils/scalar_data.hpp"

namespace alaya {

/**
 * @brief Immutable in-memory scalar index published as one record-store generation.
 *
 * A snapshot contains only query-hot metadata: live internal IDs and postings for configured
 * fields. Canonical ScalarData, documents and vectors remain in RocksDB. Readers retain a shared
 * pointer, so a concurrent writer can publish the next snapshot without locking active queries.
 */
template <typename IDType>
class ScalarIndexSnapshot final : public ScalarIndex<IDType> {
 public:
  using Record = std::pair<IDType, ScalarData>;

  /**
   * @brief Build a fully normalized snapshot from canonical records.
   * @param generation Persisted generation represented by records.
   * @param universe_size Exclusive upper bound for valid internal IDs.
   * @param indexed_fields Metadata fields retained in memory.
   * @param records Canonical live records with unique IDs inside the declared universe.
   * @throws std::invalid_argument if a record ID is duplicated or outside the universe.
   */
  [[nodiscard]] static auto build(uint64_t generation,
                                  size_t universe_size,
                                  const std::vector<std::string> &indexed_fields,
                                  const std::vector<Record> &records)
      -> std::shared_ptr<const ScalarIndexSnapshot> {
    auto snapshot = std::shared_ptr<ScalarIndexSnapshot>(
        new ScalarIndexSnapshot(generation, universe_size, indexed_fields));
    for (const auto &[id, scalar] : records) {
      snapshot->add_record(id, scalar);
    }
    snapshot->normalize();
    return snapshot;
  }

  /** @copydoc ScalarIndex::is_indexed_field */
  [[nodiscard]] auto is_indexed_field(const std::string &field) const -> bool override {
    return fields_.contains(field);
  }

  /** @copydoc ScalarIndex::generation */
  [[nodiscard]] auto generation() const -> uint64_t override { return generation_; }

  /** @brief Return the internal-ID universe represented by live_mask(). */
  [[nodiscard]] auto universe_size() const -> size_t { return live_mask_.size(); }

  /** @brief Return the number of live records in this generation. */
  [[nodiscard]] auto live_count() const -> size_t { return live_count_; }

  /** @brief Return a read-only bitset whose set bits identify live records. */
  [[nodiscard]] auto live_mask() const -> const DynamicBitset & { return live_mask_; }

  /** @copydoc ScalarIndex::lookup */
  [[nodiscard]] auto lookup(const FilterCondition &condition) const
      -> std::optional<std::vector<IDType>> override {
    auto field = fields_.find(condition.field);
    if (field == fields_.end()) {
      return std::nullopt;
    }

    switch (condition.op) {
      case FilterOp::EQ:
        return equality_lookup(field->second, condition.value);
      case FilterOp::NE:
        return difference(field->second.present_ids_,
                          equality_lookup(field->second, condition.value));
      case FilterOp::IN_SET:
        return union_values(field->second, condition.values);
      case FilterOp::NOT_IN_SET:
        return difference(field->second.present_ids_,
                          union_values(field->second, condition.values));
      case FilterOp::GE:
      case FilterOp::GT:
      case FilterOp::LE:
      case FilterOp::LT:
        return range_lookup(field->second, condition.op, condition.value);
      case FilterOp::CONTAINS:
        return std::nullopt;
    }
    return std::nullopt;
  }

 private:
  struct FieldIndex {
    std::map<std::string, std::vector<IDType>> postings_;  ///< Encoded value to sorted live IDs.
    std::vector<IDType> present_ids_;  ///< Sorted IDs containing this field, used by NE/NOT_IN.
  };

  ScalarIndexSnapshot(uint64_t generation,
                      size_t universe_size,
                      const std::vector<std::string> &indexed_fields)
      : generation_(generation), live_mask_(universe_size) {
    for (const auto &field : indexed_fields) {
      fields_.try_emplace(field);
    }
  }

  /** @brief Add one canonical row while the snapshot is exclusively under construction. */
  void add_record(IDType id, const ScalarData &scalar) {
    auto raw_id = static_cast<size_t>(id);
    if (raw_id >= live_mask_.size()) {
      throw std::invalid_argument("Scalar snapshot record ID is outside its universe");
    }
    if (live_mask_.get(raw_id)) {
      throw std::invalid_argument("Scalar snapshot contains a duplicate record ID");
    }
    live_mask_.set(raw_id);
    ++live_count_;

    for (auto &[field_name, field_index] : fields_) {
      auto value = scalar.metadata.find(field_name);
      if (value == scalar.metadata.end()) {
        continue;
      }
      field_index.present_ids_.push_back(id);
      field_index.postings_[index_encoding::encode_value(value->second)].push_back(id);
    }
  }

  /** @brief Sort and deduplicate every posting before the snapshot becomes externally visible. */
  void normalize() {
    for (auto &[field_name, field_index] : fields_) {
      (void)field_name;
      normalize_ids(field_index.present_ids_);
      for (auto &[encoded_value, ids] : field_index.postings_) {
        (void)encoded_value;
        normalize_ids(ids);
      }
    }
  }

  /** @brief Sort and deduplicate one posting list. */
  static void normalize_ids(std::vector<IDType> &ids) {
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  }

  /** @brief Return one exact-value posting, or an empty posting when the value is absent. */
  [[nodiscard]] static auto equality_lookup(const FieldIndex &field, const MetadataValue &value)
      -> std::vector<IDType> {
    auto posting = field.postings_.find(index_encoding::encode_value(value));
    return posting == field.postings_.end() ? std::vector<IDType>{} : posting->second;
  }

  /** @brief Union exact postings for IN/NOT_IN expressions. */
  [[nodiscard]] static auto union_values(const FieldIndex &field,
                                         const std::vector<MetadataValue> &values)
      -> std::vector<IDType> {
    std::vector<IDType> result;
    for (const auto &value : values) {
      result = ScalarIdSetAlgebra<IDType>::unite(result, equality_lookup(field, value));
    }
    return result;
  }

  /** @brief Subtract a sorted posting from the sorted IDs that contain a field. */
  [[nodiscard]] static auto difference(const std::vector<IDType> &present,
                                       const std::vector<IDType> &excluded) -> std::vector<IDType> {
    std::vector<IDType> result;
    result.reserve(present.size());
    std::set_difference(present.begin(),
                        present.end(),
                        excluded.begin(),
                        excluded.end(),
                        std::back_inserter(result));
    return result;
  }

  /** @brief Resolve one typed numeric range from lexicographically sortable encoded values. */
  [[nodiscard]] static auto range_lookup(const FieldIndex &field,
                                         FilterOp op,
                                         const MetadataValue &value)
      -> std::optional<std::vector<IDType>> {
    std::string prefix;
    if (std::holds_alternative<int64_t>(value)) {
      prefix = "i_";
    } else if (std::holds_alternative<double>(value)) {
      prefix = "d_";
    } else {
      return std::nullopt;
    }

    auto boundary = index_encoding::encode_value(value);
    std::vector<IDType> result;
    for (auto it = field.postings_.lower_bound(prefix);
         it != field.postings_.end() && starts_with(it->first, prefix);
         ++it) {
      const auto &encoded = it->first;
      bool include = (op == FilterOp::GE && encoded >= boundary) ||
                     (op == FilterOp::GT && encoded > boundary) ||
                     (op == FilterOp::LE && encoded <= boundary) ||
                     (op == FilterOp::LT && encoded < boundary);
      if (include) {
        result.insert(result.end(), it->second.begin(), it->second.end());
      }
    }
    normalize_ids(result);
    return result;
  }

  /** @brief Test an encoded-value type prefix without allocating a substring. */
  [[nodiscard]] static auto starts_with(const std::string &value, std::string_view prefix) -> bool {
    return value.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), value.begin());
  }

  uint64_t generation_ = 0;  ///< Persisted record generation represented by every posting.
  DynamicBitset live_mask_;  ///< Set bits identify live internal IDs in this generation.
  size_t live_count_ = 0;    ///< Number of set bits in live_mask_.
  std::unordered_map<std::string, FieldIndex> fields_;  ///< Configured query-hot field postings.
};

}  // namespace alaya
