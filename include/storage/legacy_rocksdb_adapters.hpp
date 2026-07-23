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
#include <vector>

#include "scalar/scalar_index.hpp"
#include "storage/record_store.hpp"
#include "storage/rocksdb_storage.hpp"

namespace alaya {

/** @brief Adapts the current RocksDB secondary indexes to ScalarIndex. */
template <typename IDType>
class LegacyRocksDBScalarIndex final : public ScalarIndex<IDType> {
 public:
  /** @brief Bind the adapter to a RocksDB storage instance without taking ownership. */
  explicit LegacyRocksDBScalarIndex(const RocksDBStorage<IDType> *storage) : storage_(storage) {
    if (storage_ == nullptr) {
      throw std::invalid_argument("Storage cannot be null");
    }
  }

  /** @copydoc ScalarIndex::is_indexed_field */
  [[nodiscard]] auto is_indexed_field(const std::string &field) const -> bool override {
    const auto &fields = storage_->config().indexed_fields_;
    return std::find(fields.begin(), fields.end(), field) != fields.end();
  }

  /** @copydoc ScalarIndex::generation */
  [[nodiscard]] auto generation() const -> uint64_t override { return 0; }

  /** @copydoc ScalarIndex::lookup */
  [[nodiscard]] auto lookup(const FilterCondition &condition) const
      -> std::optional<std::vector<IDType>> override {
    if (!is_indexed_field(condition.field)) {
      return std::nullopt;
    }

    std::vector<IDType> ids;
    switch (condition.op) {
      case FilterOp::EQ:
        return storage_->get_ids_by_field_value(condition.field, condition.value);
      case FilterOp::IN_SET:
        for (const auto &value : condition.values) {
          auto partial = storage_->get_ids_by_field_value(condition.field, value);
          ids.insert(ids.end(), partial.begin(), partial.end());
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        return ids;
      case FilterOp::GE:
        return lookup_lower_bound(condition, false);
      case FilterOp::GT:
        return lookup_lower_bound(condition, true);
      case FilterOp::LE:
        return lookup_upper_bound(condition, false);
      case FilterOp::LT:
        return lookup_upper_bound(condition, true);
      default:
        return std::nullopt;
    }
  }

 private:
  /** @brief Resolve GE or GT for the numeric type carried by a condition. */
  [[nodiscard]] auto lookup_lower_bound(const FilterCondition &condition, bool exclusive) const
      -> std::optional<std::vector<IDType>> {
    if (std::holds_alternative<int64_t>(condition.value)) {
      auto value = std::get<int64_t>(condition.value);
      if (exclusive && value == std::numeric_limits<int64_t>::max()) {
        return std::vector<IDType>{};
      }
      return storage_->get_ids_by_int_range(condition.field,
                                            exclusive ? value + 1 : value,
                                            std::numeric_limits<int64_t>::max());
    }
    if (std::holds_alternative<double>(condition.value)) {
      auto value = std::get<double>(condition.value);
      return storage_
          ->get_ids_by_double_range(condition.field,
                                    exclusive
                                        ? std::nextafter(value, std::numeric_limits<double>::max())
                                        : value,
                                    std::numeric_limits<double>::max());
    }
    return std::nullopt;
  }

  /** @brief Resolve LE or LT for the numeric type carried by a condition. */
  [[nodiscard]] auto lookup_upper_bound(const FilterCondition &condition, bool exclusive) const
      -> std::optional<std::vector<IDType>> {
    if (std::holds_alternative<int64_t>(condition.value)) {
      auto value = std::get<int64_t>(condition.value);
      if (exclusive && value == std::numeric_limits<int64_t>::min()) {
        return std::vector<IDType>{};
      }
      return storage_->get_ids_by_int_range(condition.field,
                                            std::numeric_limits<int64_t>::min(),
                                            exclusive ? value - 1 : value);
    }
    if (std::holds_alternative<double>(condition.value)) {
      auto value = std::get<double>(condition.value);
      return storage_
          ->get_ids_by_double_range(condition.field,
                                    std::numeric_limits<double>::lowest(),
                                    exclusive
                                        ? std::nextafter(value,
                                                         std::numeric_limits<double>::lowest())
                                        : value);
    }
    return std::nullopt;
  }

  const RocksDBStorage<IDType> *storage_;  ///< Non-owning legacy index provider.
};

/** @brief Adapts the current RocksDB scalar records to RecordStore. */
template <typename IDType>
class LegacyRocksDBRecordStore final : public RecordStore<IDType> {
 public:
  /** @brief Bind the adapter to a RocksDB storage instance without taking ownership. */
  explicit LegacyRocksDBRecordStore(const RocksDBStorage<IDType> *storage) : storage_(storage) {
    if (storage_ == nullptr) {
      throw std::invalid_argument("Storage cannot be null");
    }
  }

  /** @copydoc RecordStore::get_raw_scalar */
  auto get_raw_scalar(IDType id, std::string &value) const -> bool override {
    return storage_->get_raw_value(id, value);
  }

  /** @copydoc RecordStore::batch_get_raw_scalars */
  [[nodiscard]] auto batch_get_raw_scalars(const std::vector<IDType> &ids) const
      -> std::vector<std::string> override {
    return storage_->batch_get_raw_values(ids);
  }

  /** @copydoc RecordStore::find_by_item_id */
  [[nodiscard]] auto find_by_item_id(const std::string &item_id) const
      -> std::optional<IDType> override {
    return storage_->find_by_item_id(item_id);
  }

  /** @copydoc RecordStore::size */
  [[nodiscard]] auto size() const -> size_t override { return storage_->count(); }

  /** @copydoc RecordStore::generation */
  [[nodiscard]] auto generation() const -> uint64_t override { return 0; }

 private:
  const RocksDBStorage<IDType> *storage_;  ///< Non-owning canonical record provider.
};

}  // namespace alaya
