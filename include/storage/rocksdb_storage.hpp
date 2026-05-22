// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <rocksdb/cache.h>
#include <rocksdb/db.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/table.h>
#include <rocksdb/table_properties.h>
#include <rocksdb/utilities/checkpoint.h>
#include <rocksdb/write_batch.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils/index_encoding.hpp"
#include "utils/log.hpp"
#include "utils/query_utils.hpp"
#include "utils/scalar_data.hpp"

namespace alaya {

/**
 * @brief Configuration for RocksDB storage
 */
struct RocksDBConfig {
  std::string db_path_ = "./RocksDB/alayalite_rocksdb";

  bool create_if_missing_ = true;
  bool error_if_exists_ = false;

  size_t write_buffer_size_ = static_cast<size_t>(64) << 20;  // 64MB
  int max_write_buffer_number_ = 4;
  size_t target_file_size_base_ = static_cast<size_t>(64) << 20;  // 64MB
  int max_background_compactions_ = 4;
  int max_background_flushes_ = 2;
  size_t block_cache_size_mb_ = 512;  // 512MB
  bool enable_compression_ = false;   // Enable LZ4+ZSTD compression by default

  std::vector<std::string> indexed_fields_;  // Fields to create secondary indexes for

  static auto default_config() -> RocksDBConfig { return RocksDBConfig{}; }
};

/**
 * @brief RocksDB-based storage for ScalarData (item_id, document, metadata)
 *
 * IDs are managed externally by Space to ensure consistency with vector storage.
 * Supports secondary indexing by item_id for efficient lookups.
 *
 * Key schema:
 * - "d_{id}" -> ScalarData (primary data)
 * - "i_{item_id}" -> internal_id (secondary index)
 * - "f_{field}_{value}_{id}" -> "" (field index for fast filtering)
 * - "__COUNT__" -> record count
 *
 * @tparam IDType The type used for internal IDs (default: uint32_t)
 */
template <typename IDType = uint32_t>
class RocksDBStorage {
 public:
  explicit RocksDBStorage(RocksDBConfig config = RocksDBConfig::default_config())
      : config_(std::move(config)), cached_count_(0) {
    initialize_db();
  }

  ~RocksDBStorage() { close_db(); }

  RocksDBStorage(const RocksDBStorage &) = delete;
  auto operator=(const RocksDBStorage &) -> RocksDBStorage & = delete;

  RocksDBStorage(RocksDBStorage &&other) noexcept
      : db_(std::move(other.db_)),
        config_(std::move(other.config_)),
        cached_count_(other.cached_count_.load()),
        read_only_(other.read_only_) {
    other.read_only_ = false;
  }

  auto operator=(RocksDBStorage &&other) noexcept -> RocksDBStorage & {
    if (this != &other) {
      close_db();
      db_ = std::move(other.db_);
      config_ = std::move(other.config_);
      cached_count_.store(other.cached_count_.load());
      read_only_ = other.read_only_;
      invalidate_range_index_cache();
      other.read_only_ = false;
    }
    return *this;
  }

  struct IntRangeIndexEntry {
    int64_t value_;
    IDType id_;
  };

  using IntRangeIndex = std::vector<IntRangeIndexEntry>;

  struct IntRangeIndexRange {
    std::shared_ptr<const IntRangeIndex> entries_;
    size_t begin_ = 0;
    size_t end_ = 0;
  };

  struct CachedBlockedBitset {
    std::shared_ptr<const DynamicBitset> blocked_;
    size_t matched_count_ = 0;
  };

  /**
   * @brief Get ScalarData by internal ID
   * @param id Internal ID
   * @return ScalarData (empty if not found)
   */
  [[nodiscard]] auto operator[](IDType id) const -> ScalarData {  // redundant?
    std::string value;
    if (!get_data_value(id, &value)) {
      LOG_ERROR("Failed to access ScalarData for ID {}", id);
      return ScalarData{};
    }

    return ScalarData::deserialize(value.data(), value.size());
  }

  /**
   * @brief Get raw serialized value by internal ID.
   * @param id Internal ID
   * @param value Output serialized bytes
   * @return true if found
   */
  auto get_raw_value(IDType id, std::string &value) const -> bool {
    return get_data_value(id, &value);
  }

  /**
   * @brief Batch get raw serialized values by internal IDs.
   *
   * Uses RocksDB MultiGet and returns empty strings for missing entries.
   *
   * @param ids Vector of internal IDs
   * @return Vector of serialized ScalarData payloads
   */
  [[nodiscard]] auto batch_get_raw_values(const std::vector<IDType> &ids) const
      -> std::vector<std::string> {
    auto stored_values = batch_get_data_values(ids);
    std::vector<std::string> values;
    values.reserve(stored_values.size());
    for (auto &stored_value : stored_values) {
      values.push_back(stored_value.found ? std::move(stored_value.value) : std::string{});
    }
    return values;
  }

  /**
   * @brief Check if an ID exists
   */
  [[nodiscard]] auto is_valid(IDType id) const -> bool {
    std::string value;
    return get_data_value(id, &value);
  }

  /**
   * @brief Batch get only item_ids by internal IDs (lightweight batch operation)
   *
   * Uses RocksDB MultiGet for efficiency. Only deserializes the item_id field
   * from each ScalarData, avoiding the overhead of full deserialization.
   *
   * @param ids Vector of internal IDs
   * @return Vector of item_id strings (empty string for not-found entries)
   */
  [[nodiscard]] auto batch_get_item_id_only(const std::vector<IDType> &ids) const
      -> std::vector<std::string> {
    std::vector<std::string> results;
    results.reserve(ids.size());
    auto values = batch_get_raw_values(ids);

    for (size_t i = 0; i < ids.size(); ++i) {
      if (values[i].size() < sizeof(uint32_t)) {
        results.emplace_back();
        continue;
      }
      // Parse only item_id: [uint32_t length][string data]
      size_t offset = 0;
      uint32_t len;
      std::memcpy(&len, values[i].data() + offset, sizeof(len));
      offset += sizeof(len);
      if (offset + len > values[i].size()) {
        results.emplace_back();
      } else {
        results.emplace_back(values[i].data() + offset, len);
      }
    }

    return results;
  }

  /**
   * @brief Insert ScalarData with specified ID (managed by Space)
   * @param id Internal ID
   * @param data ScalarData to insert
   * @return true on success
   */
  auto insert(IDType id, const ScalarData &data) -> bool {
    std::lock_guard<std::mutex> lock(write_mutex_);
    ensure_writable("insert");
    std::string key = data_key(id);
    std::string old_serialized;
    std::string resolved_key;
    std::optional<ScalarData> old_data;
    bool replacing_existing = get_data_value(id, &old_serialized, &resolved_key);
    if (replacing_existing) {
      old_data = ScalarData::deserialize(old_serialized.data(), old_serialized.size());
    }

    if (!item_id_available_for_locked(data.item_id, id)) {
      LOG_ERROR("Failed to insert ScalarData for ID {}: duplicate item_id '{}'", id, data.item_id);
      return false;
    }

    auto serialized = data.serialize();
    rocksdb::Slice value_slice(serialized.data(), serialized.size());

    rocksdb::WriteBatch batch;
    if (old_data.has_value()) {
      if (resolved_key != key) {
        batch.Delete(resolved_key);
      }
      delete_item_id_index(batch, old_data->item_id);
      remove_field_indexes(batch, id, *old_data);
    }
    batch.Put(key, value_slice);

    // Add secondary index: item_id -> internal_id
    if (!data.item_id.empty()) {
      std::string index_key = item_id_index_key(data.item_id);
      rocksdb::Slice id_slice(reinterpret_cast<const char *>(&id), sizeof(IDType));
      batch.Put(index_key, id_slice);
    }

    // Add field indexes for indexed fields
    add_field_indexes(batch, id, data);

    rocksdb::Status status = db_->Write(rocksdb::WriteOptions(), &batch);
    if (!status.ok()) {
      LOG_ERROR("Failed to insert ScalarData for ID {}: {}", id, status.ToString());
      return false;
    }

    if (!replacing_existing) {
      ++cached_count_;
    }
    invalidate_range_index_cache();
    return true;
  }

  /**
   * @brief Batch insert ScalarData starting from specified ID
   *
   * IDs are assigned sequentially: start_id, start_id+1, start_id+2, ...
   * This must align with how Space assigns vector storage IDs.
   *
   * @param start_id Starting internal ID
   * @param begin Iterator to first ScalarData
   * @param end Iterator past last ScalarData
   * @return true on success
   */
  template <typename Iterator>
  auto batch_insert(IDType start_id, Iterator begin, Iterator end) -> bool {
    std::lock_guard<std::mutex> lock(write_mutex_);
    ensure_writable("batch_insert");
    struct PendingRecord {
      IDType id;
      ScalarData data;
      std::optional<ScalarData> old_data;
      std::string resolved_key;
    };

    std::vector<PendingRecord> records;
    std::unordered_map<std::string, IDType> batch_item_ids;
    std::vector<IDType> preflight_ids;
    std::vector<std::string> preflight_item_ids;

    IDType preflight_id = start_id;
    for (auto it = begin; it != end; ++it, ++preflight_id) {
      PendingRecord record{preflight_id, *it, std::nullopt, {}};
      preflight_ids.push_back(preflight_id);

      if (!record.data.item_id.empty()) {
        if (!batch_item_ids.emplace(record.data.item_id, preflight_id).second) {
          LOG_ERROR("Batch insert failed: duplicate item_id '{}' in batch", record.data.item_id);
          return false;
        }
        preflight_item_ids.push_back(record.data.item_id);
      }

      records.push_back(std::move(record));
    }

    auto existing_records = batch_get_data_values(preflight_ids);
    for (size_t i = 0; i < records.size(); ++i) {
      if (existing_records[i].found) {
        records[i].resolved_key = std::move(existing_records[i].resolved_key);
        records[i].old_data = ScalarData::deserialize(existing_records[i].value.data(),
                                                      existing_records[i].value.size());
      }
    }

    auto existing_item_owners = batch_find_item_id_owners(preflight_item_ids);
    for (const auto &record : records) {
      if (record.data.item_id.empty()) {
        continue;
      }
      auto existing = existing_item_owners.find(record.data.item_id);
      if (existing != existing_item_owners.end() && existing->second != record.id) {
        LOG_ERROR("Batch insert failed: duplicate item_id '{}'", record.data.item_id);
        return false;
      }
    }

    rocksdb::WriteBatch batch;
    size_t inserted_count = 0;

    for (const auto &record : records) {
      std::string key = data_key(record.id);
      if (record.old_data.has_value()) {
        if (record.resolved_key != key) {
          batch.Delete(record.resolved_key);
        }
        delete_item_id_index(batch, record.old_data->item_id);
        remove_field_indexes(batch, record.id, *record.old_data);
      } else {
        ++inserted_count;
      }

      auto serialized = record.data.serialize();
      batch.Put(key, rocksdb::Slice(serialized.data(), serialized.size()));

      // Add secondary index
      if (!record.data.item_id.empty()) {
        std::string idx_key = item_id_index_key(record.data.item_id);
        rocksdb::Slice id_slice(reinterpret_cast<const char *>(&record.id), sizeof(IDType));
        batch.Put(idx_key, id_slice);
      }

      // Add field indexes for indexed fields
      add_field_indexes(batch, record.id, record.data);
    }

    rocksdb::Status status = db_->Write(rocksdb::WriteOptions(), &batch);
    if (!status.ok()) {
      LOG_ERROR("Batch insert failed: {}", status.ToString());
      return false;
    }

    cached_count_ += inserted_count;
    invalidate_range_index_cache();
    return true;
  }

  /**
   * @brief Remove ScalarData by ID
   */
  auto remove(IDType id) -> bool {
    std::lock_guard<std::mutex> lock(write_mutex_);
    ensure_writable("remove");

    std::string serialized;
    std::string resolved_key;
    if (!get_data_value(id, &serialized, &resolved_key)) {
      LOG_ERROR("Failed to remove ID({}) that doesn't exist.", id);
      return false;
    }

    auto data = ScalarData::deserialize(serialized.data(), serialized.size());

    rocksdb::WriteBatch batch;
    batch.Delete(resolved_key);

    delete_item_id_index(batch, data.item_id);

    // Remove field indexes
    remove_field_indexes(batch, id, data);

    rocksdb::Status status = db_->Write(rocksdb::WriteOptions(), &batch);
    if (!status.ok()) {
      LOG_ERROR("Failed to remove ID {}: {}", id, status.ToString());
      return false;
    }

    if (cached_count_ > 0) {
      --cached_count_;
    }
    invalidate_range_index_cache();
    return true;
  }

  /**
   * @brief Update ScalarData
   */
  auto update(IDType id, const ScalarData &data) -> bool {
    std::lock_guard<std::mutex> lock(write_mutex_);
    ensure_writable("update");

    std::string serialized_value;
    std::string resolved_key;
    if (!get_data_value(id, &serialized_value, &resolved_key)) {
      LOG_ERROR("Failed to update ID({}) that doesn't exist.", id);
      return false;
    }
    auto old_data = ScalarData::deserialize(serialized_value.data(), serialized_value.size());
    if (!item_id_available_for_locked(data.item_id, id)) {
      LOG_ERROR("Failed to update ID {}: duplicate item_id '{}'", id, data.item_id);
      return false;
    }

    rocksdb::WriteBatch batch;

    // Update primary data
    auto serialized = data.serialize();
    auto canonical_key = data_key(id);
    if (resolved_key != canonical_key) {
      batch.Delete(resolved_key);
    }
    batch.Put(canonical_key, rocksdb::Slice(serialized.data(), serialized.size()));

    // Update secondary index if item_id changed
    if (old_data.item_id != data.item_id) {
      delete_item_id_index(batch, old_data.item_id);
      if (!data.item_id.empty()) {
        rocksdb::Slice id_slice(reinterpret_cast<const char *>(&id), sizeof(IDType));
        batch.Put(item_id_index_key(data.item_id), id_slice);
      }
    }

    // Update field indexes: remove old, add new
    remove_field_indexes(batch, id, old_data);
    add_field_indexes(batch, id, data);

    rocksdb::Status status = db_->Write(rocksdb::WriteOptions(), &batch);
    if (!status.ok()) {
      LOG_ERROR("Failed to update ID {}: {}", id, status.ToString());
      return false;
    }

    invalidate_range_index_cache();
    return true;
  }

  /**
   * @brief Find internal ID by item_id
   */
  [[nodiscard]] auto find_by_item_id(const std::string &item_id) const -> std::optional<IDType> {
    std::string key = item_id_index_key(item_id);
    std::string value;
    rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), key, &value);

    if (!status.ok() || value.size() != sizeof(IDType)) {
      return std::nullopt;
    }

    IDType id;
    std::memcpy(&id, value.data(), sizeof(IDType));
    return id;
  }

  [[nodiscard]] auto item_id_available(const std::string &item_id,
                                       std::optional<IDType> allowed_id = std::nullopt) const
      -> bool {
    if (item_id.empty()) {
      return true;
    }
    auto existing = find_by_item_id(item_id);
    return !existing.has_value() || (allowed_id.has_value() && *existing == *allowed_id);
  }

  /**
   * @brief Batch get ScalarData by IDs
   */
  [[nodiscard]] auto batch_get(const std::vector<IDType> &ids) const -> std::vector<ScalarData> {
    std::vector<ScalarData> results;
    results.reserve(ids.size());
    auto values = batch_get_raw_values(ids);

    for (size_t i = 0; i < ids.size(); ++i) {
      if (!values[i].empty()) {
        results.push_back(ScalarData::deserialize(values[i].data(), values[i].size()));
      } else {
        results.emplace_back();
      }
    }

    return results;
  }

  [[nodiscard]] auto count() const -> size_t { return cached_count_.load(); }

  [[nodiscard]] auto is_read_only() const -> bool { return read_only_; }

  /**
   * @brief Scan *ALL* ScalarData with a filter function
   * @param filter_fn Filter function, return true to include the record
   * @param limit Maximum number of results (0 = no limit)
   * @return Vector of (internal_id, ScalarData) pairs
   */
  [[nodiscard]] auto scan_with_filter(const std::function<bool(const ScalarData &)> &filter_fn,
                                      size_t limit = 0) const
      -> std::vector<std::pair<IDType, ScalarData>> {
    // TODO(review - filter scan performance): route equality/range-capable predicates through the
    // secondary indexes first and keep this full deserialize-everything scan only as a fallback.
    std::vector<std::pair<IDType, ScalarData>> results;

    rocksdb::ReadOptions read_opts;
    read_opts.fill_cache = false;

    std::unique_ptr<rocksdb::Iterator> iter(db_->NewIterator(read_opts));

    for (iter->Seek("d_"); iter->Valid(); iter->Next()) {
      auto key = iter->key().ToString();
      if (!key.starts_with(data_key_prefix())) {
        break;
      }

      auto id = parse_data_key(key);
      if (!id.has_value()) {
        continue;
      }
      auto value = iter->value();
      ScalarData sd = ScalarData::deserialize(value.data(), value.size());

      if (filter_fn(sd)) {
        results.emplace_back(*id, std::move(sd));
      }
    }

    std::sort(results.begin(), results.end(), [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    if (limit > 0 && results.size() > limit) {
      results.resize(limit);
    }

    return results;
  }

  /**
   * @brief Get IDs by exact field value match using index
   * @param field Field name (must be in indexed_fields_)
   * @param value Field value to match
   * @return Vector of matching internal IDs
   */
  [[nodiscard]] auto get_ids_by_field_value(const std::string &field,
                                            const MetadataValue &value) const
      -> std::vector<IDType> {
    std::vector<IDType> ids;
    visit_ids_by_field_value(field, value, [&ids](IDType id) {
      ids.push_back(id);
    });
    return ids;
  }

  template <typename Visitor>
  void visit_ids_by_field_value(const std::string &field,
                                const MetadataValue &value,
                                Visitor &&visitor) const {
    std::string encoded = index_encoding::encode_value(value);
    std::string prefix = index_encoding::make_field_index_prefix(field, encoded);

    rocksdb::ReadOptions read_opts;
    read_opts.fill_cache = false;
    auto upper_bound = prefix_upper_bound(prefix);
    rocksdb::Slice upper_bound_slice;
    if (upper_bound.has_value()) {
      upper_bound_slice = rocksdb::Slice(*upper_bound);
      read_opts.iterate_upper_bound = &upper_bound_slice;
    }
    std::unique_ptr<rocksdb::Iterator> iter(db_->NewIterator(read_opts));

    for (iter->Seek(prefix); iter->Valid(); iter->Next()) {
      auto key = iter->key();
      if (!slice_starts_with(key, prefix)) {
        break;
      }
      IDType id = extract_id_from_key_slice(key);
      visitor(id);
    }
  }

  /**
   * @brief Get IDs by int64 range query using index
   * @param field Field name (must be in indexed_fields_)
   * @param min_value Minimum value (inclusive)
   * @param max_value Maximum value (inclusive)
   * @return Vector of matching internal IDs
   */
  [[nodiscard]] auto get_ids_by_int_range(const std::string &field,
                                          int64_t min_value,
                                          int64_t max_value) const -> std::vector<IDType> {
    if (auto range = get_int_range_index_range(field, min_value, max_value)) {
      std::vector<IDType> ids;
      ids.reserve(range->end_ - range->begin_);
      for (size_t i = range->begin_; i < range->end_; ++i) {
        ids.push_back((*range->entries_)[i].id_);
      }
      return ids;
    }

    std::vector<IDType> ids;
    visit_ids_by_int_range(field, min_value, max_value, [&ids](IDType id) {
      ids.push_back(id);
    });
    return ids;
  }

  [[nodiscard]] auto get_int_range_index_range(const std::string &field,
                                               int64_t min_value,
                                               int64_t max_value) const
      -> std::optional<IntRangeIndexRange> {
    if (min_value > max_value) {
      return IntRangeIndexRange{std::make_shared<IntRangeIndex>(), 0, 0};
    }

    auto entries = get_or_build_int_range_index(field);
    if (entries == nullptr) {
      return std::nullopt;
    }

    auto lower = std::lower_bound(entries->begin(),
                                  entries->end(),
                                  min_value,
                                  [](const IntRangeIndexEntry &entry, int64_t value) {
                                    return entry.value_ < value;
                                  });
    auto upper = std::upper_bound(entries->begin(),
                                  entries->end(),
                                  max_value,
                                  [](int64_t value, const IntRangeIndexEntry &entry) {
                                    return value < entry.value_;
                                  });

    return IntRangeIndexRange{
        entries,
        static_cast<size_t>(std::distance(entries->begin(), lower)),
        static_cast<size_t>(std::distance(entries->begin(), upper)),
    };
  }

  [[nodiscard]] auto get_int_range_blocked_bitset(const std::string &field,
                                                  int64_t min_value,
                                                  int64_t max_value,
                                                  size_t data_num) const
      -> std::optional<CachedBlockedBitset> {
    auto cache_key = int_range_bitset_cache_key(field, min_value, max_value, data_num);
    {
      std::lock_guard<std::mutex> lock(range_index_cache_mutex_);
      auto cached = int_range_bitset_cache_.find(cache_key);
      if (cached != int_range_bitset_cache_.end()) {
        return cached->second;
      }
    }

    auto range = get_int_range_index_range(field, min_value, max_value);
    if (!range.has_value()) {
      return std::nullopt;
    }

    auto built = build_int_range_blocked_bitset(*range, data_num);
    {
      std::lock_guard<std::mutex> lock(range_index_cache_mutex_);
      auto [it, inserted] = int_range_bitset_cache_.emplace(cache_key, built);
      (void)inserted;
      return it->second;
    }
  }

  template <typename Visitor>
  void visit_ids_by_int_range(const std::string &field,
                              int64_t min_value,
                              int64_t max_value,
                              Visitor &&visitor) const {
    if (min_value > max_value) {
      return;
    }

    std::string field_prefix = index_encoding::make_field_prefix(field);
    std::string typed_prefix = field_prefix + "i_";
    std::string start_key = typed_prefix + index_encoding::encode_int64(min_value);
    std::string end_encoded = index_encoding::encode_int64(max_value);
    std::string end_key_prefix = typed_prefix + end_encoded + "_";

    rocksdb::ReadOptions read_opts;
    read_opts.fill_cache = false;
    auto upper_bound = prefix_upper_bound(end_key_prefix);
    rocksdb::Slice upper_bound_slice;
    if (upper_bound.has_value()) {
      upper_bound_slice = rocksdb::Slice(*upper_bound);
      read_opts.iterate_upper_bound = &upper_bound_slice;
    }
    std::unique_ptr<rocksdb::Iterator> iter(db_->NewIterator(read_opts));

    for (iter->Seek(start_key); iter->Valid(); iter->Next()) {
      auto key = iter->key();
      if (!slice_starts_with(key, typed_prefix)) {
        break;
      }
      IDType id = extract_id_from_key_slice(key);
      visitor(id);
    }
  }

  /**
   * @brief Get IDs by double range query using index
   * @param field Field name (must be in indexed_fields_)
   * @param min_value Minimum value (inclusive)
   * @param max_value Maximum value (inclusive)
   * @return Vector of matching internal IDs
   */
  [[nodiscard]] auto get_ids_by_double_range(const std::string &field,
                                             double min_value,
                                             double max_value) const -> std::vector<IDType> {
    std::vector<IDType> ids;
    visit_ids_by_double_range(field, min_value, max_value, [&ids](IDType id) {
      ids.push_back(id);
    });
    return ids;
  }

  template <typename Visitor>
  void visit_ids_by_double_range(const std::string &field,
                                 double min_value,
                                 double max_value,
                                 Visitor &&visitor) const {
    if (min_value > max_value) {
      return;
    }

    std::string field_prefix = index_encoding::make_field_prefix(field);
    std::string typed_prefix = field_prefix + "d_";
    std::string start_key = typed_prefix + index_encoding::encode_double(min_value);
    std::string end_encoded = index_encoding::encode_double(max_value);
    std::string end_key_prefix = typed_prefix + end_encoded + "_";

    rocksdb::ReadOptions read_opts;
    read_opts.fill_cache = false;
    auto upper_bound = prefix_upper_bound(end_key_prefix);
    rocksdb::Slice upper_bound_slice;
    if (upper_bound.has_value()) {
      upper_bound_slice = rocksdb::Slice(*upper_bound);
      read_opts.iterate_upper_bound = &upper_bound_slice;
    }
    std::unique_ptr<rocksdb::Iterator> iter(db_->NewIterator(read_opts));

    for (iter->Seek(start_key); iter->Valid(); iter->Next()) {
      auto key = iter->key();
      if (!slice_starts_with(key, typed_prefix)) {
        break;
      }
      IDType id = extract_id_from_key_slice(key);
      visitor(id);
    }
  }

  [[nodiscard]] auto config() const -> const RocksDBConfig & { return config_; }

  [[nodiscard]] auto get_db_path() const -> const std::string & { return config_.db_path_; }

  void flush() const {
    ensure_writable("flush");
    if (db_ != nullptr) {
      save_count();
      db_->Flush(rocksdb::FlushOptions());
    }
  }

  void compact() {
    ensure_writable("compact");
    if (db_ != nullptr) {
      db_->CompactRange(rocksdb::CompactRangeOptions(), nullptr, nullptr);
    }
  }

  [[nodiscard]] auto get_statistics() const -> std::string {
    if (db_ == nullptr) {
      return "";
    }
    std::string stats;
    db_->GetProperty("rocksdb.stats", &stats);
    return stats;
  }

  void save(const std::string &filepath) const {
    if (!read_only_) {
      flush();
    }

    rocksdb::Checkpoint *checkpoint_raw = nullptr;
    rocksdb::Status status = rocksdb::Checkpoint::Create(db_.get(), &checkpoint_raw);
    std::unique_ptr<rocksdb::Checkpoint> checkpoint(checkpoint_raw);
    if (!status.ok()) {
      throw std::runtime_error("Failed to create RocksDB checkpoint: " + status.ToString());
    }

    status = checkpoint->CreateCheckpoint(filepath);

    if (!status.ok()) {
      throw std::runtime_error("Failed to save RocksDB checkpoint to " + filepath + ": " +
                               status.ToString());
    }
  }

 private:
  void close_db() noexcept {
    if (db_ == nullptr) {
      return;
    }

    if (!read_only_) {
      try {
        save_count();
      } catch (const std::exception &e) {
        LOG_ERROR("Failed to persist RocksDB count during close: {}", e.what());
      }
    }

    rocksdb::Status status = db_->Close();
    if (!status.ok()) {
      LOG_ERROR("Failed to close RocksDB: {}", status.ToString());
    }
    db_.reset();
  }

  void ensure_writable(const char *operation) const {
    if (db_ == nullptr) {
      throw std::runtime_error(std::string("RocksDB is not open for ") + operation);
    }
    if (read_only_) {
      throw std::runtime_error(std::string("RocksDB storage is read-only; cannot ") + operation);
    }
  }

  static constexpr auto data_key_prefix() -> std::string_view { return "d_"; }

  using UnsignedIDType = std::make_unsigned_t<IDType>;
  static constexpr size_t kSortableDigits = std::numeric_limits<UnsignedIDType>::digits10 + 1;

  struct StoredValue {
    bool found{false};
    std::string value;
    std::string resolved_key;
  };

  static auto data_key(IDType id) -> std::string {
    auto digits = std::to_string(static_cast<UnsignedIDType>(id));
    if (digits.size() >= kSortableDigits) {
      return std::string(data_key_prefix()) + digits;
    }
    return std::string(data_key_prefix()) + std::string(kSortableDigits - digits.size(), '0') +
           digits;
  }

  static auto legacy_data_key(IDType id) -> std::string {
    return std::string(data_key_prefix()) + std::to_string(static_cast<UnsignedIDType>(id));
  }

  static auto slice_starts_with(const rocksdb::Slice &key, std::string_view prefix) -> bool {
    return key.size() >= prefix.size() &&
           std::memcmp(key.data(), prefix.data(), prefix.size()) == 0;
  }

  static auto prefix_upper_bound(std::string_view prefix) -> std::optional<std::string> {
    std::string upper(prefix);
    for (size_t i = upper.size(); i > 0; --i) {
      auto byte = static_cast<unsigned char>(upper[i - 1]);
      if (byte == 0xFFU) {
        continue;
      }
      upper[i - 1] = static_cast<char>(byte + 1U);
      upper.resize(i);
      return upper;
    }
    return std::nullopt;
  }

  static auto extract_id_from_key_slice(const rocksdb::Slice &key) -> IDType {
    return index_encoding::extract_id_from_key<IDType>(std::string_view(key.data(), key.size()));
  }

  static auto int_range_bitset_cache_key(const std::string &field,
                                         int64_t min_value,
                                         int64_t max_value,
                                         size_t data_num) -> std::string {
    return std::to_string(field.size()) + ":" + field + ":" + std::to_string(min_value) + ":" +
           std::to_string(max_value) + ":" + std::to_string(data_num);
  }

  static auto parse_int_field_index_key(std::string_view key, size_t prefix_size)
      -> std::optional<IntRangeIndexEntry> {
    constexpr size_t kEncodedInt64Size = 16;
    if (key.size() <= prefix_size + kEncodedInt64Size ||
        key[prefix_size + kEncodedInt64Size] != '_') {
      return std::nullopt;
    }

    auto encoded = std::string(key.substr(prefix_size, kEncodedInt64Size));
    auto value = index_encoding::decode_int64(encoded);
    auto id = index_encoding::extract_id_from_key<IDType>(key);
    return IntRangeIndexEntry{value, id};
  }

  [[nodiscard]] auto get_or_build_int_range_index(const std::string &field) const
      -> std::shared_ptr<const IntRangeIndex> {
    {
      std::lock_guard<std::mutex> lock(range_index_cache_mutex_);
      auto cached = int_range_indexes_.find(field);
      if (cached != int_range_indexes_.end()) {
        return cached->second;
      }
    }

    auto built = build_int_range_index(field);
    {
      std::lock_guard<std::mutex> lock(range_index_cache_mutex_);
      auto [it, inserted] = int_range_indexes_.emplace(field, built);
      (void)inserted;
      return it->second;
    }
  }

  [[nodiscard]] auto build_int_range_index(const std::string &field) const
      -> std::shared_ptr<const IntRangeIndex> {
    auto entries = std::make_shared<IntRangeIndex>();
    std::string prefix = index_encoding::make_field_prefix(field) + "i_";

    rocksdb::ReadOptions read_opts;
    read_opts.fill_cache = false;
    auto upper_bound = prefix_upper_bound(prefix);
    rocksdb::Slice upper_bound_slice;
    if (upper_bound.has_value()) {
      upper_bound_slice = rocksdb::Slice(*upper_bound);
      read_opts.iterate_upper_bound = &upper_bound_slice;
    }

    std::unique_ptr<rocksdb::Iterator> iter(db_->NewIterator(read_opts));
    for (iter->Seek(prefix); iter->Valid(); iter->Next()) {
      auto key = iter->key();
      if (!slice_starts_with(key, prefix)) {
        break;
      }

      auto parsed =
          parse_int_field_index_key(std::string_view(key.data(), key.size()), prefix.size());
      if (parsed.has_value()) {
        entries->push_back(*parsed);
      }
    }

    std::sort(entries->begin(), entries->end(), [](const auto &lhs, const auto &rhs) {
      if (lhs.value_ != rhs.value_) {
        return lhs.value_ < rhs.value_;
      }
      return lhs.id_ < rhs.id_;
    });

    return entries;
  }

  [[nodiscard]] auto build_int_range_blocked_bitset(const IntRangeIndexRange &range,
                                                    size_t data_num) const -> CachedBlockedBitset {
    DynamicBitset blocked(data_num);
    auto entries = range.entries_;
    auto hit_count = range.end_ - range.begin_;

    if (hit_count == 0) {
      blocked.set_all();
      return CachedBlockedBitset{
          std::make_shared<DynamicBitset>(std::move(blocked)),
          0,
      };
    }

    if (entries->size() == data_num && hit_count > entries->size() / 2) {
      size_t matched_count = data_num;
      auto block_id = [&](IDType id) {
        auto raw_id = static_cast<size_t>(id);
        if (raw_id < data_num && !blocked.get(raw_id)) {
          blocked.set(raw_id);
          --matched_count;
        }
      };

      for (size_t i = 0; i < range.begin_; ++i) {
        block_id((*entries)[i].id_);
      }
      for (size_t i = range.end_; i < entries->size(); ++i) {
        block_id((*entries)[i].id_);
      }

      return CachedBlockedBitset{
          std::make_shared<DynamicBitset>(std::move(blocked)),
          matched_count,
      };
    }

    blocked.set_all();
    size_t matched_count = 0;
    for (size_t i = range.begin_; i < range.end_; ++i) {
      auto raw_id = static_cast<size_t>((*entries)[i].id_);
      if (raw_id < data_num && blocked.get(raw_id)) {
        blocked.reset(raw_id);
        ++matched_count;
      }
    }

    return CachedBlockedBitset{
        std::make_shared<DynamicBitset>(std::move(blocked)),
        matched_count,
    };
  }

  void invalidate_range_index_cache() const {
    std::lock_guard<std::mutex> lock(range_index_cache_mutex_);
    int_range_indexes_.clear();
    int_range_bitset_cache_.clear();
  }

  static auto parse_data_key(std::string_view key) -> std::optional<IDType> {
    if (!key.starts_with(data_key_prefix())) {
      return std::nullopt;
    }

    auto digits = key.substr(data_key_prefix().size());
    if (digits.empty() || !std::all_of(digits.begin(), digits.end(), [](char ch) {
          return ch >= '0' && ch <= '9';
        })) {
      return std::nullopt;
    }

    auto parsed = std::stoull(std::string(digits));
    if (parsed > static_cast<decltype(parsed)>(std::numeric_limits<UnsignedIDType>::max())) {
      return std::nullopt;
    }
    return static_cast<IDType>(parsed);
  }

  auto get_data_value(IDType id, std::string *value, std::string *resolved_key = nullptr) const
      -> bool {
    auto primary_key = data_key(id);
    rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), primary_key, value);
    if (status.ok()) {
      if (resolved_key != nullptr) {
        *resolved_key = std::move(primary_key);
      }
      return true;
    }

    auto fallback_key = legacy_data_key(id);
    if (fallback_key == primary_key) {
      return false;
    }

    status = db_->Get(rocksdb::ReadOptions(), fallback_key, value);
    if (!status.ok()) {
      return false;
    }
    if (resolved_key != nullptr) {
      *resolved_key = std::move(fallback_key);
    }
    return true;
  }

  [[nodiscard]] auto batch_get_data_values(const std::vector<IDType> &ids) const
      -> std::vector<StoredValue> {
    std::vector<StoredValue> results(ids.size());
    std::vector<std::string> primary_key_strings;
    std::vector<rocksdb::Slice> primary_keys;
    primary_key_strings.reserve(ids.size());
    primary_keys.reserve(ids.size());

    for (auto id : ids) {
      primary_key_strings.push_back(data_key(id));
      primary_keys.emplace_back(primary_key_strings.back());
    }

    std::vector<std::string> primary_values(ids.size());
    auto primary_statuses = db_->MultiGet(rocksdb::ReadOptions(), primary_keys, &primary_values);

    std::vector<size_t> fallback_indexes;
    std::vector<std::string> fallback_key_strings;
    std::vector<rocksdb::Slice> fallback_keys;
    fallback_indexes.reserve(ids.size());
    fallback_key_strings.reserve(ids.size());
    fallback_keys.reserve(ids.size());

    for (size_t i = 0; i < primary_statuses.size(); ++i) {
      if (primary_statuses[i].ok()) {
        results[i].found = true;
        results[i].value = std::move(primary_values[i]);
        results[i].resolved_key = primary_key_strings[i];
        continue;
      }

      auto fallback_key = legacy_data_key(ids[i]);
      if (fallback_key == primary_key_strings[i]) {
        continue;
      }
      fallback_indexes.push_back(i);
      fallback_key_strings.push_back(std::move(fallback_key));
      fallback_keys.emplace_back(fallback_key_strings.back());
    }

    if (fallback_keys.empty()) {
      return results;
    }

    std::vector<std::string> fallback_values(fallback_keys.size());
    auto fallback_statuses = db_->MultiGet(rocksdb::ReadOptions(), fallback_keys, &fallback_values);
    for (size_t i = 0; i < fallback_statuses.size(); ++i) {
      if (!fallback_statuses[i].ok()) {
        continue;
      }
      auto result_index = fallback_indexes[i];
      results[result_index].found = true;
      results[result_index].value = std::move(fallback_values[i]);
      results[result_index].resolved_key = fallback_key_strings[i];
    }
    return results;
  }

  [[nodiscard]] auto batch_find_item_id_owners(const std::vector<std::string> &item_ids) const
      -> std::unordered_map<std::string, IDType> {
    std::unordered_map<std::string, IDType> owners;
    if (item_ids.empty()) {
      return owners;
    }

    std::vector<std::string> key_strings;
    std::vector<rocksdb::Slice> keys;
    key_strings.reserve(item_ids.size());
    keys.reserve(item_ids.size());
    for (const auto &item_id : item_ids) {
      key_strings.push_back(item_id_index_key(item_id));
      keys.emplace_back(key_strings.back());
    }

    std::vector<std::string> values(item_ids.size());
    auto statuses = db_->MultiGet(rocksdb::ReadOptions(), keys, &values);
    for (size_t i = 0; i < statuses.size(); ++i) {
      if (!statuses[i].ok() || values[i].size() != sizeof(IDType)) {
        continue;
      }
      IDType owner_id;
      std::memcpy(&owner_id, values[i].data(), sizeof(IDType));
      owners.emplace(item_ids[i], owner_id);
    }
    return owners;
  }

  void initialize_db() {
    rocksdb::Options options;

    options.create_if_missing = config_.create_if_missing_;
    options.error_if_exists = config_.error_if_exists_;

    options.write_buffer_size = config_.write_buffer_size_;
    options.max_write_buffer_number = config_.max_write_buffer_number_;
    options.target_file_size_base = config_.target_file_size_base_;
    options.max_background_compactions = config_.max_background_compactions_;
    options.max_background_flushes = config_.max_background_flushes_;

    options.compaction_style = rocksdb::kCompactionStyleLevel;
    options.level_compaction_dynamic_level_bytes = true;

    if (config_.enable_compression_) {
      options.compression = rocksdb::kLZ4Compression;
      options.bottommost_compression = rocksdb::kZSTD;
    } else {
      options.compression = rocksdb::kNoCompression;
      options.bottommost_compression = rocksdb::kNoCompression;
    }

    rocksdb::BlockBasedTableOptions table_options;
    table_options.block_cache = rocksdb::NewLRUCache(config_.block_cache_size_mb_ * 1024 * 1024);
    table_options.cache_index_and_filter_blocks = true;
    table_options.cache_index_and_filter_blocks_with_high_priority = true;
    table_options.pin_l0_filter_and_index_blocks_in_cache = true;
    table_options.block_size = static_cast<size_t>(16) * 1024;
    table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));
    options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_options));

    options.max_open_files = -1;
    options.allow_mmap_reads = true;

    // Create parent directories if they don't exist
    auto parent_path = std::filesystem::path(config_.db_path_).parent_path();
    if (!parent_path.empty()) {
      std::filesystem::create_directories(parent_path);
    }

    rocksdb::DB *db = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(options, config_.db_path_, &db);
    if (!status.ok()) {
      std::string err_msg = status.ToString();
      if (err_msg.find("lock file") != std::string::npos ||
          err_msg.find("lock hold") != std::string::npos ||
          err_msg.find("LOCK:") != std::string::npos ||
          err_msg.find("No locks available") != std::string::npos) {
        LOG_INFO("Lock conflict, opening RocksDB in read-only mode at {}", config_.db_path_);
        status = rocksdb::DB::OpenForReadOnly(options, config_.db_path_, &db);
        if (status.ok()) {
          read_only_ = true;
        }
      }
    }
    if (!status.ok()) {
      LOG_ERROR("Failed to open RocksDB at {}: {}", config_.db_path_, status.ToString());
      throw std::runtime_error("Failed to open RocksDB: " + status.ToString());
    }
    db_.reset(db);

    load_count();
    LOG_INFO("RocksDB initialized at {} with {} items{}",
             config_.db_path_,
             cached_count_.load(),
             read_only_ ? " [read-only]" : "");
  }

  void load_count() {
    std::string count_str;
    rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), count_key(), &count_str);
    if (status.ok() && count_str.size() == sizeof(size_t)) {
      size_t count = 0;
      std::memcpy(&count, count_str.data(), sizeof(size_t));
      cached_count_.store(count);
    }
  }

  void save_count() const {
    if (db_ == nullptr || read_only_) {
      return;
    }
    rocksdb::WriteOptions sync_options;
    sync_options.sync = true;

    size_t cnt = cached_count_.load();
    rocksdb::Slice count_slice(reinterpret_cast<const char *>(&cnt), sizeof(size_t));
    db_->Put(sync_options, count_key(), count_slice);
  }

  static auto item_id_index_key(const std::string &item_id) -> std::string {
    return "i_" + item_id;
  }

  static auto count_key() -> std::string { return "__COUNT__"; }

  [[nodiscard]] auto item_id_available_for_locked(const std::string &item_id,
                                                  IDType allowed_id) const -> bool {
    return item_id_available(item_id, std::optional<IDType>{allowed_id});
  }

  void delete_item_id_index(rocksdb::WriteBatch &batch, const std::string &item_id) const {
    if (item_id.empty()) {
      return;
    }
    batch.Delete(item_id_index_key(item_id));
  }

  /**
   * @brief Add field indexes for indexed fields
   */
  void add_field_indexes(rocksdb::WriteBatch &batch, IDType id, const ScalarData &data) const {
    for (const auto &field : config_.indexed_fields_) {
      auto it = data.metadata.find(field);
      if (it != data.metadata.end()) {
        std::string encoded = index_encoding::encode_value(it->second);
        std::string idx_key = index_encoding::make_field_index_key(field, encoded, id);
        batch.Put(idx_key, "");
      }
    }
  }

  /**
   * @brief Remove field indexes for indexed fields
   */
  void remove_field_indexes(rocksdb::WriteBatch &batch, IDType id, const ScalarData &data) const {
    for (const auto &field : config_.indexed_fields_) {
      auto it = data.metadata.find(field);
      if (it != data.metadata.end()) {
        std::string encoded = index_encoding::encode_value(it->second);
        std::string idx_key = index_encoding::make_field_index_key(field, encoded, id);
        batch.Delete(idx_key);
      }
    }
  }

  std::unique_ptr<rocksdb::DB> db_ = nullptr;
  RocksDBConfig config_;
  mutable std::atomic<size_t> cached_count_;
  mutable std::mutex write_mutex_;
  mutable std::mutex range_index_cache_mutex_;
  mutable std::unordered_map<std::string, std::shared_ptr<const IntRangeIndex>> int_range_indexes_;
  mutable std::unordered_map<std::string, CachedBlockedBitset> int_range_bitset_cache_;
  bool read_only_{false};
};

}  // namespace alaya
