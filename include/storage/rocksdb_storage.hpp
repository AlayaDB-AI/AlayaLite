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
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

#include "utils/index_encoding.hpp"
#include "utils/log.hpp"
#include "utils/scalar_data.hpp"

namespace alaya {

namespace detail {
inline void create_directories_recursive(const std::string &path) {
  if (path.empty()) {
    return;
  }
  std::string current;
  for (size_t i = 0; i < path.size(); ++i) {
    current += path[i];
    if (path[i] == '/' && i > 0) {
      mkdir(current.c_str(), 0755);  // NOLINT
    }
  }
  if (!current.empty() && current.back() != '/') {
    mkdir(current.c_str(), 0755);  // NOLINT
  }
}

inline auto get_parent_path(const std::string &path) -> std::string {
  size_t pos = path.rfind('/');
  if (pos == std::string::npos || pos == 0) {
    return "";
  }
  return path.substr(0, pos);
}
}  // namespace detail

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
      other.read_only_ = false;
    }
    return *this;
  }

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
    std::vector<rocksdb::Slice> keys;
    std::vector<std::string> key_strings;
    keys.reserve(ids.size());
    key_strings.reserve(ids.size());

    for (auto id : ids) {
      key_strings.push_back(data_key(id));
      keys.emplace_back(key_strings.back());
    }

    std::vector<std::string> values(ids.size());
    std::vector<rocksdb::Status> statuses = db_->MultiGet(rocksdb::ReadOptions(), keys, &values);

    for (size_t i = 0; i < statuses.size(); ++i) {
      if (!statuses[i].ok()) {
        values[i].clear();
        auto legacy_key = legacy_data_key(ids[i]);
        if (legacy_key != key_strings[i]) {
          rocksdb::Status fallback = db_->Get(rocksdb::ReadOptions(), legacy_key, &values[i]);
          if (!fallback.ok()) {
            values[i].clear();
          }
        }
      }
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
    ensure_writable("insert");
    std::string key = data_key(id);
    auto serialized = data.serialize();
    rocksdb::Slice value_slice(serialized.data(), serialized.size());

    rocksdb::WriteBatch batch;
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

    ++cached_count_;
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
    ensure_writable("batch_insert");
    rocksdb::WriteBatch batch;
    IDType current_id = start_id;
    size_t count = 0;

    for (auto it = begin; it != end; ++it, ++current_id) {
      std::string key = data_key(current_id);
      auto serialized = it->serialize();
      batch.Put(key, rocksdb::Slice(serialized.data(), serialized.size()));

      // Add secondary index
      if (!it->item_id.empty()) {
        std::string idx_key = item_id_index_key(it->item_id);
        rocksdb::Slice id_slice(reinterpret_cast<const char *>(&current_id), sizeof(IDType));
        batch.Put(idx_key, id_slice);
      }

      // Add field indexes for indexed fields
      add_field_indexes(batch, current_id, *it);

      ++count;
    }

    rocksdb::Status status = db_->Write(rocksdb::WriteOptions(), &batch);
    if (!status.ok()) {
      LOG_ERROR("Batch insert failed: {}", status.ToString());
      return false;
    }

    cached_count_ += count;
    return true;
  }

  /**
   * @brief Remove ScalarData by ID
   */
  auto remove(IDType id) -> bool {
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

    if (!data.item_id.empty()) {
      batch.Delete(item_id_index_key(data.item_id));
    }

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
    return true;
  }

  /**
   * @brief Update ScalarData
   */
  auto update(IDType id, const ScalarData &data) -> bool {
    ensure_writable("update");

    std::string serialized_value;
    std::string resolved_key;
    if (!get_data_value(id, &serialized_value, &resolved_key)) {
      LOG_ERROR("Failed to update ID({}) that doesn't exist.", id);
      return false;
    }
    auto old_data = ScalarData::deserialize(serialized_value.data(), serialized_value.size());

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
      if (!old_data.item_id.empty()) {
        batch.Delete(item_id_index_key(old_data.item_id));
      }
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
    std::string encoded = index_encoding::encode_value(value);
    std::string prefix = index_encoding::make_field_index_prefix(field, encoded);

    rocksdb::ReadOptions read_opts;
    read_opts.fill_cache = false;
    std::unique_ptr<rocksdb::Iterator> iter(db_->NewIterator(read_opts));

    for (iter->Seek(prefix); iter->Valid(); iter->Next()) {
      auto key = iter->key().ToString();
      if (!key.starts_with(prefix)) {
        break;
      }
      IDType id = index_encoding::extract_id_from_key<IDType>(key);
      ids.push_back(id);
    }

    return ids;
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
    std::vector<IDType> ids;

    std::string field_prefix = index_encoding::make_field_prefix(field);
    std::string start_key = field_prefix + "i_" + index_encoding::encode_int64(min_value);
    std::string end_encoded = index_encoding::encode_int64(max_value);

    rocksdb::ReadOptions read_opts;
    read_opts.fill_cache = false;
    std::unique_ptr<rocksdb::Iterator> iter(db_->NewIterator(read_opts));

    for (iter->Seek(start_key); iter->Valid(); iter->Next()) {
      auto key = iter->key().ToString();
      // Check if still in the same field
      if (!key.starts_with(field_prefix)) {
        break;
      }
      // Check if it's an int type (prefix "i_")
      if (key.size() < field_prefix.size() + 2 || key.substr(field_prefix.size(), 2) != "i_") {
        continue;
      }
      // Extract encoded value and check range
      auto value_start = field_prefix.size() + 2;
      auto value_end = key.rfind('_');
      if (value_end == std::string::npos || value_end <= value_start) {
        continue;
      }
      std::string encoded_value = key.substr(value_start, value_end - value_start);
      if (encoded_value > end_encoded) {
        break;  // Exceeded max value
      }
      IDType id = index_encoding::extract_id_from_key<IDType>(key);
      ids.push_back(id);
    }

    return ids;
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

    std::string field_prefix = index_encoding::make_field_prefix(field);
    std::string start_key = field_prefix + "d_" + index_encoding::encode_double(min_value);
    std::string end_encoded = index_encoding::encode_double(max_value);

    rocksdb::ReadOptions read_opts;
    read_opts.fill_cache = false;
    std::unique_ptr<rocksdb::Iterator> iter(db_->NewIterator(read_opts));

    for (iter->Seek(start_key); iter->Valid(); iter->Next()) {
      auto key = iter->key().ToString();
      if (!key.starts_with(field_prefix)) {
        break;
      }
      if (key.size() < field_prefix.size() + 2 || key.substr(field_prefix.size(), 2) != "d_") {
        continue;
      }
      auto value_start = field_prefix.size() + 2;
      auto value_end = key.rfind('_');
      if (value_end == std::string::npos || value_end <= value_start) {
        continue;
      }
      std::string encoded_value = key.substr(value_start, value_end - value_start);
      if (encoded_value > end_encoded) {
        break;
      }
      IDType id = index_encoding::extract_id_from_key<IDType>(key);
      ids.push_back(id);
    }

    return ids;
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
      LOG_ERROR("Failed to create checkpoint: {}", status.ToString());
      return;
    }

    status = checkpoint->CreateCheckpoint(filepath);

    if (!status.ok()) {
      LOG_ERROR("Failed to save checkpoint to {}: {}", filepath, status.ToString());
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
    std::string parent_path = detail::get_parent_path(config_.db_path_);
    if (!parent_path.empty()) {
      detail::create_directories_recursive(parent_path);
      // Ignore error if directory already exists
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
  bool read_only_{false};
};

}  // namespace alaya
