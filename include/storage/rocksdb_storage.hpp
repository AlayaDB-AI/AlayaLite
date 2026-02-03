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

#include <rocksdb/cache.h>
#include <rocksdb/db.h>
#include <rocksdb/filter_policy.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/table.h>
#include <rocksdb/table_properties.h>
#include <rocksdb/utilities/checkpoint.h>
#include <rocksdb/write_batch.h>

#include <atomic>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

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

  ~RocksDBStorage() {
    if (db_ != nullptr) {
      save_count();
      rocksdb::Status status = db_->Close();
      if (!status.ok()) {
        LOG_ERROR("Failed to close RocksDB: {}", status.ToString());
      }
      delete db_;
    }
  }

  RocksDBStorage(const RocksDBStorage &) = delete;
  auto operator=(const RocksDBStorage &) -> RocksDBStorage & = delete;

  RocksDBStorage(RocksDBStorage &&other) noexcept
      : db_(std::exchange(other.db_, nullptr)),
        config_(std::move(other.config_)),
        cached_count_(other.cached_count_.load()) {}

  auto operator=(RocksDBStorage &&other) noexcept -> RocksDBStorage & {
    if (this != &other) {
      if (db_ != nullptr) {
        save_count();
        db_->Close();
        delete db_;
      }
      db_ = std::exchange(other.db_, nullptr);
      config_ = std::move(other.config_);
      cached_count_ = other.cached_count_.load();
    }
    return *this;
  }

  /**
   * @brief Get ScalarData by internal ID
   * @param id Internal ID
   * @return ScalarData (empty if not found)
   */
  [[nodiscard]] auto operator[](IDType id) const -> ScalarData {
    std::string key = data_key(id);
    std::string value;
    rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), key, &value);

    if (!status.ok()) {
      LOG_ERROR("Failed to access ScalarData for ID {}: {}", id, status.ToString());
      return ScalarData{};
    }

    return ScalarData::deserialize(value.data(), value.size());
  }

  /**
   * @brief Check if an ID exists
   */
  [[nodiscard]] auto is_valid(IDType id) const -> bool {
    std::string key = data_key(id);
    std::string value;
    rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), key, &value);
    return status.ok();
  }

  /**
   * @brief Insert ScalarData with specified ID (managed by Space)
   * @param id Internal ID
   * @param data ScalarData to insert
   * @return true on success
   */
  auto insert(IDType id, const ScalarData &data) -> bool {
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
    // First check if the ID exists
    if (!is_valid(id)) {
      LOG_ERROR("Failed to remove ID({}) that doesn't exist.", id);
      return false;
    }

    auto data = (*this)[id];

    rocksdb::WriteBatch batch;
    batch.Delete(data_key(id));

    if (!data.item_id.empty()) {
      batch.Delete(item_id_index_key(data.item_id));
    }

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
    auto old_data = (*this)[id];

    rocksdb::WriteBatch batch;

    // Update primary data
    auto serialized = data.serialize();
    batch.Put(data_key(id), rocksdb::Slice(serialized.data(), serialized.size()));

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

    for (size_t i = 0; i < ids.size(); ++i) {
      if (statuses[i].ok()) {
        results.push_back(ScalarData::deserialize(values[i].data(), values[i].size()));
      } else {
        results.emplace_back();
      }
    }

    return results;
  }

  [[nodiscard]] auto count() const -> size_t { return cached_count_.load(); }

  /**
   * @brief Scan all ScalarData with a filter function
   * @param filter_fn Filter function, return true to include the record
   * @param limit Maximum number of results (0 = no limit)
   * @return Vector of (internal_id, ScalarData) pairs
   */
  [[nodiscard]] auto scan_with_filter(std::function<bool(const ScalarData &)> filter_fn,
                                      size_t limit = 0) const
      -> std::vector<std::pair<IDType, ScalarData>> {
    std::vector<std::pair<IDType, ScalarData>> results;

    rocksdb::ReadOptions read_opts;
    read_opts.fill_cache = false;

    std::unique_ptr<rocksdb::Iterator> iter(db_->NewIterator(read_opts));

    for (iter->Seek("d_"); iter->Valid(); iter->Next()) {
      auto key = iter->key().ToString();
      if (key.size() < 2 || key[0] != 'd' || key[1] != '_') {
        break;
      }

      IDType id = static_cast<IDType>(std::stoul(key.substr(2)));
      auto value = iter->value();
      ScalarData sd = ScalarData::deserialize(value.data(), value.size());

      if (filter_fn(sd)) {
        results.emplace_back(id, std::move(sd));
        if (limit > 0 && results.size() >= limit) {
          break;
        }
      }
    }

    return results;
  }

  [[nodiscard]] auto config() const -> const RocksDBConfig & { return config_; }

  [[nodiscard]] auto get_db_path() const -> const std::string & { return config_.db_path_; }

  void flush() const {
    if (db_ != nullptr) {
      save_count();
      db_->Flush(rocksdb::FlushOptions());
    }
  }

  void compact() {
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
    flush();

    rocksdb::Checkpoint *checkpoint;
    rocksdb::Status status = rocksdb::Checkpoint::Create(db_, &checkpoint);
    if (!status.ok()) {
      LOG_ERROR("Failed to create checkpoint: {}", status.ToString());
      return;
    }

    status = checkpoint->CreateCheckpoint(filepath);
    delete checkpoint;

    if (!status.ok()) {
      LOG_ERROR("Failed to save checkpoint to {}: {}", filepath, status.ToString());
    }
  }

 private:
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

    rocksdb::Status status = rocksdb::DB::Open(options, config_.db_path_, &db_);
    if (!status.ok()) {
      LOG_ERROR("Failed to open RocksDB at {}: {}", config_.db_path_, status.ToString());
      throw std::runtime_error("Failed to open RocksDB: " + status.ToString());
    }

    load_count();
    LOG_INFO("RocksDB initialized at {} with {} items", config_.db_path_, cached_count_.load());
  }

  void load_count() {
    std::string count_str;
    rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), count_key(), &count_str);
    if (status.ok() && count_str.size() == sizeof(size_t)) {
      cached_count_ = *reinterpret_cast<const size_t *>(count_str.data());
    }
  }

  void save_count() const {
    rocksdb::WriteOptions sync_options;
    sync_options.sync = true;

    size_t cnt = cached_count_.load();
    rocksdb::Slice count_slice(reinterpret_cast<const char *>(&cnt), sizeof(size_t));
    db_->Put(sync_options, count_key(), count_slice);
  }

  static auto data_key(IDType id) -> std::string { return "d_" + std::to_string(id); }

  static auto item_id_index_key(const std::string &item_id) -> std::string {
    return "i_" + item_id;
  }

  static auto count_key() -> std::string { return "__COUNT__"; }

  rocksdb::DB *db_ = nullptr;
  RocksDBConfig config_;
  mutable std::atomic<size_t> cached_count_;
};

}  // namespace alaya
