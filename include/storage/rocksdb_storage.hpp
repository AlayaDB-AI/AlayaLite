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
#include <spdlog/spdlog.h>
#include <atomic>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "utils/log.hpp"

namespace alaya {

struct RocksDBConfig {
  std::string db_path_ = "./RocksDB/alayalite_rocksdb";
  size_t write_buffer_size_ = static_cast<size_t>(64) << 20;  // 64MB
  int max_write_buffer_number_ = 4;
  size_t target_file_size_base_ = static_cast<size_t>(64) << 20;  // 64MB
  int max_background_compactions_ = 4;
  int max_background_flushes_ = 2;
  bool create_if_missing_ = true;
  bool error_if_exists_ = false;
  size_t block_cache_size_mb_ = 512;  // 512MB
  bool enable_compression_ = true;    // Enable LZ4+ZSTD compression by default

  static auto default_config() -> RocksDBConfig { return RocksDBConfig{}; }
};

template <typename DataType, typename IDType = uint32_t>
class RocksDBStorage {
  static_assert(std::is_trivially_copyable_v<DataType>,
                "DataType must be trivially copyable for RocksDB storage");

 public:
  explicit RocksDBStorage(RocksDBConfig config = RocksDBConfig::default_config())
      : config_(std::move(config)), next_id_(0), cached_count_(0) {
    initialize_db();
  }

  ~RocksDBStorage() {
    if (db_ != nullptr) {
      // Save metadata variables before closing
      save_variables();
      // Close() will automatically flush and wait for all background operations
      rocksdb::Status status = db_->Close();
      if (!status.ok()) {
        LOG_ERROR("Failed to close RocksDB: {}", status.ToString());
      }
      delete db_;
    }
  }

  // Disable copy operations
  RocksDBStorage(const RocksDBStorage &) = delete;
  auto operator=(const RocksDBStorage &) -> RocksDBStorage & = delete;

  // Move operations
  RocksDBStorage(RocksDBStorage &&other) noexcept
      : db_(std::exchange(other.db_, nullptr)),
        config_(std::move(other.config_)),
        next_id_(other.next_id_.load()),
        cached_count_(other.cached_count_.load()) {}

  auto operator=(RocksDBStorage &&other) noexcept -> RocksDBStorage & {
    if (this != &other) {
      RocksDBStorage temp(std::move(*this));
      db_ = std::exchange(other.db_, nullptr);
      config_ = std::move(other.config_);
      next_id_ = other.next_id_.load();
      cached_count_ = other.cached_count_.load();
    }
    return *this;
  }

  // Storage Concept Implementation

  [[nodiscard]] auto operator[](IDType id) const -> DataType {
    if (!is_valid(id)) {
      LOG_ERROR("Invalid ID: {}", id);
      return DataType{};
    }

    std::string key = id_to_key(id);
    std::string value;
    rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), key, &value);

    if (!status.ok()) {
      LOG_ERROR("Failed to get data for ID {}: {}", id, status.ToString());
      return DataType{};
    }

    if (value.size() != sizeof(DataType)) {
      throw std::runtime_error("Data corruption detected for ID " + std::to_string(id) +
                               ": expected size " + std::to_string(sizeof(DataType)) + ", got " +
                               std::to_string(value.size()));
    }

    DataType result{};
    std::memcpy(&result, value.data(), sizeof(DataType));
    return result;
  }

  [[nodiscard]] auto is_valid(IDType id) const -> bool {
    if (id >= next_id_.load()) {
      return false;
    }

    std::string key = id_to_key(id);
    std::string value;
    const rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), key, &value);  // NOLINT

    if (!status.ok() && !status.IsNotFound()) {
      LOG_ERROR("Failed to check ID {}: {}", id, status.ToString());
    }

    return status.ok() && !value.empty();
  }

  auto insert(const DataType &data) -> IDType {
    IDType id = next_id_++;
    std::string key = id_to_key(id);
    rocksdb::Slice value_slice(reinterpret_cast<const char *>(&data), sizeof(DataType));

    rocksdb::Status status = db_->Put(rocksdb::WriteOptions(), key, value_slice);
    if (!status.ok()) {
      LOG_ERROR("Failed to insert data for ID {}: {}", id, status.ToString());
      return static_cast<IDType>(-1);
    }

    ++cached_count_;
    return id;
  }

  [[nodiscard]] auto remove(IDType id) -> IDType {
    // todo: recycle empty space
    if (!is_valid(id)) {
      LOG_ERROR("Cannot remove invalid ID: {}", id);
      return static_cast<IDType>(-1);
    }

    std::string key = id_to_key(id);
    rocksdb::Status status = db_->Delete(rocksdb::WriteOptions(), key);

    if (!status.ok()) {
      LOG_ERROR("Failed to remove data for ID {}: {}", id, status.ToString());
      return static_cast<IDType>(-1);
    }

    --cached_count_;
    return id;
  }

  [[nodiscard]] auto update(IDType id, const DataType &data) -> IDType {
    if (!is_valid(id)) {
      LOG_ERROR("Cannot update invalid ID: {}", id);
      return static_cast<IDType>(-1);
    }

    std::string key = id_to_key(id);
    rocksdb::Slice value_slice(reinterpret_cast<const char *>(&data), sizeof(DataType));

    rocksdb::Status status = db_->Put(rocksdb::WriteOptions(), key, value_slice);
    if (!status.ok()) {
      LOG_ERROR("Failed to update data for ID {}: {}", id, status.ToString());
      return static_cast<IDType>(-1);
    }

    return id;
  }

  // Batch operations for better performance
  template <typename Iterator>
  auto batch_insert(Iterator begin, Iterator end) -> bool {
    rocksdb::WriteBatch batch;
    size_t count = std::distance(begin, end);

    for (auto it = begin; it != end; ++it) {
      IDType id = next_id_++;
      std::string key = id_to_key(id);
      rocksdb::Slice value_slice(reinterpret_cast<const char *>(&(*it)), sizeof(DataType));
      batch.Put(key, value_slice);
    }

    rocksdb::Status status = db_->Write(rocksdb::WriteOptions(), &batch);
    if (!status.ok()) {
      LOG_ERROR("Batch insert failed: {}", status.ToString());
      return false;
    }

    cached_count_ += count;
    return true;
  }

  // Maintenance operations
  [[nodiscard]] auto count() const -> size_t { return cached_count_.load(); }

  [[nodiscard]] auto next_id() const -> IDType { return next_id_.load(); }

  auto config() const -> const RocksDBConfig & { return config_; }

  void flush() const {
    if (db_ != nullptr) {
      save_variables();
      db_->Flush(rocksdb::FlushOptions());
    } else {
      LOG_WARN("db is nullptr.");
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

  // Save to file (compatibility with existing storage pattern)
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

  auto get_db_path() const -> const std::string & { return config_.db_path_; }

 private:
  void initialize_db() {
    rocksdb::Options options;

    // Configure basic options
    options.create_if_missing = config_.create_if_missing_;
    options.error_if_exists = config_.error_if_exists_;

    // Optimize for performance
    options.write_buffer_size = config_.write_buffer_size_;
    options.max_write_buffer_number = config_.max_write_buffer_number_;
    options.target_file_size_base = config_.target_file_size_base_;
    options.max_background_compactions = config_.max_background_compactions_;
    options.max_background_flushes = config_.max_background_flushes_;

    // Use level compaction for better read performance
    options.compaction_style = rocksdb::kCompactionStyleLevel;
    options.level_compaction_dynamic_level_bytes = true;

    // Configure compression
    if (config_.enable_compression_) {
      // Use LZ4 for upper levels (best performance for frequently accessed data)
      options.compression = rocksdb::kLZ4Compression;
      // Use ZSTD for bottommost level (best compression ratio for cold data)
      options.bottommost_compression = rocksdb::kZSTD;
    } else {
      // Disable compression
      options.compression = rocksdb::kNoCompression;
      options.bottommost_compression = rocksdb::kNoCompression;
    }

    // Configure block-based table
    rocksdb::BlockBasedTableOptions table_options;
    table_options.block_cache = rocksdb::NewLRUCache(config_.block_cache_size_mb_ * 1024 * 1024);
    table_options.cache_index_and_filter_blocks = true;
    table_options.cache_index_and_filter_blocks_with_high_priority = true;
    table_options.pin_l0_filter_and_index_blocks_in_cache = true;
    table_options.block_size = static_cast<size_t>(16) * 1024;  // 16KB blocks

    // Add bloom filter
    table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10, false));

    options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_options));

    // Optimize for concurrent workloads
    options.max_open_files = -1;      // Keep all files open
    options.allow_mmap_reads = true;  // Use mmap for reads

    // Open the database
    rocksdb::Status status = rocksdb::DB::Open(options, config_.db_path_, &db_);
    if (!status.ok()) {
      LOG_ERROR("Failed to open RocksDB at {}: {}", config_.db_path_, status.ToString());
      throw std::runtime_error("Failed to open RocksDB: " + status.ToString());
    }

    load_variables();

    LOG_INFO("RocksDB initialized at {} with {} items", config_.db_path_, cached_count_.load());
  }

  void load_variables() {
    // Load next_id
    std::string next_id_str;
    rocksdb::Status status = db_->Get(rocksdb::ReadOptions(), "__NEXT_ID__", &next_id_str);
    if (status.ok() && !next_id_str.empty() && next_id_str.size() == sizeof(IDType)) {
      next_id_ = *reinterpret_cast<const IDType *>(next_id_str.data());
    } else {
      LOG_WARN("Failed to load next_id: {}", status.ToString());
    }

    // Load count
    std::string count_str;
    status = db_->Get(rocksdb::ReadOptions(), "__COUNT__", &count_str);
    if (status.ok() && !count_str.empty() && count_str.size() == sizeof(size_t)) {
      cached_count_ = *reinterpret_cast<const size_t *>(count_str.data());
    } else {
      LOG_WARN("Failed to load count: {}", status.ToString());
    }
  }

  void save_variables() const {
    rocksdb::WriteOptions sync_options;
    sync_options.sync = true;  // Ensure metadata is persisted to disk

    // Save next_id
    rocksdb::Slice next_id_slice(reinterpret_cast<const char *>(&next_id_), sizeof(IDType));
    const rocksdb::Status status =  // NOLINT
        db_->Put(sync_options, "__NEXT_ID__", next_id_slice);
    if (!status.ok()) {
      LOG_ERROR("Failed to save next_id: {}", status.ToString());
    }

    // Save count
    size_t count = cached_count_.load();
    rocksdb::Slice count_slice(reinterpret_cast<const char *>(&count), sizeof(size_t));
    const rocksdb::Status status2 =  // NOLINT
        db_->Put(sync_options, "__COUNT__", count_slice);
    if (!status2.ok()) {
      LOG_ERROR("Failed to save count: {}", status2.ToString());
    }
  }

  static auto id_to_key(IDType id) -> std::string { return "data_" + std::to_string(id); }

  rocksdb::DB *db_;
  RocksDBConfig config_;
  mutable std::atomic<IDType> next_id_;
  mutable std::atomic<size_t> cached_count_;
};

}  // namespace alaya
