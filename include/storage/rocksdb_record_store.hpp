// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/slice.h>
#include <rocksdb/utilities/checkpoint.h>
#include <rocksdb/write_batch.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
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

#include "scalar/scalar_index_snapshot.hpp"
#include "storage/record_store.hpp"
#include "utils/index_encoding.hpp"
#include "utils/scalar_data.hpp"

namespace alaya {

/** @brief Stable v2 column-family names shared by storage, migration and recovery code. */
struct RocksDBRecordStoreSchema {
  static constexpr uint32_t kVersion = 2;  ///< On-disk schema version stored in metadata.
  static constexpr std::string_view kRecords = "records";  ///< ID -> serialized ScalarData.
  static constexpr std::string_view kVectors = "vectors";  ///< ID -> exact/raw vector bytes.
  static constexpr std::string_view kQuantizedVectors =
      "quantized_vectors";  ///< ID -> optional quantized vector bytes used by ANN search.
  static constexpr std::string_view kItemIds = "item_ids";  ///< External item ID -> internal ID.
  static constexpr std::string_view kScalarIndexes =
      "scalar_indexes";  ///< Durable field/value/ID mirror; memory currently rebuilds from records.
  static constexpr std::string_view kMetadata =
      "metadata";  ///< Schema version, generation, live count and ID universe.

  /** @brief Return every CF required to open a v2 database, including RocksDB's default CF. */
  [[nodiscard]] static auto column_families() -> std::vector<std::string> {
    return {rocksdb::kDefaultColumnFamilyName,
            std::string(kRecords),
            std::string(kVectors),
            std::string(kQuantizedVectors),
            std::string(kItemIds),
            std::string(kScalarIndexes),
            std::string(kMetadata)};
  }
};

/** @brief Configuration for the v2 canonical record store. */
struct RocksDBRecordStoreConfig {
  std::string db_path_;  ///< Dedicated v2 directory; legacy single-CF directories are rejected.
  std::vector<std::string> indexed_fields_;  ///< Metadata fields copied into the memory snapshot.
  bool create_if_missing_ = true;  ///< Create a fresh v2 directory when CURRENT is absent.
  bool sync_writes_ = false;  ///< Fsync each atomic mutation in addition to RocksDB WAL ordering.
};

/**
 * @brief Canonical v2 RocksDB store for scalar rows, raw vectors and quantized vectors.
 *
 * Every logical row mutation is one cross-CF WriteBatch. The batch also advances generation, so a
 * recovered database cannot expose scalar/vector/item-ID components from different mutations. An
 * immutable ScalarIndexSnapshot is prepared before commit and atomically published after the batch
 * succeeds. This first implementation rebuilds that snapshot from canonical rows on each mutation;
 * Phase 3 can replace the O(N) preparation with copy-on-write field deltas without changing the
 * query contract.
 */
template <typename IDType = uint32_t>
class RocksDBRecordStore final : public RecordStore<IDType> {
  static_assert(std::is_integral_v<IDType> && std::is_unsigned_v<IDType>,
                "RocksDBRecordStore IDs must be unsigned integral values");

 public:
  using ScalarSnapshot = ScalarIndexSnapshot<IDType>;
  using SnapshotRecord = typename ScalarSnapshot::Record;

  /**
   * @brief Generation-stable record view paired with one immutable scalar snapshot.
   *
   * The parent store must outlive this view. Reads use a RocksDB snapshot captured under the same
   * publication mutex as scalar_snapshot_, so residual evaluation cannot observe a newer row.
   */
  class QueryView final : public RecordStore<IDType> {
   public:
    ~QueryView() override {
      if (rocks_snapshot_ != nullptr) {
        owner_->db_->ReleaseSnapshot(rocks_snapshot_);
      }
    }

    QueryView(const QueryView &) = delete;
    auto operator=(const QueryView &) -> QueryView & = delete;
    QueryView(QueryView &&) = delete;
    auto operator=(QueryView &&) -> QueryView & = delete;

    /** @copydoc RecordStore::get_raw_scalar */
    auto get_raw_scalar(IDType id, std::string &value) const -> bool override {
      return owner_->get_cf_value(owner_->records_cf_, encode_id(id), value, read_options());
    }

    /** @copydoc RecordStore::batch_get_raw_scalars */
    [[nodiscard]] auto batch_get_raw_scalars(const std::vector<IDType> &ids) const
        -> std::vector<std::string> override {
      std::vector<std::string> result(ids.size());
      auto options = read_options();
      for (size_t i = 0; i < ids.size(); ++i) {
        (void)owner_->get_cf_value(owner_->records_cf_, encode_id(ids[i]), result[i], options);
      }
      return result;
    }

    /** @copydoc RecordStore::find_by_item_id */
    [[nodiscard]] auto find_by_item_id(const std::string &item_id) const
        -> std::optional<IDType> override {
      return owner_->find_by_item_id_with_options(item_id, read_options());
    }

    /** @copydoc RecordStore::size */
    [[nodiscard]] auto size() const -> size_t override { return scalar_snapshot_->live_count(); }

    /** @copydoc RecordStore::generation */
    [[nodiscard]] auto generation() const -> uint64_t override {
      return scalar_snapshot_->generation();
    }

    /** @brief Return the scalar postings paired with this RocksDB read snapshot. */
    [[nodiscard]] auto scalar_index() const -> const ScalarSnapshot & { return *scalar_snapshot_; }

    /** @brief Read exact/raw vector bytes from this generation. */
    auto get_raw_vector(IDType id, std::string &value) const -> bool {
      return owner_->get_cf_value(owner_->vectors_cf_, encode_id(id), value, read_options());
    }

    /** @brief Read optional quantized vector bytes from this generation. */
    auto get_quantized_vector(IDType id, std::string &value) const -> bool {
      return owner_->get_cf_value(owner_->quantized_vectors_cf_,
                                  encode_id(id),
                                  value,
                                  read_options());
    }

   private:
    friend class RocksDBRecordStore;

    QueryView(const RocksDBRecordStore *owner,
              const rocksdb::Snapshot *rocks_snapshot,
              std::shared_ptr<const ScalarSnapshot> scalar_snapshot)
        : owner_(owner),
          rocks_snapshot_(rocks_snapshot),
          scalar_snapshot_(std::move(scalar_snapshot)) {}

    /** @brief Build read options pinned to this view's RocksDB sequence number. */
    [[nodiscard]] auto read_options() const -> rocksdb::ReadOptions {
      rocksdb::ReadOptions options;
      options.snapshot = rocks_snapshot_;
      return options;
    }

    const RocksDBRecordStore *owner_;  ///< Non-owning store that must outlive this query view.
    const rocksdb::Snapshot *rocks_snapshot_;  ///< RocksDB sequence paired with scalar_snapshot_.
    std::shared_ptr<const ScalarSnapshot> scalar_snapshot_;  ///< Immutable in-memory postings.
  };

  /** @brief Open or create a dedicated v2 record-store directory. */
  explicit RocksDBRecordStore(RocksDBRecordStoreConfig config) : config_(std::move(config)) {
    if (config_.db_path_.empty()) {
      throw std::invalid_argument("RocksDBRecordStore path cannot be empty");
    }
    std::sort(config_.indexed_fields_.begin(), config_.indexed_fields_.end());
    config_.indexed_fields_.erase(std::unique(config_.indexed_fields_.begin(),
                                              config_.indexed_fields_.end()),
                                  config_.indexed_fields_.end());
    open();
  }

  ~RocksDBRecordStore() override { close(); }

  RocksDBRecordStore(const RocksDBRecordStore &) = delete;
  auto operator=(const RocksDBRecordStore &) -> RocksDBRecordStore & = delete;
  RocksDBRecordStore(RocksDBRecordStore &&) = delete;
  auto operator=(RocksDBRecordStore &&) -> RocksDBRecordStore & = delete;

  /**
   * @brief Atomically insert or replace one complete scalar/vector row.
   * @param id Stable internal ID shared by scalar and vector indexes.
   * @param scalar Canonical external ID, document and metadata payload.
   * @param raw_vector Exact vector bytes required for reranking and brute force.
   * @param quantized_vector Optional ANN representation; absence deletes an older representation.
   * @return false for duplicate item IDs or a failed RocksDB commit.
   */
  auto upsert(IDType id,
              const ScalarData &scalar,
              std::string_view raw_vector,
              std::optional<std::string_view> quantized_vector = std::nullopt) -> bool {
    if (raw_vector.empty()) {
      throw std::invalid_argument("Raw vector bytes cannot be empty");
    }

    std::lock_guard<std::mutex> lock(publication_mutex_);
    auto existing_owner = find_by_item_id_with_options(scalar.item_id, rocksdb::ReadOptions{});
    if (!scalar.item_id.empty() && existing_owner.has_value() && *existing_owner != id) {
      return false;
    }

    std::string old_raw;
    std::optional<ScalarData> old_scalar;
    if (get_cf_value(records_cf_, encode_id(id), old_raw, rocksdb::ReadOptions{})) {
      old_scalar = ScalarData::deserialize(old_raw.data(), old_raw.size());
    }

    auto next_generation = checked_next_generation();
    auto next_count =
        live_count_.load(std::memory_order_relaxed) + static_cast<size_t>(!old_scalar.has_value());
    auto raw_id = static_cast<size_t>(id);
    if (raw_id == std::numeric_limits<size_t>::max()) {
      throw std::overflow_error("Internal ID cannot define an exclusive universe bound");
    }
    auto next_universe = std::max(universe_size_.load(std::memory_order_relaxed), raw_id + 1);
    auto records = load_all_records();
    replace_snapshot_record(records, id, scalar);
    auto next_snapshot =
        ScalarSnapshot::build(next_generation, next_universe, config_.indexed_fields_, records);

    rocksdb::WriteBatch batch;
    auto key = encode_id(id);
    if (old_scalar.has_value()) {
      delete_item_id_entry(batch, old_scalar->item_id);
      delete_scalar_index_entries(batch, id, *old_scalar);
    }

    auto serialized = scalar.serialize();
    batch.Put(records_cf_, key, rocksdb::Slice(serialized.data(), serialized.size()));
    batch.Put(vectors_cf_, key, rocksdb::Slice(raw_vector.data(), raw_vector.size()));
    if (quantized_vector.has_value()) {
      batch.Put(quantized_vectors_cf_,
                key,
                rocksdb::Slice(quantized_vector->data(), quantized_vector->size()));
    } else {
      batch.Delete(quantized_vectors_cf_, key);
    }
    put_item_id_entry(batch, scalar.item_id, id);
    put_scalar_index_entries(batch, id, scalar);
    put_metadata(batch, next_generation, next_count, next_universe);

    if (!write(batch)) {
      return false;
    }
    publish(std::move(next_snapshot), next_count, next_universe);
    return true;
  }

  /** @brief Atomically delete every persisted component for one internal ID. */
  auto remove(IDType id) -> bool {
    std::lock_guard<std::mutex> lock(publication_mutex_);
    std::string old_raw;
    auto key = encode_id(id);
    if (!get_cf_value(records_cf_, key, old_raw, rocksdb::ReadOptions{})) {
      return false;
    }
    auto old_scalar = ScalarData::deserialize(old_raw.data(), old_raw.size());
    auto next_generation = checked_next_generation();
    auto next_count = live_count_.load(std::memory_order_relaxed) - 1;
    auto next_universe = universe_size_.load(std::memory_order_relaxed);
    auto records = load_all_records();
    erase_snapshot_record(records, id);
    auto next_snapshot =
        ScalarSnapshot::build(next_generation, next_universe, config_.indexed_fields_, records);

    rocksdb::WriteBatch batch;
    batch.Delete(records_cf_, key);
    batch.Delete(vectors_cf_, key);
    batch.Delete(quantized_vectors_cf_, key);
    delete_item_id_entry(batch, old_scalar.item_id);
    delete_scalar_index_entries(batch, id, old_scalar);
    put_metadata(batch, next_generation, next_count, next_universe);

    if (!write(batch)) {
      return false;
    }
    publish(std::move(next_snapshot), next_count, next_universe);
    return true;
  }

  /** @copydoc RecordStore::get_raw_scalar */
  auto get_raw_scalar(IDType id, std::string &value) const -> bool override {
    std::lock_guard<std::mutex> lock(publication_mutex_);
    return get_cf_value(records_cf_, encode_id(id), value, rocksdb::ReadOptions{});
  }

  /** @copydoc RecordStore::batch_get_raw_scalars */
  [[nodiscard]] auto batch_get_raw_scalars(const std::vector<IDType> &ids) const
      -> std::vector<std::string> override {
    std::lock_guard<std::mutex> lock(publication_mutex_);
    std::vector<std::string> result(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
      (void)get_cf_value(records_cf_, encode_id(ids[i]), result[i], rocksdb::ReadOptions{});
    }
    return result;
  }

  /** @copydoc RecordStore::find_by_item_id */
  [[nodiscard]] auto find_by_item_id(const std::string &item_id) const
      -> std::optional<IDType> override {
    std::lock_guard<std::mutex> lock(publication_mutex_);
    return find_by_item_id_with_options(item_id, rocksdb::ReadOptions{});
  }

  /** @copydoc RecordStore::size */
  [[nodiscard]] auto size() const -> size_t override {
    auto snapshot = current_scalar_snapshot();
    return snapshot == nullptr ? 0 : snapshot->live_count();
  }

  /** @copydoc RecordStore::generation */
  [[nodiscard]] auto generation() const -> uint64_t override {
    auto snapshot = current_scalar_snapshot();
    return snapshot == nullptr ? 0 : snapshot->generation();
  }

  /** @brief Return the stable internal-ID universe; deleted tail IDs are not reused implicitly. */
  [[nodiscard]] auto universe_size() const -> size_t {
    auto snapshot = current_scalar_snapshot();
    return snapshot == nullptr ? 0 : snapshot->universe_size();
  }

  /** @brief Read current exact/raw vector bytes without pinning a multi-read query view. */
  auto get_raw_vector(IDType id, std::string &value) const -> bool {
    std::lock_guard<std::mutex> lock(publication_mutex_);
    return get_cf_value(vectors_cf_, encode_id(id), value, rocksdb::ReadOptions{});
  }

  /** @brief Read current quantized vector bytes without pinning a multi-read query view. */
  auto get_quantized_vector(IDType id, std::string &value) const -> bool {
    std::lock_guard<std::mutex> lock(publication_mutex_);
    return get_cf_value(quantized_vectors_cf_, encode_id(id), value, rocksdb::ReadOptions{});
  }

  /** @brief Acquire a generation-stable RecordStore and ScalarIndex pair for one query. */
  [[nodiscard]] auto acquire_query_view() const -> std::unique_ptr<QueryView> {
    std::lock_guard<std::mutex> lock(publication_mutex_);
    auto scalar_snapshot = std::atomic_load_explicit(&scalar_snapshot_, std::memory_order_acquire);
    auto *rocks_snapshot = db_->GetSnapshot();
    if (rocks_snapshot == nullptr) {
      throw std::runtime_error("Failed to acquire RocksDB query snapshot");
    }
    return std::unique_ptr<QueryView>(
        new QueryView(this, rocks_snapshot, std::move(scalar_snapshot)));
  }

  /** @brief Return the latest immutable scalar snapshot for index-only planning. */
  [[nodiscard]] auto current_scalar_snapshot() const -> std::shared_ptr<const ScalarSnapshot> {
    return std::atomic_load_explicit(&scalar_snapshot_, std::memory_order_acquire);
  }

  /** @brief Create a RocksDB checkpoint containing every v2 column family. */
  void save_checkpoint(const std::string &path) const {
    rocksdb::Checkpoint *raw_checkpoint = nullptr;
    auto status = rocksdb::Checkpoint::Create(db_.get(), &raw_checkpoint);
    std::unique_ptr<rocksdb::Checkpoint> checkpoint(raw_checkpoint);
    if (!status.ok()) {
      throw std::runtime_error("Failed to create RocksDB checkpoint object: " + status.ToString());
    }
    status = checkpoint->CreateCheckpoint(path);
    if (!status.ok()) {
      throw std::runtime_error("Failed to create RocksDB checkpoint at " + path + ": " +
                               status.ToString());
    }
  }

 private:
  static constexpr std::string_view kSchemaVersionKey = "schema_version";
  static constexpr std::string_view kGenerationKey = "generation";
  static constexpr std::string_view kLiveCountKey = "live_count";
  static constexpr std::string_view kUniverseSizeKey = "universe_size";
  static constexpr std::string_view kIndexedFieldsKey = "indexed_fields";

  using UnsignedIDType = std::make_unsigned_t<IDType>;

  /** @brief Encode an internal ID as fixed-width big-endian bytes for stable ordering. */
  [[nodiscard]] static auto encode_id(IDType id) -> std::string {
    auto value = static_cast<UnsignedIDType>(id);
    std::string encoded(sizeof(UnsignedIDType), '\0');
    for (size_t offset = 0; offset < sizeof(UnsignedIDType); ++offset) {
      encoded[sizeof(UnsignedIDType) - offset - 1] = static_cast<char>(value & 0xFFU);
      value >>= 8U;
    }
    return encoded;
  }

  /** @brief Decode one fixed-width big-endian internal-ID key. */
  [[nodiscard]] static auto decode_id(const rocksdb::Slice &key) -> IDType {
    if (key.size() != sizeof(UnsignedIDType)) {
      throw std::runtime_error("Invalid v2 internal-ID key width");
    }
    UnsignedIDType value = 0;
    for (size_t offset = 0; offset < key.size(); ++offset) {
      value = static_cast<UnsignedIDType>((value << 8U) |
                                          static_cast<unsigned char>(key.data()[offset]));
    }
    return static_cast<IDType>(value);
  }

  /** @brief Parse one unsigned decimal metadata value with full overflow validation. */
  template <typename ValueType>
  [[nodiscard]] static auto parse_unsigned(std::string_view raw, const char *name) -> ValueType {
    static_assert(std::is_unsigned_v<ValueType>);
    ValueType result = 0;
    auto [end, error] = std::from_chars(raw.data(), raw.data() + raw.size(), result);
    if (error != std::errc{} ||  // NOLINT(whitespace/braces)
        end != raw.data() + raw.size()) {
      throw std::runtime_error(std::string("Invalid v2 metadata value for ") + name);
    }
    return result;
  }

  /** @brief Encode normalized indexed fields with length prefixes to avoid delimiter ambiguity. */
  [[nodiscard]] auto encode_indexed_fields() const -> std::string {
    std::string encoded;
    for (const auto &field : config_.indexed_fields_) {
      encoded += std::to_string(field.size());
      encoded.push_back(':');
      encoded += field;
    }
    return encoded;
  }

  /** @brief Open all required CFs or reject a legacy/partial directory before modifying it. */
  void open() {
    namespace fs = std::filesystem;
    auto current_path = fs::path(config_.db_path_) / "CURRENT";
    auto existing = fs::exists(current_path);
    auto expected = RocksDBRecordStoreSchema::column_families();

    if (existing) {
      std::vector<std::string> actual;
      rocksdb::DBOptions list_options;
      auto status = rocksdb::DB::ListColumnFamilies(list_options, config_.db_path_, &actual);
      if (!status.ok()) {
        throw std::runtime_error("Failed to list RocksDB column families: " + status.ToString());
      }
      auto sorted_actual = actual;
      auto sorted_expected = expected;
      std::sort(sorted_actual.begin(), sorted_actual.end());
      std::sort(sorted_expected.begin(), sorted_expected.end());
      if (sorted_actual != sorted_expected) {
        throw std::runtime_error(
            "Refusing to open a legacy or partial RocksDB directory as v2; migrate to a new path");
      }
    } else if (!config_.create_if_missing_) {
      throw std::runtime_error("v2 RocksDB record store does not exist");
    }

    auto parent = fs::path(config_.db_path_).parent_path();
    if (!parent.empty()) {
      fs::create_directories(parent);
    }

    rocksdb::Options options;
    options.create_if_missing = config_.create_if_missing_;
    options.create_missing_column_families = !existing;
    std::vector<rocksdb::ColumnFamilyDescriptor> descriptors;
    descriptors.reserve(expected.size());
    for (const auto &name : expected) {
      descriptors.emplace_back(name, rocksdb::ColumnFamilyOptions(options));
    }

    auto status = rocksdb::DB::Open(rocksdb::DBOptions(options),
                                    config_.db_path_,
                                    descriptors,
                                    &handles_,
                                    &db_);
    if (!status.ok()) {
      throw std::runtime_error("Failed to open v2 RocksDB record store: " + status.ToString());
    }
    try {
      if (handles_.size() != expected.size()) {
        throw std::runtime_error("RocksDB returned an incomplete v2 column-family handle set");
      }
      for (size_t i = 0; i < expected.size(); ++i) {
        handles_by_name_.emplace(expected[i], handles_[i]);
      }
      records_cf_ = require_cf(RocksDBRecordStoreSchema::kRecords);
      vectors_cf_ = require_cf(RocksDBRecordStoreSchema::kVectors);
      quantized_vectors_cf_ = require_cf(RocksDBRecordStoreSchema::kQuantizedVectors);
      item_ids_cf_ = require_cf(RocksDBRecordStoreSchema::kItemIds);
      scalar_indexes_cf_ = require_cf(RocksDBRecordStoreSchema::kScalarIndexes);
      metadata_cf_ = require_cf(RocksDBRecordStoreSchema::kMetadata);

      initialize_or_load_metadata(existing);
      auto records = load_all_records();
      if (records.size() != live_count_.load(std::memory_order_relaxed)) {
        throw std::runtime_error("v2 live_count does not match canonical records CF");
      }
      auto snapshot = ScalarSnapshot::build(generation_.load(std::memory_order_relaxed),
                                            universe_size_.load(std::memory_order_relaxed),
                                            config_.indexed_fields_,
                                            records);
      std::atomic_store_explicit(&scalar_snapshot_, std::move(snapshot), std::memory_order_release);
    } catch (...) {
      close();
      throw;
    }
  }

  /** @brief Initialize metadata for a new directory or validate and load an existing v2 store. */
  void initialize_or_load_metadata(bool existing) {
    if (!existing) {
      rocksdb::WriteBatch batch;
      put_metadata(batch, 0, 0, 0);
      batch.Put(metadata_cf_,
                kSchemaVersionKey,
                std::to_string(RocksDBRecordStoreSchema::kVersion));
      batch.Put(metadata_cf_, kIndexedFieldsKey, encode_indexed_fields());
      if (!write(batch)) {
        throw std::runtime_error("Failed to initialize v2 RocksDB metadata");
      }
      return;
    }

    auto schema = require_metadata(kSchemaVersionKey);
    if (schema != std::to_string(RocksDBRecordStoreSchema::kVersion)) {
      throw std::runtime_error("Unsupported RocksDB record-store schema version: " + schema);
    }
    if (require_metadata(kIndexedFieldsKey) != encode_indexed_fields()) {
      throw std::runtime_error(
          "Configured indexed fields do not match the persisted v2 record-store schema");
    }
    generation_.store(parse_unsigned<uint64_t>(require_metadata(kGenerationKey), "generation"),
                      std::memory_order_relaxed);
    live_count_.store(parse_unsigned<size_t>(require_metadata(kLiveCountKey), "live_count"),
                      std::memory_order_relaxed);
    universe_size_.store(parse_unsigned<size_t>(require_metadata(kUniverseSizeKey),
                                                "universe_size"),
                         std::memory_order_relaxed);
  }

  /** @brief Close CF handles before closing the owning DB. */
  void close() noexcept {
    if (db_ == nullptr) {
      return;
    }
    std::atomic_store_explicit(&scalar_snapshot_,
                               std::shared_ptr<const ScalarSnapshot>{},
                               std::memory_order_release);
    for (auto *handle : handles_) {
      (void)db_->DestroyColumnFamilyHandle(handle);
    }
    handles_.clear();
    handles_by_name_.clear();
    (void)db_->Close();
    db_.reset();
  }

  /** @brief Resolve a required CF handle after the ordered open operation. */
  [[nodiscard]] auto require_cf(std::string_view name) const -> rocksdb::ColumnFamilyHandle * {
    auto handle = handles_by_name_.find(std::string(name));
    if (handle == handles_by_name_.end()) {
      throw std::runtime_error("Missing v2 RocksDB column family: " + std::string(name));
    }
    return handle->second;
  }

  /** @brief Get one CF value and distinguish absence from storage failures. */
  [[nodiscard]] auto get_cf_value(rocksdb::ColumnFamilyHandle *cf,
                                  const std::string &key,
                                  std::string &value,
                                  const rocksdb::ReadOptions &options) const -> bool {
    auto status = db_->Get(options, cf, key, &value);
    if (status.IsNotFound()) {
      value.clear();
      return false;
    }
    if (!status.ok()) {
      throw std::runtime_error("RocksDB v2 read failed: " + status.ToString());
    }
    return true;
  }

  /** @brief Read one required metadata value or fail opening a corrupt/partial v2 store. */
  [[nodiscard]] auto require_metadata(std::string_view key) const -> std::string {
    std::string value;
    if (!get_cf_value(metadata_cf_, std::string(key), value, rocksdb::ReadOptions{})) {
      throw std::runtime_error("Missing v2 RocksDB metadata key: " + std::string(key));
    }
    return value;
  }

  /** @brief Resolve an item ID using caller-provided latest or snapshot read options. */
  [[nodiscard]] auto find_by_item_id_with_options(const std::string &item_id,
                                                  const rocksdb::ReadOptions &options) const
      -> std::optional<IDType> {
    if (item_id.empty()) {
      return std::nullopt;
    }
    std::string encoded;
    if (!get_cf_value(item_ids_cf_, item_id, encoded, options)) {
      return std::nullopt;
    }
    return decode_id(rocksdb::Slice(encoded));
  }

  /** @brief Scan canonical rows in internal-ID order for restart and snapshot construction. */
  [[nodiscard]] auto load_all_records() const -> std::vector<SnapshotRecord> {
    std::vector<SnapshotRecord> records;
    rocksdb::ReadOptions options;
    options.fill_cache = false;
    std::unique_ptr<rocksdb::Iterator> iterator(db_->NewIterator(options, records_cf_));
    for (iterator->SeekToFirst(); iterator->Valid(); iterator->Next()) {
      auto id = decode_id(iterator->key());
      auto value = iterator->value();
      records.emplace_back(id, ScalarData::deserialize(value.data(), value.size()));
    }
    if (!iterator->status().ok()) {
      throw std::runtime_error("Failed to scan v2 records CF: " + iterator->status().ToString());
    }
    return records;
  }

  /** @brief Replace or append one row in the sorted snapshot-construction input. */
  static void replace_snapshot_record(std::vector<SnapshotRecord> &records,
                                      IDType id,
                                      const ScalarData &scalar) {
    auto existing =
        std::lower_bound(records.begin(), records.end(), id, [](const auto &record, IDType target) {
          return record.first < target;
        });
    if (existing != records.end() && existing->first == id) {
      existing->second = scalar;
    } else {
      records.insert(existing, {id, scalar});
    }
  }

  /** @brief Remove one row from the sorted snapshot-construction input. */
  static void erase_snapshot_record(std::vector<SnapshotRecord> &records, IDType id) {
    auto existing =
        std::lower_bound(records.begin(), records.end(), id, [](const auto &record, IDType target) {
          return record.first < target;
        });
    if (existing != records.end() && existing->first == id) {
      records.erase(existing);
    }
  }

  /** @brief Return the next mutation generation or reject uint64 exhaustion. */
  [[nodiscard]] auto checked_next_generation() const -> uint64_t {
    auto current = generation_.load(std::memory_order_relaxed);
    if (current == std::numeric_limits<uint64_t>::max()) {
      throw std::overflow_error("RocksDB record-store generation exhausted");
    }
    return current + 1;
  }

  /** @brief Put an external item-ID mapping when the external ID is non-empty. */
  void put_item_id_entry(rocksdb::WriteBatch &batch, const std::string &item_id, IDType id) const {
    if (!item_id.empty()) {
      batch.Put(item_ids_cf_, item_id, encode_id(id));
    }
  }

  /** @brief Delete an external item-ID mapping when the external ID is non-empty. */
  void delete_item_id_entry(rocksdb::WriteBatch &batch, const std::string &item_id) const {
    if (!item_id.empty()) {
      batch.Delete(item_ids_cf_, item_id);
    }
  }

  /** @brief Add persisted index entries for configured fields in one canonical row. */
  void put_scalar_index_entries(rocksdb::WriteBatch &batch,
                                IDType id,
                                const ScalarData &scalar) const {
    for (const auto &field : config_.indexed_fields_) {
      auto value = scalar.metadata.find(field);
      if (value != scalar.metadata.end()) {
        batch.Put(scalar_indexes_cf_,
                  index_encoding::make_field_index_key(field,
                                                       index_encoding::encode_value(value->second),
                                                       id),
                  rocksdb::Slice{});
      }
    }
  }

  /** @brief Delete persisted index entries derived from an older canonical row. */
  void delete_scalar_index_entries(rocksdb::WriteBatch &batch,
                                   IDType id,
                                   const ScalarData &scalar) const {
    for (const auto &field : config_.indexed_fields_) {
      auto value = scalar.metadata.find(field);
      if (value != scalar.metadata.end()) {
        batch.Delete(scalar_indexes_cf_,
                     index_encoding::make_field_index_key(field,
                                                          index_encoding::encode_value(
                                                              value->second),
                                                          id));
      }
    }
  }

  /** @brief Persist generation and cardinality in the same batch as row components. */
  void put_metadata(rocksdb::WriteBatch &batch,
                    uint64_t generation,
                    size_t live_count,
                    size_t universe_size) const {
    batch.Put(metadata_cf_, kGenerationKey, std::to_string(generation));
    batch.Put(metadata_cf_, kLiveCountKey, std::to_string(live_count));
    batch.Put(metadata_cf_, kUniverseSizeKey, std::to_string(universe_size));
  }

  /** @brief Commit one cross-CF logical mutation through RocksDB's WAL. */
  auto write(rocksdb::WriteBatch &batch) const -> bool {
    rocksdb::WriteOptions options;
    options.sync = config_.sync_writes_;
    return db_->Write(options, &batch).ok();
  }

  /** @brief Publish counters and immutable postings only after the matching DB batch commits. */
  void publish(std::shared_ptr<const ScalarSnapshot> snapshot,
               size_t live_count,
               size_t universe_size) {
    generation_.store(snapshot->generation(), std::memory_order_release);
    live_count_.store(live_count, std::memory_order_release);
    universe_size_.store(universe_size, std::memory_order_release);
    std::atomic_store_explicit(&scalar_snapshot_, std::move(snapshot), std::memory_order_release);
  }

  RocksDBRecordStoreConfig config_;  ///< Immutable storage path, durability and indexed fields.
  std::unique_ptr<rocksdb::DB> db_;  ///< Owns the database after all CF handles are released.
  std::vector<rocksdb::ColumnFamilyHandle *> handles_;  ///< All handles in open descriptor order.
  std::unordered_map<std::string, rocksdb::ColumnFamilyHandle *>
      handles_by_name_;  ///< Name lookup used only during initialization.
  rocksdb::ColumnFamilyHandle *records_cf_ = nullptr;            ///< Canonical ScalarData CF.
  rocksdb::ColumnFamilyHandle *vectors_cf_ = nullptr;            ///< Exact/raw vector CF.
  rocksdb::ColumnFamilyHandle *quantized_vectors_cf_ = nullptr;  ///< Optional ANN vector CF.
  rocksdb::ColumnFamilyHandle *item_ids_cf_ = nullptr;           ///< External-to-internal ID CF.
  rocksdb::ColumnFamilyHandle *scalar_indexes_cf_ = nullptr;     ///< Durable scalar postings CF.
  rocksdb::ColumnFamilyHandle *metadata_cf_ = nullptr;  ///< Schema and generation metadata CF.
  mutable std::mutex publication_mutex_;  ///< Serializes writes and paired query-view acquisition.
  std::atomic<uint64_t> generation_{0};   ///< Latest committed and published logical generation.
  std::atomic<size_t> live_count_{0};     ///< Number of canonical records in the latest generation.
  std::atomic<size_t> universe_size_{0};  ///< Exclusive upper bound of all assigned internal IDs.
  std::shared_ptr<const ScalarSnapshot>
      scalar_snapshot_;  ///< Atomically published immutable query-hot scalar postings.
};

}  // namespace alaya
