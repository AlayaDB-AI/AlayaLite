// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <functional>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "storage/rocksdb_record_store.hpp"
#include "storage/rocksdb_storage.hpp"
#include "utils/platform_fs.hpp"

namespace alaya {

/** @brief Active-store pointer atomically published after a complete v2 migration. */
struct RecordStoreManifest {
  uint32_t format_version_ = 1;                                   ///< Text manifest format version.
  uint32_t schema_version_ = RocksDBRecordStoreSchema::kVersion;  ///< RocksDB CF schema version.
  std::string store_directory_;  ///< Root-relative directory containing the active RocksDB store.
  uint64_t generation_ = 0;      ///< Persisted mutation generation of the published store.
  size_t live_count_ = 0;        ///< Canonical records present when the store was published.

  /** @brief Serialize the active-store pointer as inspectable line-oriented text. */
  [[nodiscard]] auto serialize() const -> std::string {
    std::ostringstream output;
    output << "format_version=" << format_version_ << '\n';
    output << "schema_version=" << schema_version_ << '\n';
    output << "store_directory=" << store_directory_ << '\n';
    output << "generation=" << generation_ << '\n';
    output << "live_count=" << live_count_ << '\n';
    return output.str();
  }

  /** @brief Parse and validate an active-store manifest. */
  [[nodiscard]] static auto deserialize(const std::string &raw)
      -> std::optional<RecordStoreManifest> {
    RecordStoreManifest manifest;
    bool has_format_version = false;
    bool has_schema_version = false;
    bool has_store_directory = false;
    bool has_generation = false;
    bool has_live_count = false;
    std::istringstream input(raw);
    std::string line;
    while (std::getline(input, line)) {
      auto delimiter = line.find('=');
      if (delimiter == std::string::npos) {
        return std::nullopt;
      }
      auto key = line.substr(0, delimiter);
      auto value = line.substr(delimiter + 1);
      if (key == "format_version") {
        if (has_format_version || !parse_unsigned(value, manifest.format_version_)) {
          return std::nullopt;
        }
        has_format_version = true;
      } else if (key == "schema_version") {
        if (has_schema_version || !parse_unsigned(value, manifest.schema_version_)) {
          return std::nullopt;
        }
        has_schema_version = true;
      } else if (key == "store_directory") {
        if (has_store_directory) {
          return std::nullopt;
        }
        manifest.store_directory_ = value;
        has_store_directory = true;
      } else if (key == "generation") {
        if (has_generation || !parse_unsigned(value, manifest.generation_)) {
          return std::nullopt;
        }
        has_generation = true;
      } else if (key == "live_count") {
        if (has_live_count || !parse_unsigned(value, manifest.live_count_)) {
          return std::nullopt;
        }
        has_live_count = true;
      }
    }
    if (!has_format_version || !has_schema_version || !has_store_directory || !has_generation ||
        !has_live_count || manifest.format_version_ != 1 ||
        manifest.schema_version_ != RocksDBRecordStoreSchema::kVersion ||
        !is_safe_relative_store_path(manifest.store_directory_)) {
      return std::nullopt;
    }
    return manifest;
  }

 private:
  /** @brief Parse one complete unsigned decimal value. */
  template <typename ValueType>
  [[nodiscard]] static auto parse_unsigned(std::string_view raw, ValueType &result) -> bool {
    static_assert(std::is_unsigned_v<ValueType>);
    auto [end, error] = std::from_chars(raw.data(), raw.data() + raw.size(), result);
    return error == std::errc{} &&  // NOLINT(whitespace/braces)
           end == raw.data() + raw.size();
  }

  /** @brief Reject absolute and parent-traversing store paths before joining with the layout root.
   */
  [[nodiscard]] static auto is_safe_relative_store_path(const std::string &raw) -> bool {
    auto path = std::filesystem::path(raw);
    if (raw.empty() || path.is_absolute()) {
      return false;
    }
    for (const auto &component : path) {
      if (component == "..") {
        return false;
      }
    }
    return true;
  }
};

/** @brief Owns the atomic CURRENT pointer for independently built record-store directories. */
class RecordStoreLayout {
 public:
  /** @brief Bind the layout to a root containing CURRENT and stores/. */
  explicit RecordStoreLayout(std::filesystem::path root) : root_(std::move(root)) {
    if (root_.empty()) {
      throw std::invalid_argument("RecordStoreLayout root cannot be empty");
    }
  }

  /** @brief Return the root-relative directory used for a named completed migration. */
  [[nodiscard]] auto store_relative_path(const std::string &migration_id) const
      -> std::filesystem::path {
    validate_migration_id(migration_id);
    return std::filesystem::path("stores") / migration_id;
  }

  /** @brief Resolve the absolute directory used for a named completed migration. */
  [[nodiscard]] auto store_path(const std::string &migration_id) const -> std::filesystem::path {
    return root_ / store_relative_path(migration_id);
  }

  /** @brief Read CURRENT, returning no manifest for a fresh layout. */
  [[nodiscard]] auto current_manifest() const -> std::optional<RecordStoreManifest> {
    auto path = current_path();
    if (!std::filesystem::exists(path)) {
      return std::nullopt;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
      throw std::runtime_error("Failed to open record-store CURRENT manifest");
    }
    std::string raw((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    auto manifest = RecordStoreManifest::deserialize(raw);
    if (!manifest.has_value()) {
      throw std::runtime_error("Invalid record-store CURRENT manifest");
    }
    return manifest;
  }

  /** @brief Resolve the currently published store directory, if any. */
  [[nodiscard]] auto current_store_path() const -> std::optional<std::filesystem::path> {
    auto manifest = current_manifest();
    if (!manifest.has_value()) {
      return std::nullopt;
    }
    return root_ / manifest->store_directory_;
  }

  /**
   * @brief Atomically publish one already-complete store directory through CURRENT.
   *
   * The previous CURRENT remains authoritative until the temporary manifest has been fsynced and
   * atomically renamed. A crash before replacement leaves only an unreferenced new store.
   */
  void publish(const RecordStoreManifest &manifest) const {
    auto relative = std::filesystem::path(manifest.store_directory_);
    auto store = root_ / relative;
    if (!RecordStoreManifest::deserialize(manifest.serialize()).has_value() ||
        !std::filesystem::exists(store / "CURRENT")) {
      throw std::invalid_argument("Cannot publish an invalid or incomplete record-store manifest");
    }
    std::filesystem::create_directories(root_);
    auto temporary = current_path();
    temporary += ".tmp." + std::to_string(platform::get_pid());
    {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      if (!output.is_open()) {
        throw std::runtime_error("Failed to create temporary record-store CURRENT manifest");
      }
      auto raw = manifest.serialize();
      output.write(raw.data(), static_cast<std::streamsize>(raw.size()));
      output.flush();
      if (!output.good()) {
        throw std::runtime_error("Failed to write temporary record-store CURRENT manifest");
      }
    }
    platform::sync_file_or_throw(temporary);
    platform::atomic_replace(temporary, current_path());
    platform::sync_directory_or_throw(root_);
  }

 private:
  /** @brief Return the atomic active-store pointer path. */
  [[nodiscard]] auto current_path() const -> std::filesystem::path { return root_ / "CURRENT"; }

  /** @brief Restrict migration IDs to one safe directory component. */
  static void validate_migration_id(const std::string &migration_id) {
    auto path = std::filesystem::path(migration_id);
    if (migration_id.empty() || path.is_absolute() || path.has_parent_path() ||
        migration_id == "." || migration_id == "..") {
      throw std::invalid_argument("Migration ID must be one relative path component");
    }
  }

  std::filesystem::path root_;  ///< Parent of CURRENT and immutable stores/ generations.
};

/** @brief Raw and optional quantized bytes supplied while migrating one legacy row. */
struct MigratedVectorPayload {
  std::string raw_vector_;  ///< Exact vector bytes required for reranking and brute force.
  std::optional<std::string> quantized_vector_;  ///< Optional ANN representation for this ID.
};

/**
 * @brief Migrate legacy scalar rows and caller-provided vectors into a new v2 directory.
 *
 * The legacy store is read only through its const API. The vector reader must return the raw vector
 * belonging to the same internal ID and may also provide a quantized representation. On any failure
 * the staging directory is removed and CURRENT is not changed.
 */
template <typename IDType>
auto migrate_legacy_record_store(
    const RocksDBStorage<IDType> &legacy,
    const RecordStoreLayout &layout,
    const std::string &migration_id,
    std::vector<std::string> indexed_fields,
    const std::function<std::optional<MigratedVectorPayload>(IDType)> &read_vector)
    -> RecordStoreManifest {
  namespace fs = std::filesystem;
  auto final_path = layout.store_path(migration_id);
  auto stores_path = final_path.parent_path();
  fs::create_directories(stores_path);
  if (fs::exists(final_path)) {
    throw std::runtime_error("Record-store migration target already exists: " +
                             final_path.string());
  }

  auto staging_path = final_path;
  staging_path += ".tmp." + std::to_string(platform::get_pid());
  if (fs::exists(staging_path)) {
    throw std::runtime_error("Record-store migration staging path already exists: " +
                             staging_path.string());
  }

  try {
    RocksDBRecordStoreConfig config;
    config.db_path_ = staging_path.string();
    config.indexed_fields_ = std::move(indexed_fields);
    config.sync_writes_ = true;
    uint64_t generation = 0;
    size_t live_count = 0;
    {
      RocksDBRecordStore<IDType> target(std::move(config));
      auto records = legacy.scan_with_filter([](const ScalarData &) {
        return true;
      });
      for (const auto &[id, scalar] : records) {
        auto vectors = read_vector(id);
        if (!vectors.has_value() || vectors->raw_vector_.empty()) {
          throw std::runtime_error("Missing raw vector while migrating internal ID " +
                                   std::to_string(id));
        }
        std::optional<std::string_view> quantized;
        if (vectors->quantized_vector_.has_value()) {
          quantized = std::string_view(*vectors->quantized_vector_);
        }
        if (!target.upsert(id, scalar, vectors->raw_vector_, quantized)) {
          throw std::runtime_error("Failed to migrate internal ID " + std::to_string(id));
        }
      }
      generation = target.generation();
      live_count = target.size();
    }

    platform::atomic_replace_no_overwrite(staging_path, final_path);
    platform::sync_directory_or_throw(stores_path);
    RecordStoreManifest manifest{
        .store_directory_ = layout.store_relative_path(migration_id).generic_string(),
        .generation_ = generation,
        .live_count_ = live_count,
    };
    layout.publish(manifest);
    return manifest;
  } catch (...) {
    std::error_code cleanup_error;
    fs::remove_all(staging_path, cleanup_error);
    throw;
  }
}

}  // namespace alaya
