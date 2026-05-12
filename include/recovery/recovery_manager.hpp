// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "recovery/snapshot_manifest.hpp"
#include "recovery/write_ahead_log.hpp"
#include "utils/log.hpp"
#include "utils/platform_fs.hpp"

namespace alaya::recovery {

namespace fs = std::filesystem;

/**
 * @brief Coordinates snapshot metadata, WAL replay and RocksDB checkpoint restoration.
 *
 * RecoveryManager owns the recovery directory layout under root_dir. It publishes snapshots through
 * an atomic CURRENT pointer, delegates mutation durability to WriteAheadLog and restores the active
 * RocksDB directory from snapshot checkpoints when needed.
 */
class RecoveryManager {
 public:
  RecoveryManager(fs::path root_dir, fs::path active_rocksdb_path)
      : root_dir_(std::move(root_dir)),
        snapshots_dir_(root_dir_ / "snapshots"),
        current_path_(root_dir_ / "CURRENT"),
        wal_(root_dir_ / "wal.bin"),
        active_rocksdb_path_(std::move(active_rocksdb_path)) {}

  [[nodiscard]] auto enabled() const -> bool { return !root_dir_.empty(); }

  auto ensure_layout() const -> void {
    if (!enabled()) {
      return;
    }
    fs::create_directories(snapshots_dir_);
  }

  /**
   * @brief Calculates the next operation id from the current snapshot and WAL contents.
   *
   * The snapshot applied-through id gives the lower bound, and scanning the WAL advances it to the
   * largest operation id seen in prepared or committed frames so new operations keep increasing
   * monotonically after restart.
   */
  [[nodiscard]] auto next_operation_id() const -> uint64_t {
    uint64_t max_seen = 0;
    auto manifest = current_snapshot();
    if (manifest.has_value()) {
      max_seen = manifest->applied_through_op_id_;
    }
    (void)wal_.replayable_records(0, &max_seen);
    return max_seen + 1;
  }

  [[nodiscard]] auto create_snapshot_dir() const -> fs::path {
    ensure_layout();
    auto now = SnapshotManifest::current_unix_ms();
    std::ostringstream name;
    name << "snapshot-" << now;
    auto snapshot_dir = snapshots_dir_ / name.str();
    fs::create_directories(snapshot_dir);
    return snapshot_dir;
  }

  // TODO(P2): The snapshot directory is only published after graph files and the RocksDB
  // checkpoint are written, so a process crash before CURRENT is updated leaves the old snapshot
  // active. Remaining risk: component writes are not durably staged as a unit, RocksDB checkpoint
  // failures are not surfaced to the publisher, and there is no READY marker or checksum proving
  // that every component reached stable storage before CURRENT is advanced.
  /**
   * @brief Publishes a completed snapshot and makes it the recovery CURRENT target.
   *
   * The manifest and CURRENT pointer are written atomically, then the WAL is truncated because all
   * operations through the manifest are represented in the snapshot. Older snapshot directories are
   * cleaned up after the new snapshot is visible.
   */
  auto publish_snapshot(const SnapshotManifest &manifest, const fs::path &snapshot_dir) const
      -> void {
    ensure_layout();
    auto manifest_path = snapshot_dir / "manifest.txt";
    write_text_atomically(manifest_path, manifest.serialize());
    write_text_atomically(current_path_, manifest.snapshot_id_ + "\n");
    LOG_INFO("recovery: published snapshot id={} applied_through={}",
             manifest.snapshot_id_,
             manifest.applied_through_op_id_);
    wal_.truncate();
    remove_old_snapshots(manifest.snapshot_id_);
  }

  /**
   * @brief Loads the manifest for the snapshot named by the CURRENT pointer.
   *
   * Missing recovery state, unreadable files and empty CURRENT contents are treated as no snapshot
   * being available, allowing callers to fall back to WAL-only recovery or a fresh layout.
   */
  [[nodiscard]] auto current_snapshot() const -> std::optional<SnapshotManifest> {
    if (!enabled() || !fs::exists(current_path_)) {
      return std::nullopt;
    }

    auto current_id = read_text(current_path_);
    if (!current_id.has_value()) {
      return std::nullopt;
    }

    std::string snapshot_id = trim(current_id.value());
    if (snapshot_id.empty()) {
      return std::nullopt;
    }

    auto manifest_path = snapshots_dir_ / snapshot_id / "manifest.txt";
    if (!fs::exists(manifest_path)) {
      return std::nullopt;
    }

    auto manifest_raw = read_text(manifest_path);
    if (!manifest_raw.has_value()) {
      return std::nullopt;
    }
    return SnapshotManifest::deserialize(manifest_raw.value());
  }

  [[nodiscard]] auto current_snapshot_dir() const -> std::optional<fs::path> {
    auto manifest = current_snapshot();
    if (!manifest.has_value()) {
      return std::nullopt;
    }
    return snapshots_dir_ / manifest->snapshot_id_;
  }

  [[nodiscard]] auto replayable_records(uint64_t applied_through,
                                        uint64_t *max_seen_op_id = nullptr) const
      -> std::vector<WalRecord> {
    return wal_.replayable_records(applied_through, max_seen_op_id);
  }

  auto append_prepare(const WalRecord &record) const -> void { wal_.append_prepare(record); }

  auto append_commit(uint64_t op_id, MutationType mutation_type) const -> void {
    wal_.append_commit(op_id, mutation_type);
  }

  auto sync_wal() const -> void { wal_.sync(); }

  /**
   * @brief Replaces the active RocksDB directory with the checkpoint stored in a snapshot.
   *
   * If RocksDB recovery is not configured or the snapshot does not include a checkpoint, the method
   * exits without modifying the active directory. Existing active contents are removed before the
   * checkpoint is copied into place.
   */
  auto restore_active_rocksdb_from_snapshot(const SnapshotManifest &manifest,
                                            const fs::path &snapshot_dir) const -> void {
    if (active_rocksdb_path_.empty() || manifest.rocksdb_dir_.empty()) {
      return;
    }

    auto source = manifest.rocksdb_path(snapshot_dir);
    if (source.empty() || !fs::exists(source)) {
      LOG_WARN("recovery: snapshot rocksdb checkpoint missing at {}", source.string());
      return;
    }

    fs::create_directories(active_rocksdb_path_.parent_path());
    std::error_code ec;
    fs::remove_all(active_rocksdb_path_, ec);
    ec.clear();
    copy_directory_recursive(source, active_rocksdb_path_);
    LOG_INFO("recovery: restored rocksdb checkpoint from {}", source.string());
  }

 private:
  /**
   * @brief Removes stale snapshot directories after a new snapshot is published.
   *
   * The currently published snapshot is preserved while all other directories under the snapshot
   * root are best-effort deleted with warnings for failures.
   */
  auto remove_old_snapshots(const std::string &current_snapshot_id) const -> void {
    if (!fs::exists(snapshots_dir_)) {
      return;
    }
    std::error_code ec;
    for (const auto &entry : fs::directory_iterator(snapshots_dir_, ec)) {
      if (ec) {
        break;
      }
      if (!entry.is_directory()) {
        continue;
      }
      if (entry.path().filename().string() == current_snapshot_id) {
        continue;
      }
      std::error_code remove_ec;
      fs::remove_all(entry.path(), remove_ec);
      if (remove_ec) {
        LOG_WARN("recovery: failed to remove old snapshot {}: {}",
                 entry.path().string(),
                 remove_ec.message());
      } else {
        LOG_INFO("recovery: removed old snapshot {}", entry.path().string());
      }
    }
  }

  static auto trim(std::string value) -> std::string {
    while (!value.empty() &&
           (value.back() == '\n' || value.back() == '\r' || value.back() == ' ')) {
      value.pop_back();
    }
    size_t first = 0;
    while (first < value.size() && value[first] == ' ') {
      ++first;
    }
    return value.substr(first);
  }

  static auto read_text(const fs::path &path) -> std::optional<std::string> {
    std::ifstream input(path);
    if (!input.is_open()) {
      return std::nullopt;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
  }

  /**
   * @brief Writes text through a temporary file and atomically replaces the target path.
   *
   * The temporary file is flushed and synced before replacement, then the parent directory is
   * synced so snapshot manifests and CURRENT pointer updates survive process or system crashes.
   */
  static auto write_text_atomically(const fs::path &path, const std::string &content) -> void {
    fs::create_directories(path.parent_path());
    auto tmp_path = path;
    tmp_path += ".tmp";

    {
      std::ofstream output(tmp_path, std::ios::binary | std::ios::trunc);
      if (!output.is_open()) {
        throw std::runtime_error("Failed to open " + tmp_path.string() + " for writing");
      }
      output.write(content.data(), static_cast<std::streamsize>(content.size()));
      output.flush();
      output.close();
    }

    platform::sync_file(tmp_path);
    platform::atomic_replace(tmp_path, path);
    platform::sync_directory(path.parent_path());
  }

  /**
   * @brief Recursively copies a snapshot checkpoint directory into the active target directory.
   *
   * Directories are recreated and regular files are overwritten at the destination, followed by a
   * directory sync to make the restored checkpoint durable.
   */
  static auto copy_directory_recursive(const fs::path &source, const fs::path &target) -> void {
    fs::create_directories(target);
    for (const auto &entry : fs::recursive_directory_iterator(source)) {
      auto relative = fs::relative(entry.path(), source);
      auto dest = target / relative;
      if (entry.is_directory()) {
        fs::create_directories(dest);
      } else if (entry.is_regular_file()) {
        fs::create_directories(dest.parent_path());
        fs::copy_file(entry.path(), dest, fs::copy_options::overwrite_existing);
      }
    }
    platform::sync_directory(target);
  }

  fs::path root_dir_;             ///< Recovery metadata root; empty disables recovery.
  fs::path snapshots_dir_;        ///< Directory that stores snapshot subdirectories.
  fs::path current_path_;         ///< Atomic pointer file for the current snapshot id.
  WriteAheadLog wal_;             ///< WAL for mutations not yet covered by a snapshot.
  fs::path active_rocksdb_path_;  ///< Live RocksDB directory restored from checkpoints.
};

}  // namespace alaya::recovery
