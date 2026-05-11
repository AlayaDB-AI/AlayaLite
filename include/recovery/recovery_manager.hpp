// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "recovery/snapshot_manifest.hpp"
#include "recovery/write_ahead_log.hpp"
#include "utils/log.hpp"

namespace alaya::recovery {

namespace fs = std::filesystem;

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

  [[nodiscard]] auto next_operation_id() const -> uint64_t {
    uint64_t max_seen = 0;
    auto manifest = current_snapshot();
    if (manifest.has_value()) {
      max_seen = manifest->applied_through_op_id;
    }
    wal_.replayable_records(0, &max_seen);
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

  // TODO(P2): Graph save and RocksDB checkpoint are not atomic. A crash between
  // them creates an inconsistent snapshot. Consider a two-phase commit protocol
  // where both components are written before the CURRENT pointer is updated.
  auto publish_snapshot(const SnapshotManifest &manifest, const fs::path &snapshot_dir) const
      -> void {
    ensure_layout();
    auto manifest_path = snapshot_dir / "manifest.txt";
    write_text_atomically(manifest_path, manifest.serialize());
    write_text_atomically(current_path_, manifest.snapshot_id + "\n");
    LOG_INFO("recovery: published snapshot id={} applied_through={}",
             manifest.snapshot_id,
             manifest.applied_through_op_id);
    wal_.truncate();
    remove_old_snapshots(manifest.snapshot_id);
  }

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
    return snapshots_dir_ / manifest->snapshot_id;
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

  auto restore_active_rocksdb_from_snapshot(const SnapshotManifest &manifest,
                                            const fs::path &snapshot_dir) const -> void {
    if (active_rocksdb_path_.empty() || manifest.rocksdb_dir.empty()) {
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

    sync_file(tmp_path);
    std::error_code ec;
    fs::rename(tmp_path, path, ec);
    if (ec) {
      fs::remove(path, ec);
      ec.clear();
      fs::rename(tmp_path, path, ec);
      if (ec) {
        throw std::runtime_error("Failed to publish recovery file at " + path.string());
      }
    }
    sync_directory(path.parent_path());
  }

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
    sync_directory(target);
  }

  static auto sync_file(const fs::path &path) -> void {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      return;
    }
    ::fsync(fd);
    ::close(fd);
  }

  static auto sync_directory(const fs::path &path) -> void {
    int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd < 0) {
      return;
    }
    ::fsync(fd);
    ::close(fd);
  }

  fs::path root_dir_;
  fs::path snapshots_dir_;
  fs::path current_path_;
  WriteAheadLog wal_;
  fs::path active_rocksdb_path_;
};

}  // namespace alaya::recovery
