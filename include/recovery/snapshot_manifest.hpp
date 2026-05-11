// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <chrono>
#include <filesystem>  // NOLINT(build/c++17)
#include <optional>
#include <sstream>
#include <string>

namespace alaya::recovery {

namespace fs = std::filesystem;

struct SnapshotManifest {
  uint32_t format_version{1};
  std::string snapshot_id{};
  std::string reason{};
  uint64_t applied_through_op_id{0};
  uint64_t created_unix_ms{0};
  std::string graph_file{};
  std::string data_file{};
  std::string quant_file{};
  std::string rocksdb_dir{};

  [[nodiscard]] auto serialize() const -> std::string {
    std::ostringstream output;
    output << "format_version=" << format_version << '\n';
    output << "snapshot_id=" << snapshot_id << '\n';
    output << "reason=" << reason << '\n';
    output << "applied_through_op_id=" << applied_through_op_id << '\n';
    output << "created_unix_ms=" << created_unix_ms << '\n';
    output << "graph_file=" << graph_file << '\n';
    output << "data_file=" << data_file << '\n';
    output << "quant_file=" << quant_file << '\n';
    output << "rocksdb_dir=" << rocksdb_dir << '\n';
    return output.str();
  }

  static auto deserialize(const std::string &raw) -> std::optional<SnapshotManifest> {
    SnapshotManifest manifest;
    std::istringstream input(raw);
    std::string line;
    while (std::getline(input, line)) {
      auto delimiter = line.find('=');
      if (delimiter == std::string::npos) {
        continue;
      }
      auto key = line.substr(0, delimiter);
      auto value = line.substr(delimiter + 1);
      if (key == "format_version") {
        manifest.format_version = static_cast<uint32_t>(std::stoul(value));
      } else if (key == "snapshot_id") {
        manifest.snapshot_id = value;
      } else if (key == "reason") {
        manifest.reason = value;
      } else if (key == "applied_through_op_id") {
        manifest.applied_through_op_id = std::stoull(value);
      } else if (key == "created_unix_ms") {
        manifest.created_unix_ms = std::stoull(value);
      } else if (key == "graph_file") {
        manifest.graph_file = value;
      } else if (key == "data_file") {
        manifest.data_file = value;
      } else if (key == "quant_file") {
        manifest.quant_file = value;
      } else if (key == "rocksdb_dir") {
        manifest.rocksdb_dir = value;
      }
    }
    if (manifest.snapshot_id.empty()) {
      return std::nullopt;
    }
    return manifest;
  }

  static auto current_unix_ms() -> uint64_t {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count());
  }

  [[nodiscard]] auto graph_path(const fs::path &snapshot_dir) const -> std::string {
    return graph_file.empty() ? std::string() : (snapshot_dir / graph_file).string();
  }

  [[nodiscard]] auto data_path(const fs::path &snapshot_dir) const -> std::string {
    return data_file.empty() ? std::string() : (snapshot_dir / data_file).string();
  }

  [[nodiscard]] auto quant_path(const fs::path &snapshot_dir) const -> std::string {
    return quant_file.empty() ? std::string() : (snapshot_dir / quant_file).string();
  }

  [[nodiscard]] auto rocksdb_path(const fs::path &snapshot_dir) const -> fs::path {
    return rocksdb_dir.empty() ? fs::path{} : snapshot_dir / rocksdb_dir;
  }
};

}  // namespace alaya::recovery
