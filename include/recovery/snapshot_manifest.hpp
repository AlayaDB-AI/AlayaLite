// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <charconv>
#include <chrono>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

namespace alaya::recovery {

namespace fs = std::filesystem;

namespace detail {

inline auto parse_uint64_decimal(std::string_view value, uint64_t &out) -> bool {
  if (value.empty()) {
    return false;
  }
  uint64_t parsed = 0;
  const auto *begin = value.data();
  const auto *end = begin + value.size();
  auto [ptr, ec] = std::from_chars(begin, end, parsed, 10);
  if (ec != std::errc() || ptr != end) {
    return false;
  }
  out = parsed;
  return true;
}

inline auto parse_uint32_decimal(std::string_view value, uint32_t &out) -> bool {
  uint64_t parsed = 0;
  if (!parse_uint64_decimal(value, parsed) ||
      parsed > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }
  out = static_cast<uint32_t>(parsed);
  return true;
}

}  // namespace detail

/**
 * @brief Persistent metadata describing one published recovery snapshot.
 *
 * The manifest is stored as line-oriented key=value text inside each snapshot directory. Paths are
 * relative to the snapshot directory so the whole snapshot tree can be moved as one unit.
 */
struct SnapshotManifest {
  uint32_t format_version_{1};         ///< Manifest text format version.
  std::string snapshot_id_;            ///< Snapshot directory id stored in CURRENT.
  std::string reason_;                 ///< Human-readable snapshot trigger.
  uint64_t applied_through_op_id_{0};  ///< Highest operation id represented by this snapshot.
  uint64_t created_unix_ms_{0};        ///< Snapshot creation timestamp in Unix milliseconds.
  std::string graph_file_;             ///< Relative graph file path.
  std::string data_file_;              ///< Relative vector data file path.
  std::string quant_file_;             ///< Relative quantizer file path.
  std::string rocksdb_dir_;            ///< Relative RocksDB checkpoint directory path.

  [[nodiscard]] auto serialize() const -> std::string {
    std::ostringstream output;
    output << "format_version=" << format_version_ << '\n';
    output << "snapshot_id=" << snapshot_id_ << '\n';
    output << "reason=" << reason_ << '\n';
    output << "applied_through_op_id=" << applied_through_op_id_ << '\n';
    output << "created_unix_ms=" << created_unix_ms_ << '\n';
    output << "graph_file=" << graph_file_ << '\n';
    output << "data_file=" << data_file_ << '\n';
    output << "quant_file=" << quant_file_ << '\n';
    output << "rocksdb_dir=" << rocksdb_dir_ << '\n';
    return output.str();
  }

  /**
   * @brief Parses a text manifest into a SnapshotManifest object.
   *
   * Unknown lines are ignored for forward compatibility, while required snapshot identity must be
   * present. Numeric fields are converted from their persisted decimal representation.
   */
  static auto deserialize(const std::string &raw) -> std::optional<SnapshotManifest> {
    SnapshotManifest manifest;
    std::istringstream input(raw);
    std::string line;
    while (std::getline(input, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      auto delimiter = line.find('=');
      if (delimiter == std::string::npos) {
        continue;
      }
      auto key = line.substr(0, delimiter);
      auto value = line.substr(delimiter + 1);
      if (key == "format_version") {
        if (!detail::parse_uint32_decimal(value, manifest.format_version_)) {
          return std::nullopt;
        }
      } else if (key == "snapshot_id") {
        manifest.snapshot_id_ = value;
      } else if (key == "reason") {
        manifest.reason_ = value;
      } else if (key == "applied_through_op_id") {
        if (!detail::parse_uint64_decimal(value, manifest.applied_through_op_id_)) {
          return std::nullopt;
        }
      } else if (key == "created_unix_ms") {
        if (!detail::parse_uint64_decimal(value, manifest.created_unix_ms_)) {
          return std::nullopt;
        }
      } else if (key == "graph_file") {
        manifest.graph_file_ = value;
      } else if (key == "data_file") {
        manifest.data_file_ = value;
      } else if (key == "quant_file") {
        manifest.quant_file_ = value;
      } else if (key == "rocksdb_dir") {
        manifest.rocksdb_dir_ = value;
      }
    }
    if (manifest.snapshot_id_.empty()) {
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
    return graph_file_.empty() ? std::string() : (snapshot_dir / graph_file_).string();
  }

  [[nodiscard]] auto data_path(const fs::path &snapshot_dir) const -> std::string {
    return data_file_.empty() ? std::string() : (snapshot_dir / data_file_).string();
  }

  [[nodiscard]] auto quant_path(const fs::path &snapshot_dir) const -> std::string {
    return quant_file_.empty() ? std::string() : (snapshot_dir / quant_file_).string();
  }

  [[nodiscard]] auto rocksdb_path(const fs::path &snapshot_dir) const -> fs::path {
    return rocksdb_dir_.empty() ? fs::path{} : snapshot_dir / rocksdb_dir_;
  }
};

}  // namespace alaya::recovery
