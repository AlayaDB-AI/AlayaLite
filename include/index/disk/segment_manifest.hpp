// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "index/disk/types.hpp"
#include "utils/log.hpp"
#include "utils/metric_type.hpp"
#include "utils/platform_fs.hpp"

namespace alaya::disk {

inline constexpr uint64_t kManifestVersion = 1;

namespace detail {

inline auto is_ascii_ws(char c) -> bool { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

inline auto strip(std::string_view sv) -> std::string_view {
  while (!sv.empty() && is_ascii_ws(sv.front())) {
    sv.remove_prefix(1);
  }
  while (!sv.empty() && is_ascii_ws(sv.back())) {
    sv.remove_suffix(1);
  }
  return sv;
}

inline auto parse_uint64(std::string_view sv, std::string_view field) -> uint64_t {
  if (sv.empty()) {
    throw std::invalid_argument(std::string("manifest field ") + std::string(field) +
                                " has empty number");
  }
  uint64_t out = 0;
  for (char c : sv) {
    if (c < '0' || c > '9') {
      throw std::invalid_argument(std::string("manifest field ") + std::string(field) +
                                  " has non-digit value: " + std::string(sv));
    }
    auto d = static_cast<uint64_t>(c - '0');
    constexpr uint64_t kMax = static_cast<uint64_t>(-1);
    if (out > (kMax - d) / 10) {
      // Overflow is a malformed-input error from the user-data perspective;
      // unify under invalid_argument so callers only see two manifest
      // exception classes (invalid_argument for malformed user data,
      // runtime_error for syscall failures).
      throw std::invalid_argument(std::string("manifest field ") + std::string(field) +
                                  " overflows uint64: " + std::string(sv));
    }
    out = out * 10 + d;
  }
  return out;
}

// A basename for ids_file / vectors_file: non-empty, no path separator, no
// `..`/`.` traversal, no embedded NUL. The manifest values are joined to the
// segment directory at open time, so anything that could escape that join
// (slash, dot-dot) is rejected at parse time.
inline auto is_valid_basename(std::string_view sv) -> bool {
  if (sv.empty()) {
    return false;
  }
  if (sv == "." || sv == "..") {
    return false;
  }
  for (char c : sv) {
    if (c == '/' || c == '\0') {
      return false;
    }
  }
  return true;
}

inline auto is_valid_segment_id(std::string_view sv) -> bool {
  if (sv.size() != 12) {
    return false;
  }
  if (sv.substr(0, 4) != "seg_") {
    return false;
  }
  for (size_t i = 4; i < 12; ++i) {
    if (sv[i] < '0' || sv[i] > '9') {
      return false;
    }
  }
  return true;
}

inline auto metric_to_string(MetricType m) -> std::string_view {
  for (const auto &[k, v] : MetricMap::kStaticMap) {
    if (v == m) {
      return k;
    }
  }
  throw std::invalid_argument(
      "metric value cannot be serialized to manifest (NONE or unknown enum)");
}

struct KvPair {
  std::string key;
  std::string value;
};

// Manifests are tiny text files (a typical segment manifest is ~256 bytes;
// even with x_* extensions and a long segment list, well under 1 MiB). Cap
// the size both as a sanity check on the stored value and as a DoS guard
// against an attacker handing us a TB-scale file.
inline constexpr size_t kMaxManifestBytes = 1U << 20;  // 1 MiB

inline auto read_manifest_file(const std::filesystem::path &path) -> std::string {
  // platform_fs::read_regular_file_bounded preserves the historic exception
  // contract: malformed user data surfaces as std::invalid_argument (size
  // overflow, non-regular file, symlink), syscall failures as runtime_error.
  auto buf = ::alaya::platform::read_regular_file_bounded(path, kMaxManifestBytes);
  if (buf.size() >= 3 && static_cast<unsigned char>(buf[0]) == 0xEF &&
      static_cast<unsigned char>(buf[1]) == 0xBB && static_cast<unsigned char>(buf[2]) == 0xBF) {
    throw std::invalid_argument("manifest rejected: UTF-8 BOM at file start (encoding)");
  }
  return buf;
}

inline auto write_manifest_file(const std::filesystem::path &path, std::string_view contents)
    -> void {
  ::alaya::platform::write_all_fsync(path, contents.data(), contents.size());
}

inline auto parse_kv_lines(std::string_view body) -> std::vector<KvPair> {
  std::vector<KvPair> out;
  size_t pos = 0;
  while (pos <= body.size()) {
    size_t next = body.find('\n', pos);
    if (next == std::string_view::npos) {
      next = body.size();
    }
    auto line = body.substr(pos, next - pos);
    pos = next + 1;
    auto stripped = strip(line);
    if (stripped.empty()) {
      continue;
    }
    if (stripped.front() == '#') {
      continue;
    }
    size_t eq = stripped.find('=');
    if (eq == std::string_view::npos) {
      throw std::invalid_argument("manifest line missing '=' separator: " + std::string(line));
    }
    out.push_back(KvPair{
        std::string(strip(stripped.substr(0, eq))),
        std::string(strip(stripped.substr(eq + 1))),
    });
  }
  return out;
}

inline auto parse_segments_csv(std::string_view value) -> std::vector<std::string> {
  if (value.empty()) {
    throw std::invalid_argument("collection manifest segments= has empty value");
  }
  if (value.front() == ',' || value.back() == ',') {
    throw std::invalid_argument(
        "collection manifest segments= has leading or trailing comma; entries must be separated "
        "by single commas");
  }
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos < value.size()) {
    size_t comma = value.find(',', pos);
    if (comma == std::string_view::npos) {
      comma = value.size();
    }
    auto entry = value.substr(pos, comma - pos);
    if (entry.empty()) {
      throw std::invalid_argument(
          "collection manifest segments= has empty entry (consecutive commas)");
    }
    out.emplace_back(entry);
    pos = comma + 1;
  }
  return out;
}

}  // namespace detail

struct SegmentManifest {
  uint64_t version{kManifestVersion};
  std::string segment_id;
  DiskIndexType index_type{DiskIndexType::Flat};
  MetricType metric{MetricType::L2};
  uint64_t dim{0};
  uint64_t count{0};
  std::string ids_file{"ids.u64.bin"};
  std::string vectors_file{"vectors.f32.bin"};
  std::map<std::string, std::string> x_extras;

  auto save(const std::filesystem::path &path) const -> void {
    std::string out;
    out.reserve(256);
    out += "version=";
    out += std::to_string(version);
    out += "\nsegment_id=";
    out += segment_id;
    out += "\nindex_type=";
    out += index_type_to_string(index_type);
    out += "\nmetric=";
    out += detail::metric_to_string(metric);
    out += "\ndim=";
    out += std::to_string(dim);
    out += "\ncount=";
    out += std::to_string(count);
    out += "\nids_file=";
    out += ids_file;
    out += "\nvectors_file=";
    out += vectors_file;
    out += "\n";
    for (const auto &[k, v] : x_extras) {
      out += k;
      out += "=";
      out += v;
      out += "\n";
    }
    detail::write_manifest_file(path, out);
  }

  static auto load(const std::filesystem::path &path) -> SegmentManifest {
    auto body = detail::read_manifest_file(path);
    auto pairs = detail::parse_kv_lines(body);

    static constexpr std::array<std::string_view, 8> kRequired{
        "version",
        "segment_id",
        "index_type",
        "metric",
        "dim",
        "count",
        "ids_file",
        "vectors_file",
    };

    std::map<std::string, std::string> kv;
    for (auto &p : pairs) {
      if (kv.contains(p.key)) {
        throw std::invalid_argument("manifest contains duplicate key '" + p.key +
                                    "' (each scalar key must appear at most once)");
      }
      kv.emplace(std::move(p.key), std::move(p.value));
    }

    for (auto k : kRequired) {
      auto it = kv.find(std::string(k));
      if (it == kv.end()) {
        throw std::invalid_argument("manifest missing required key '" + std::string(k) +
                                    "' (missing key)");
      }
      // v1 Laser contract: a Laser segment publishes no engine-agnostic
      // `vectors.f32.bin`, so its manifest serialises `vectors_file=` with an
      // empty value. The presence check above still fires for missing keys;
      // only the empty-value gate is skipped for vectors_file. Other engines
      // continue to require a non-empty basename.
      if (it->second.empty() && k != "vectors_file") {
        throw std::invalid_argument("manifest required key '" + std::string(k) +
                                    "' has empty value (empty value)");
      }
    }

    SegmentManifest m;

    m.version = detail::parse_uint64(kv["version"], "version");
    if (m.version != kManifestVersion) {
      throw std::invalid_argument(
          "manifest version unsupported: read=" + std::to_string(m.version) +
          " max supported=" + std::to_string(kManifestVersion));
    }

    m.segment_id = kv["segment_id"];
    if (!detail::is_valid_segment_id(m.segment_id)) {
      throw std::invalid_argument("manifest segment_id malformed (must match ^seg_[0-9]{8}$): '" +
                                  m.segment_id + "'");
    }

    m.index_type = index_type_from_string(kv["index_type"]);

    m.metric = kMetricMap[kv["metric"]];
    if (m.metric == MetricType::NONE) {
      throw std::invalid_argument("manifest metric is unknown or NONE: '" + kv["metric"] +
                                  "' (must be one of L2, IP, COS)");
    }

    m.dim = detail::parse_uint64(kv["dim"], "dim");
    if (m.dim == 0) {
      throw std::invalid_argument("manifest dim must be > 0");
    }
    m.count = detail::parse_uint64(kv["count"], "count");
    if (m.count == 0) {
      throw std::invalid_argument("manifest count must be > 0");
    }

    m.ids_file = kv["ids_file"];
    m.vectors_file = kv["vectors_file"];
    if (!detail::is_valid_basename(m.ids_file)) {
      throw std::invalid_argument("manifest ids_file must be a basename (no '/', '..', NUL): '" +
                                  m.ids_file + "'");
    }
    // v1 Laser segments intentionally publish no engine-agnostic vectors file.
    // Non-empty values still must be safe basenames.
    if (!m.vectors_file.empty() && !detail::is_valid_basename(m.vectors_file)) {
      throw std::invalid_argument(
          "manifest vectors_file must be a basename (no '/', '..', NUL): '" + m.vectors_file + "'");
    }

    static const std::set<std::string_view> kKnown{kRequired.begin(), kRequired.end()};
    for (const auto &[k, v] : kv) {
      if (kKnown.contains(k)) {
        continue;
      }
      if (k.starts_with("x_")) {
        m.x_extras[k] = v;
        LOG_DEBUG("manifest retained unknown extension key '{}' (forward-compat namespace)", k);
      } else {
        throw std::invalid_argument("manifest contains unknown key '" + k +
                                    "' without 'x_' prefix (use 'x_*' for forward-compat keys)");
      }
    }

    return m;
  }
};

struct CollectionManifest {
  uint64_t version{kManifestVersion};
  uint64_t dim{0};
  MetricType metric{MetricType::L2};
  DiskIndexType index_type{DiskIndexType::Flat};
  uint64_t next_segment_id{1};
  std::vector<std::string> segment_ids;
  std::map<std::string, std::string> x_extras;

  auto save(const std::filesystem::path &path) const -> void {
    std::string out;
    out.reserve(512);
    out += "version=";
    out += std::to_string(version);
    out += "\ndim=";
    out += std::to_string(dim);
    out += "\nmetric=";
    out += detail::metric_to_string(metric);
    out += "\nindex_type=";
    out += index_type_to_string(index_type);
    out += "\nnext_segment_id=";
    out += std::to_string(next_segment_id);
    out += "\n";
    for (const auto &id : segment_ids) {
      out += "segment=";
      out += id;
      out += "\n";
    }
    for (const auto &[k, v] : x_extras) {
      out += k;
      out += "=";
      out += v;
      out += "\n";
    }
    detail::write_manifest_file(path, out);
  }

  static auto load(const std::filesystem::path &path) -> CollectionManifest {
    auto body = detail::read_manifest_file(path);
    auto pairs = detail::parse_kv_lines(body);

    static constexpr std::array<std::string_view, 5> kRequiredScalar{
        "version",
        "dim",
        "metric",
        "index_type",
        "next_segment_id",
    };
    static constexpr std::string_view kSegmentKey = "segment";
    static constexpr std::string_view kSegmentsKey = "segments";

    std::map<std::string, std::string> scalars;
    std::vector<std::string> segments_from_repeated;
    bool saw_repeated = false;
    bool saw_csv = false;
    std::string csv_value;

    for (auto &p : pairs) {
      if (p.key == kSegmentKey) {
        saw_repeated = true;
        segments_from_repeated.push_back(std::move(p.value));
      } else if (p.key == kSegmentsKey) {
        if (saw_csv) {
          throw std::invalid_argument(
              "collection manifest contains duplicate 'segments=' line "
              "(use exactly one segments= or repeated segment= entries)");
        }
        saw_csv = true;
        csv_value = std::move(p.value);
      } else {
        if (scalars.contains(p.key)) {
          throw std::invalid_argument("collection manifest contains duplicate scalar key '" +
                                      p.key + "' (each scalar key must appear at most once)");
        }
        scalars.emplace(std::move(p.key), std::move(p.value));
      }
    }

    if (saw_repeated && saw_csv) {
      throw std::invalid_argument(
          "collection manifest mixes 'segment=' and 'segments=' forms; pick exactly one");
    }

    for (auto k : kRequiredScalar) {
      auto it = scalars.find(std::string(k));
      if (it == scalars.end()) {
        throw std::invalid_argument("collection manifest missing required key '" + std::string(k) +
                                    "' (missing key)");
      }
      if (it->second.empty()) {
        throw std::invalid_argument("collection manifest required key '" + std::string(k) +
                                    "' has empty value (empty value)");
      }
    }

    CollectionManifest m;

    m.version = detail::parse_uint64(scalars["version"], "version");
    if (m.version != kManifestVersion) {
      throw std::invalid_argument(
          "collection manifest version unsupported: read=" + std::to_string(m.version) +
          " max supported=" + std::to_string(kManifestVersion));
    }
    m.dim = detail::parse_uint64(scalars["dim"], "dim");
    if (m.dim == 0) {
      throw std::invalid_argument("collection manifest dim must be > 0");
    }
    m.metric = kMetricMap[scalars["metric"]];
    if (m.metric == MetricType::NONE) {
      throw std::invalid_argument("collection manifest metric is unknown or NONE: '" +
                                  scalars["metric"] + "' (must be one of L2, IP, COS)");
    }
    m.index_type = index_type_from_string(scalars["index_type"]);
    m.next_segment_id = detail::parse_uint64(scalars["next_segment_id"], "next_segment_id");

    if (saw_csv) {
      m.segment_ids = detail::parse_segments_csv(csv_value);
    } else {
      m.segment_ids = std::move(segments_from_repeated);
    }
    for (const auto &id : m.segment_ids) {
      if (!detail::is_valid_segment_id(id)) {
        throw std::invalid_argument(
            "collection manifest segment id malformed (must match ^seg_[0-9]{8}$): '" + id + "'");
      }
    }

    static const std::set<std::string_view> kKnown{kRequiredScalar.begin(), kRequiredScalar.end()};
    for (const auto &[k, v] : scalars) {
      if (kKnown.contains(k)) {
        continue;
      }
      if (k.starts_with("x_")) {
        m.x_extras[k] = v;
        LOG_DEBUG("collection manifest retained unknown extension key '{}'", k);
      } else {
        throw std::invalid_argument("collection manifest contains unknown key '" + k +
                                    "' without 'x_' prefix");
      }
    }

    return m;
  }
};

}  // namespace alaya::disk
