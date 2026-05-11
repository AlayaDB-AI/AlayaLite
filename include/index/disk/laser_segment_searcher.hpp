// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "storage/mmap_file.hpp"
#include "utils/metric_type.hpp"

#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0
  #include "index/graph/laser/qg/qg.hpp"
#endif

namespace alaya::disk {

#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0

[[nodiscard]] constexpr auto engine_supported_v1(DiskIndexType type) noexcept -> bool;

namespace detail {

inline auto laser_metric_name(MetricType metric) -> std::string {
  switch (metric) {
    case MetricType::L2:
      return "l2";
    case MetricType::IP:
      return "ip";
    case MetricType::COS:
      return "cos";
    case MetricType::NONE:
      return "none";
  }
  return "unknown";
}

inline auto laser_required_extra(const SegmentManifest &manifest,
                                 const char *key,
                                 const std::filesystem::path &seg_dir) -> const std::string & {
  const auto it = manifest.x_extras.find(key);
  if (it == manifest.x_extras.end() || it->second.empty()) {
    throw std::runtime_error("LaserSegmentSearcher: " + std::string(key) +
                             " missing in manifest for segment " + seg_dir.string());
  }
  return it->second;
}

inline auto laser_parse_u32_extra(const SegmentManifest &manifest,
                                  const char *key,
                                  const std::filesystem::path &seg_dir) -> uint32_t {
  const auto &value = laser_required_extra(manifest, key, seg_dir);
  const auto parsed = parse_uint64(value, key);
  if (parsed == 0 || parsed > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error("LaserSegmentSearcher: " + std::string(key) +
                             " must be in [1, uint32_max] for segment " + seg_dir.string() +
                             " (got " + value + ")");
  }
  return static_cast<uint32_t>(parsed);
}

inline auto laser_parse_float_extra_default(const SegmentManifest &manifest,
                                            const char *key,
                                            float fallback,
                                            const std::filesystem::path &seg_dir) -> float {
  const auto it = manifest.x_extras.find(key);
  if (it == manifest.x_extras.end() || it->second.empty()) {
    return fallback;
  }
  try {
    size_t pos = 0;
    const float parsed = std::stof(it->second, &pos);
    if (pos != it->second.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return parsed;
  } catch (const std::exception &e) {
    throw std::runtime_error("LaserSegmentSearcher: " + std::string(key) +
                             " is not a valid float for segment " + seg_dir.string() + ": " +
                             e.what());
  }
}

inline auto laser_compute_expected_ids_bytes(uint64_t count, const std::filesystem::path &seg_dir)
    -> uint64_t {
  uint64_t bytes = 0;
  if (__builtin_mul_overflow(count, static_cast<uint64_t>(sizeof(uint64_t)), &bytes)) {
    throw std::runtime_error("LaserSegmentSearcher: manifest count×8 exceeds uint64 range in " +
                             seg_dir.string());
  }
  return bytes;
}

inline auto laser_read_index_metadata_count(const std::filesystem::path &index_path,
                                            const std::filesystem::path &seg_dir) -> uint64_t {
  #ifdef _WIN32
  (void)index_path;
  (void)seg_dir;
  throw std::runtime_error("LaserSegmentSearcher: disk_laser not implemented in v1 on Windows");
  #else
  int fd = ::open(index_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    const int saved = errno;
    throw std::runtime_error("LaserSegmentSearcher: index metadata open failed for " +
                             index_path.string() + " in segment " + seg_dir.string() + ": " +
                             std::strerror(saved));
  }

  struct ::stat st{};
  if (::fstat(fd, &st) != 0) {
    const int saved = errno;
    ::close(fd);
    throw std::runtime_error("LaserSegmentSearcher: index metadata fstat failed for " +
                             index_path.string() + " in segment " + seg_dir.string() + ": " +
                             std::strerror(saved));
  }
  if (!S_ISREG(st.st_mode)) {
    ::close(fd);
    throw std::runtime_error("LaserSegmentSearcher: index metadata path is not a regular file: " +
                             index_path.string() + " in segment " + seg_dir.string());
  }

  uint64_t count = 0;
  auto *dst = reinterpret_cast<char *>(&count);
  size_t total = 0;
  while (total < sizeof(count)) {
    const ssize_t n = ::read(fd, dst + total, sizeof(count) - total);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      const int saved = errno;
      ::close(fd);
      throw std::runtime_error("LaserSegmentSearcher: index metadata read failed for " +
                               index_path.string() + " in segment " + seg_dir.string() + ": " +
                               std::strerror(saved));
    }
    if (n == 0) {
      break;
    }
    total += static_cast<size_t>(n);
  }
  ::close(fd);
  if (total != sizeof(count)) {
    throw std::runtime_error("LaserSegmentSearcher: index metadata short read for " +
                             index_path.string() + " in segment " + seg_dir.string() + " (read " +
                             std::to_string(total) + " of " + std::to_string(sizeof(count)) +
                             " bytes)");
  }
  return count;
  #endif
}

inline auto laser_validate_manifest_artifact(const SegmentManifest &manifest,
                                             const char *key,
                                             const std::filesystem::path &seg_dir,
                                             bool required) -> void {
  const auto it = manifest.x_extras.find(key);
  if (it == manifest.x_extras.end()) {
    if (required) {
      throw std::runtime_error("LaserSegmentSearcher: " + std::string(key) +
                               " missing in manifest for segment " + seg_dir.string());
    }
    return;
  }
  if (it->second.empty()) {
    throw std::runtime_error("LaserSegmentSearcher: " + std::string(key) +
                             " is empty in manifest for segment " + seg_dir.string());
  }
  if (!is_valid_basename(it->second)) {
    throw std::runtime_error("LaserSegmentSearcher: " + std::string(key) +
                             " must be a basename (no '/', '..', NUL): '" + it->second +
                             "' in segment " + seg_dir.string());
  }

  #ifdef _WIN32
  throw std::runtime_error("LaserSegmentSearcher: disk_laser not implemented in v1 on Windows");
  #else
  const auto path = seg_dir / it->second;
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    const int saved = errno;
    throw std::runtime_error("LaserSegmentSearcher: native artifact open failed for " +
                             path.string() + " (" + key + ") in segment " + seg_dir.string() +
                             ": " + std::strerror(saved));
  }
  struct ::stat st{};
  if (::fstat(fd, &st) != 0) {
    const int saved = errno;
    ::close(fd);
    throw std::runtime_error("LaserSegmentSearcher: native artifact fstat failed for " +
                             path.string() + " (" + key + ") in segment " + seg_dir.string() +
                             ": " + std::strerror(saved));
  }
  ::close(fd);
  if (!S_ISREG(st.st_mode) || st.st_size <= 0) {
    throw std::runtime_error(
        "LaserSegmentSearcher: native artifact is missing, empty, or not "
        "regular for " +
        path.string() + " (" + key + ") in segment " + seg_dir.string());
  }
  #endif
}

}  // namespace detail

static_assert(std::endian::native == std::endian::little,
              "LaserSegmentSearcher v1 supports only little-endian hosts");

class LaserSegmentSearcher : public SegmentSearcher {
 public:
  explicit LaserSegmentSearcher(const std::filesystem::path &seg_dir)
      : manifest_(SegmentManifest::load(seg_dir / "manifest.txt")) {
    if (manifest_.index_type != DiskIndexType::Laser) {
      throw std::runtime_error(
          "LaserSegmentSearcher: manifest index_type is not disk_laser (got '" +
          std::string(index_type_to_string(manifest_.index_type)) + "') in segment " +
          seg_dir.string());
    }
    if (manifest_.dim == 0 || manifest_.count == 0) {
      throw std::runtime_error("LaserSegmentSearcher: manifest dim or count is zero in segment " +
                               seg_dir.string());
    }
    if (manifest_.dim > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::runtime_error("LaserSegmentSearcher: manifest dim exceeds uint32 (" +
                               std::to_string(manifest_.dim) + ") in segment " + seg_dir.string());
    }
    if (manifest_.count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      throw std::runtime_error("LaserSegmentSearcher: manifest count exceeds uint32 (" +
                               std::to_string(manifest_.count) + ") in segment " +
                               seg_dir.string());
    }
    if (manifest_.metric != MetricType::L2) {
      throw std::runtime_error("LaserSegmentSearcher: metric '" +
                               detail::laser_metric_name(manifest_.metric) +
                               "' not implemented in v1 (disk_laser adapter v1 supports L2 only) "
                               "in segment " +
                               seg_dir.string());
    }
    if (!engine_supported_v1(DiskIndexType::Laser)) {
      throw std::runtime_error("LaserSegmentSearcher: engine 'disk_laser' not implemented in v1");
    }

    const auto &prefix =
        detail::laser_required_extra(manifest_, "x_laser_filename_prefix", seg_dir);
    if (!detail::is_valid_basename(prefix)) {
      throw std::runtime_error(
          "LaserSegmentSearcher: x_laser_filename_prefix must be a basename (no '/', '..', NUL): "
          "'" +
          prefix + "' in segment " + seg_dir.string());
    }
    const uint32_t r = detail::laser_parse_u32_extra(manifest_, "x_R", seg_dir);
    const uint32_t main_dim = detail::laser_parse_u32_extra(manifest_, "x_main_dim", seg_dir);
    const auto &manifest_index_file =
        detail::laser_required_extra(manifest_, "x_laser_index_file", seg_dir);
    if (!detail::is_valid_basename(manifest_index_file)) {
      throw std::runtime_error(
          "LaserSegmentSearcher: x_laser_index_file must be a basename (no '/', '..', NUL): '" +
          manifest_index_file + "' in segment " + seg_dir.string());
    }

    const std::string computed_index_file =
        prefix + "_R" + std::to_string(r) + "_MD" + std::to_string(main_dim) + ".index";
    if (computed_index_file != manifest_index_file) {
      throw std::runtime_error("LaserSegmentSearcher: x_laser_index_file mismatch in segment " +
                               seg_dir.string() + " (manifest '" + manifest_index_file +
                               "', computed '" + computed_index_file + "')");
    }

    const auto index_path = seg_dir / computed_index_file;
    const uint64_t index_count = detail::laser_read_index_metadata_count(index_path, seg_dir);
    if (index_count != manifest_.count) {
      throw std::runtime_error(
          "LaserSegmentSearcher: manifest count (" + std::to_string(manifest_.count) +
          ") disagrees with LASER index metadata count (" + std::to_string(index_count) + ") in " +
          index_path.string() + " for segment " + seg_dir.string());
    }

    detail::laser_validate_manifest_artifact(manifest_, "x_laser_index_file", seg_dir, true);
    detail::laser_validate_manifest_artifact(manifest_, "x_laser_rotator_file", seg_dir, true);
    detail::laser_validate_manifest_artifact(manifest_, "x_laser_cache_ids_file", seg_dir, true);
    detail::laser_validate_manifest_artifact(manifest_, "x_laser_cache_nodes_file", seg_dir, true);
    detail::laser_validate_manifest_artifact(manifest_, "x_laser_medoids_file", seg_dir, false);
    detail::laser_validate_manifest_artifact(manifest_,
                                             "x_laser_medoids_indices_file",
                                             seg_dir,
                                             false);
    detail::laser_validate_manifest_artifact(manifest_, "x_laser_pca_file", seg_dir, false);

    try {
      quantized_graph_ =
          std::make_unique<alaya::laser::QuantizedGraph>(static_cast<size_t>(manifest_.count),
                                                         static_cast<size_t>(r),
                                                         static_cast<size_t>(main_dim),
                                                         static_cast<size_t>(manifest_.dim),
                                                         0,
                                                         std::string{});
    } catch (const std::exception &e) {
      throw std::runtime_error("LaserSegmentSearcher: QuantizedGraph construction failed for " +
                               seg_dir.string() + ": " + e.what());
    }

    const auto index_prefix = (seg_dir / prefix).string();
    const float dram_budget_gb =
        detail::laser_parse_float_extra_default(manifest_,
                                                "x_laser_search_dram_budget_gb",
                                                0.5F,
                                                seg_dir);
    try {
      quantized_graph_->load_disk_index(index_prefix.c_str(), dram_budget_gb);
    } catch (const std::exception &e) {
      throw std::runtime_error("LaserSegmentSearcher: QuantizedGraph load failed for " +
                               seg_dir.string() + ": " + e.what());
    }

    ids_mmap_ = alaya::storage::MMapFile(seg_dir / manifest_.ids_file);
    const uint64_t expected_ids_bytes =
        detail::laser_compute_expected_ids_bytes(manifest_.count, seg_dir);
    if (ids_mmap_.size() != expected_ids_bytes) {
      throw std::runtime_error(
          "LaserSegmentSearcher: ids file size mismatch in segment " + seg_dir.string() +
          " — expected " + std::to_string(expected_ids_bytes) + " (count×8) but got " +
          std::to_string(ids_mmap_.size()) + " for " + (seg_dir / manifest_.ids_file).string());
    }
    ids_view_ = ids_mmap_.as<uint64_t>();
    last_set_params_ = LastSetParams{0, 0, 0};
  }

  LaserSegmentSearcher(const LaserSegmentSearcher &) = delete;
  auto operator=(const LaserSegmentSearcher &) -> LaserSegmentSearcher & = delete;
  LaserSegmentSearcher(LaserSegmentSearcher &&) = delete;
  auto operator=(LaserSegmentSearcher &&) -> LaserSegmentSearcher & = delete;
  ~LaserSegmentSearcher() override = default;

  auto search(const float *query, const DiskSearchOptions &opts) const
      -> std::vector<DiskSearchHit> override {
    if (opts.top_k == 0) {
      throw std::invalid_argument("LaserSegmentSearcher: top_k must be > 0");
    }
    if (query == nullptr) {
      throw std::invalid_argument("LaserSegmentSearcher: query must not be null");
    }
    if (opts.beam_width > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("LaserSegmentSearcher: beam_width exceeds int max");
    }

    // Argument validation stays pre-lock per spec; the rest serialises so
    // QuantizedGraph::set_params' destroy + rebuild cannot race in-flight searches.
    const std::lock_guard<std::mutex> lock(search_mutex_);

    const auto effective_top_k = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(opts.top_k), manifest_.count));
    const LastSetParams requested{
        static_cast<size_t>(std::max(opts.ef, effective_top_k)),
        1,
        static_cast<int>(opts.beam_width),
    };
    if (requested != last_set_params_) {
      apply_set_params(requested);
      last_set_params_ = requested;
    }

    std::vector<uint32_t> pid_buf;
    pid_buf.resize(effective_top_k);
    quantized_graph_->search(query, effective_top_k, pid_buf.data());

    std::vector<DiskSearchHit> out;
    out.reserve(effective_top_k);
    for (uint32_t pid : pid_buf) {
      if (pid >= manifest_.count) {
        throw std::runtime_error("LaserSegmentSearcher: QuantizedGraph returned PID " +
                                 std::to_string(pid) + " outside segment count " +
                                 std::to_string(manifest_.count));
      }
      out.push_back(DiskSearchHit{ids_view_[pid], std::numeric_limits<float>::quiet_NaN()});
    }
    return out;
  }
  auto size() const -> uint64_t override { return manifest_.count; }
  auto dim() const -> uint32_t override { return static_cast<uint32_t>(manifest_.dim); }
  auto type() const -> DiskIndexType override { return DiskIndexType::Laser; }

  // Test-only observer for the cached set_params triple (D5). Counts how many
  // times `QuantizedGraph::set_params` has been forwarded; tests rely on this
  // to assert the cache-and-skip behaviour without reaching into private state.
  auto set_params_call_count() const noexcept -> uint64_t {
    return set_params_call_count_.load(std::memory_order_relaxed);
  }

 private:
  struct LastSetParams {
    size_t ef_search;
    size_t num_threads;
    int beam_width;

    friend auto operator==(const LastSetParams &, const LastSetParams &) -> bool = default;
  };

  void apply_set_params(const LastSetParams &requested) const {
    quantized_graph_->set_params(requested.ef_search, requested.num_threads, requested.beam_width);
    set_params_call_count_.fetch_add(1, std::memory_order_relaxed);
  }

  // Declaration order matters: quantized_graph_ is destroyed before ids_mmap_.
  SegmentManifest manifest_;
  alaya::storage::MMapFile ids_mmap_;
  std::unique_ptr<alaya::laser::QuantizedGraph> quantized_graph_;
  const uint64_t *ids_view_ = nullptr;
  mutable LastSetParams last_set_params_{0, 0, 0};
  mutable std::atomic<uint64_t> set_params_call_count_{0};
  mutable std::mutex search_mutex_;
};

#else

class LaserSegmentSearcher : public SegmentSearcher {
 public:
  [[noreturn]] explicit LaserSegmentSearcher(const std::filesystem::path & /*seg_dir*/) {
    throw std::runtime_error("LaserSegmentSearcher: engine 'disk_laser' not implemented in v1");
  }

  [[noreturn]] auto search(const float * /*query*/, const DiskSearchOptions & /*opts*/) const
      -> std::vector<DiskSearchHit> override {
    throw std::runtime_error("LaserSegmentSearcher: engine 'disk_laser' not implemented in v1");
  }
  auto size() const -> uint64_t override { return 0; }
  auto dim() const -> uint32_t override { return 0; }
  auto type() const -> DiskIndexType override { return DiskIndexType::Laser; }
};

#endif

}  // namespace alaya::disk
