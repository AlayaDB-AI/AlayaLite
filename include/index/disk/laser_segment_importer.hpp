/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <unistd.h>

  #include <cerrno>
  #include <chrono>
  #include <cstddef>
  #include <cstring>
  #include <limits>
  #include <sstream>
  #include <stdexcept>
  #include <string>
  #include <utility>
  #include <vector>

  #include "index/disk/disk_flat_builder.hpp"
#endif

#include <filesystem>  // NOLINT(build/c++17)

#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"

namespace alaya::disk {

#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0

[[nodiscard]] constexpr auto engine_supported_v1(DiskIndexType type) noexcept -> bool;

struct LaserSegmentImportParams {
  uint32_t R = 64;
  uint32_t main_dim = 0;
  uint32_t default_ef = 200;
  uint32_t default_beam_width = 4;
  float search_dram_budget_gb = 0.5F;
  bool copy_files = true;
};

class LaserSegmentImporter {
 public:
  LaserSegmentImporter(uint32_t dim, MetricType metric, LaserSegmentImportParams params);

  auto import_from(const std::filesystem::path &src_dir,
                   const uint64_t *labels,
                   uint64_t n,
                   const std::filesystem::path &seg_dir) -> SegmentManifest;

 private:
  uint32_t dim_;
  MetricType metric_;
  LaserSegmentImportParams params_;
};

namespace laser_importer_detail {

struct FileDescriptor {
  explicit FileDescriptor(int fd_in) : fd(fd_in) {}
  ~FileDescriptor() {
    if (fd >= 0) {
      (void)::close(fd);
    }
  }
  FileDescriptor(const FileDescriptor &) = delete;
  auto operator=(const FileDescriptor &) -> FileDescriptor & = delete;
  FileDescriptor(FileDescriptor &&other) noexcept : fd(std::exchange(other.fd, -1)) {}
  auto operator=(FileDescriptor &&other) noexcept -> FileDescriptor & {
    if (this != &other) {
      if (fd >= 0) {
        (void)::close(fd);
      }
      fd = std::exchange(other.fd, -1);
    }
    return *this;
  }

  int fd;
};

struct Artifact {
  std::filesystem::path src;
  std::string manifest_key;
  bool present{true};
};

inline auto is_power_of_two(uint32_t v) -> bool { return v != 0 && (v & (v - 1U)) == 0; }

inline auto metric_name(MetricType metric) -> std::string {
  if (metric == MetricType::IP) {
    return "ip";
  }
  if (metric == MetricType::COS) {
    return "cos";
  }
  if (metric == MetricType::L2) {
    return "l2";
  }
  return "unknown";
}

inline auto format_float(float value) -> std::string {
  std::ostringstream out;
  out << value;
  return out.str();
}

inline auto require_parent_dir(const std::filesystem::path &parent,
                               const std::filesystem::path &seg_dir) -> void {
  if (parent.empty()) {
    throw std::invalid_argument("LaserSegmentImporter: seg_dir must have a parent: " +
                                seg_dir.string());
  }
  std::error_code ec;
  const bool is_dir = std::filesystem::is_directory(parent, ec);
  if (ec || !is_dir) {
    throw std::runtime_error("LaserSegmentImporter: parent directory does not exist: " +
                             parent.string());
  }
}

inline auto reject_existing_segment_dir(const std::filesystem::path &seg_dir) -> void {
  struct ::stat st{};
  if (::lstat(seg_dir.c_str(), &st) == 0) {
    throw std::runtime_error("LaserSegmentImporter: target segment already exists: " +
                             seg_dir.string());
  }
  if (errno != ENOENT) {
    throw std::runtime_error("LaserSegmentImporter: target segment stat failed: " +
                             seg_dir.string() + ": " + std::strerror(errno));
  }
}

inline auto require_regular_nonempty(const std::filesystem::path &path) -> void {
  struct ::stat st{};
  if (::stat(path.c_str(), &st) != 0) {
    throw std::runtime_error("LaserSegmentImporter: required native artifact missing: " +
                             path.string() + ": " + std::strerror(errno));
  }
  if (!S_ISREG(st.st_mode)) {
    throw std::runtime_error(
        "LaserSegmentImporter: required native artifact is not a regular file: " + path.string());
  }
  if (st.st_size <= 0) {
    throw std::runtime_error(
        "LaserSegmentImporter: required native artifact is empty or invalid size: " +
        path.string());
  }
}

inline auto optional_exists(const std::filesystem::path &path) -> bool {
  std::error_code ec;
  const bool exists = std::filesystem::exists(path, ec);
  if (ec) {
    throw std::runtime_error("LaserSegmentImporter: optional artifact stat failed: " +
                             path.string() + ": " + ec.message());
  }
  return exists;
}

inline auto read_index_count(const std::filesystem::path &index_path) -> uint64_t {
  FileDescriptor fd(::open(index_path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC));
  if (fd.fd < 0) {
    throw std::runtime_error("LaserSegmentImporter: open index metadata failed: " +
                             index_path.string() + ": " + std::strerror(errno));
  }

  uint64_t count = 0;
  auto *dst = reinterpret_cast<char *>(&count);
  size_t total = 0;
  while (total < sizeof(count)) {
    const ssize_t n = ::read(fd.fd, dst + total, sizeof(count) - total);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error("LaserSegmentImporter: read index metadata failed: " +
                               index_path.string() + ": " + std::strerror(errno));
    }
    if (n == 0) {
      break;
    }
    total += static_cast<size_t>(n);
  }
  if (total != sizeof(count)) {
    throw std::runtime_error("LaserSegmentImporter: short read of index metadata: " +
                             index_path.string());
  }
  return count;
}

inline auto copy_or_link_artifact(const std::filesystem::path &src,
                                  const std::filesystem::path &dst,
                                  bool copy_files) -> void {
  std::error_code ec;
  if (copy_files) {
    std::filesystem::copy_file(src, dst, std::filesystem::copy_options::none, ec);
  } else {
    std::filesystem::create_hard_link(src, dst, ec);
  }
  if (ec) {
    throw std::runtime_error("LaserSegmentImporter: artifact publish failed: " + src.string() +
                             " -> " + dst.string() + ": " + ec.message());
  }
}

inline auto make_tmp_dir(const std::filesystem::path &parent, const std::string &seg_basename)
    -> std::filesystem::path {
  const auto pid = static_cast<int64_t>(::getpid());
  const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::string tmp_name =
      ".tmp_" + seg_basename + "_" + std::to_string(pid) + "_" + std::to_string(ts);
  const auto tmp_dir = parent / tmp_name;

  std::error_code ec;
  if (std::filesystem::exists(tmp_dir, ec)) {
    throw std::runtime_error(
        "LaserSegmentImporter: stale tmp directory already exists (refusing to overwrite): " +
        tmp_dir.string());
  }
  if (ec) {
    throw std::runtime_error("LaserSegmentImporter: tmp directory stat failed: " +
                             tmp_dir.string() + ": " + ec.message());
  }
  const bool created = std::filesystem::create_directory(tmp_dir, ec);
  if (ec || !created) {
    throw std::runtime_error("LaserSegmentImporter: mkdir tmp failed: " + tmp_dir.string() +
                             (ec ? (": " + ec.message()) : ""));
  }
  return tmp_dir;
}

}  // namespace laser_importer_detail

inline LaserSegmentImporter::LaserSegmentImporter(uint32_t dim,
                                                  MetricType metric,
                                                  LaserSegmentImportParams params)
    : dim_(dim), metric_(metric), params_(params) {
  if (dim == 0) {
    throw std::runtime_error("LaserSegmentImporter: dim must be > 0");
  }
  if (!laser_importer_detail::is_power_of_two(dim)) {
    throw std::runtime_error("LaserSegmentImporter: dim " + std::to_string(dim) +
                             " must be a power of two");
  }
  if (dim < 128) {
    throw std::runtime_error("LaserSegmentImporter: dim " + std::to_string(dim) +
                             " is below the v1 LASER floor of 128");
  }
  if (params_.main_dim != 0 && params_.main_dim != dim_) {
    throw std::runtime_error("LaserSegmentImporter: main_dim must be 0 or equal dim (main_dim=" +
                             std::to_string(params_.main_dim) + ", dim=" + std::to_string(dim_) +
                             ")");
  }
}

inline auto LaserSegmentImporter::import_from(const std::filesystem::path &src_dir,
                                              const uint64_t *labels,
                                              uint64_t n,
                                              const std::filesystem::path &seg_dir)
    -> SegmentManifest {
  const auto parent = seg_dir.parent_path();
  laser_importer_detail::require_parent_dir(parent, seg_dir);

  const std::string seg_basename = seg_dir.filename().string();
  if (!detail::is_valid_segment_id(seg_basename)) {
    throw std::invalid_argument(
        "LaserSegmentImporter: seg_dir basename must match ^seg_[0-9]{8}$: '" + seg_basename + "'");
  }
  laser_importer_detail::reject_existing_segment_dir(seg_dir);

  if (metric_ != MetricType::L2) {
    const auto m = laser_importer_detail::metric_name(metric_);
    throw std::runtime_error("LaserSegmentImporter: metric '" + m +
                             "' not implemented in v1 (laser adapter v1 supports L2 only)");
  }
  if (!engine_supported_v1(DiskIndexType::Laser)) {
    throw std::runtime_error("LaserSegmentImporter: engine 'disk_laser' not implemented in v1");
  }
  if (n == 0) {
    throw std::invalid_argument("LaserSegmentImporter: n must be >= 1");
  }
  if (labels == nullptr) {
    throw std::invalid_argument("LaserSegmentImporter: labels must be non-null");
  }
  if (n > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(uint64_t))) {
    throw std::invalid_argument("LaserSegmentImporter: n * sizeof(uint64_t) overflows size_t");
  }

  const uint32_t main_dim_resolved = params_.main_dim == 0 ? dim_ : params_.main_dim;
  const std::string prefix = "dsqg_" + seg_basename;
  const std::string index_file = prefix + "_R" + std::to_string(params_.R) + "_MD" +
                                 std::to_string(main_dim_resolved) + ".index";
  const auto index_src = src_dir / index_file;

  std::vector<laser_importer_detail::Artifact> required{
      {index_src, "x_laser_index_file", true},
      {std::filesystem::path(index_src.string() + "_rotator"), "x_laser_rotator_file", true},
      {std::filesystem::path(index_src.string() + "_cache_ids"), "x_laser_cache_ids_file", true},
      {std::filesystem::path(index_src.string() + "_cache_nodes"),
       "x_laser_cache_nodes_file",
       true},
  };
  std::vector<laser_importer_detail::Artifact> optional{
      {src_dir / (prefix + "_medoids"), "x_laser_medoids_file", false},
      {src_dir / (prefix + "_medoids_indices"), "x_laser_medoids_indices_file", false},
      {src_dir / (prefix + "_pca.bin"), "x_laser_pca_file", false},
  };

  for (const auto &artifact : required) {
    laser_importer_detail::require_regular_nonempty(artifact.src);
  }
  for (auto &artifact : optional) {
    artifact.present = laser_importer_detail::optional_exists(artifact.src);
  }

  const uint64_t index_count = laser_importer_detail::read_index_count(index_src);
  if (index_count != n) {
    throw std::runtime_error("LaserSegmentImporter: index count mismatch for " +
                             index_src.string() + " (index num_points=" +
                             std::to_string(index_count) + ", import n=" + std::to_string(n) + ")");
  }

  const auto tmp_dir = laser_importer_detail::make_tmp_dir(parent, seg_basename);
  detail::TmpDirGuard guard(tmp_dir);

  for (const auto &artifact : required) {
    laser_importer_detail::copy_or_link_artifact(artifact.src,
                                                 tmp_dir / artifact.src.filename(),
                                                 params_.copy_files);
  }
  for (const auto &artifact : optional) {
    if (!artifact.present) {
      continue;
    }
    laser_importer_detail::copy_or_link_artifact(artifact.src,
                                                 tmp_dir / artifact.src.filename(),
                                                 params_.copy_files);
  }

  detail::write_all_fsync(tmp_dir / "ids.u64.bin",
                          labels,
                          static_cast<size_t>(n) * sizeof(uint64_t));

  SegmentManifest manifest{};
  manifest.version = kManifestVersion;
  manifest.segment_id = seg_basename;
  manifest.index_type = DiskIndexType::Laser;
  manifest.metric = MetricType::L2;
  manifest.dim = dim_;
  manifest.count = n;
  manifest.ids_file = "ids.u64.bin";
  manifest.vectors_file = "";
  manifest.x_extras["x_laser_filename_prefix"] = prefix;
  for (const auto &artifact : required) {
    manifest.x_extras[artifact.manifest_key] = artifact.src.filename().string();
  }
  for (const auto &artifact : optional) {
    if (artifact.present) {
      manifest.x_extras[artifact.manifest_key] = artifact.src.filename().string();
    }
  }
  manifest.x_extras["x_R"] = std::to_string(params_.R);
  manifest.x_extras["x_main_dim"] = std::to_string(main_dim_resolved);
  manifest.x_extras["x_default_ef"] = std::to_string(params_.default_ef);
  manifest.x_extras["x_default_beam_width"] = std::to_string(params_.default_beam_width);
  manifest.x_extras["x_laser_native_format_version"] = "1";
  manifest.x_extras["x_platform_requires"] = "linux+libaio";
  manifest.x_extras["x_laser_search_dram_budget_gb"] =
      laser_importer_detail::format_float(params_.search_dram_budget_gb);
  manifest.x_extras["x_laser_distance_field_supported"] = "false";
  manifest.save(tmp_dir / "manifest.txt");

  detail::fsync_dir(tmp_dir);
  detail::rename_no_replace(tmp_dir, seg_dir);
  guard.disarm();

  try {
    detail::fsync_dir(parent);
  } catch (const std::exception &e) {
    LOG_WARN("LaserSegmentImporter: post-rename parent fsync failed (durability only): {}",
             e.what());
  }

  return manifest;
}

#else

struct LaserSegmentImportParams {
  uint32_t R = 64;
  uint32_t main_dim = 0;
  uint32_t default_ef = 200;
  uint32_t default_beam_width = 4;
  float search_dram_budget_gb = 0.5F;
  bool copy_files = true;
};

class LaserSegmentImporter {
 public:
  [[noreturn]] LaserSegmentImporter(uint32_t /*dim*/,
                                    MetricType /*metric*/,
                                    LaserSegmentImportParams /*params*/) {
    throw std::runtime_error("LaserSegmentImporter: engine 'disk_laser' not implemented in v1");
  }

  [[noreturn]] auto import_from(const std::filesystem::path & /*src_dir*/,
                                const uint64_t * /*labels*/,
                                uint64_t /*n*/,
                                const std::filesystem::path & /*seg_dir*/) -> SegmentManifest {
    throw std::runtime_error("LaserSegmentImporter: engine 'disk_laser' not implemented in v1");
  }
};

#endif

}  // namespace alaya::disk
