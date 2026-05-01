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

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__linux__)
  #include <linux/fs.h>     // RENAME_NOREPLACE
  #include <sys/syscall.h>  // SYS_renameat2
#endif

#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "utils/log.hpp"
#include "utils/metric_type.hpp"

namespace alaya::disk {

static_assert(std::endian::native == std::endian::little,
              "DiskCollection v1 supports only little-endian hosts");

namespace detail {

// IEEE-754 binary32 inspection. We avoid std::isfinite / std::isnan / std::isinf
// because the project builds with -Ofast (implies -ffast-math), under which
// the compiler assumes no NaN/Inf can occur and constant-folds those checks
// to true / false. Bit-pattern inspection survives because it does not go
// through the FPU.
inline auto float_bits(float v) -> uint32_t {
  uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

inline auto is_nan_f32(float v) -> bool {
  const uint32_t b = float_bits(v);
  return (b & 0x7F800000U) == 0x7F800000U && (b & 0x007FFFFFU) != 0U;
}

inline auto is_inf_f32(float v) -> bool {
  const uint32_t b = float_bits(v);
  return (b & 0x7F800000U) == 0x7F800000U && (b & 0x007FFFFFU) == 0U;
}

inline auto is_finite_f32(float v) -> bool { return (float_bits(v) & 0x7F800000U) != 0x7F800000U; }

inline auto is_neg_f32(float v) -> bool { return (float_bits(v) & 0x80000000U) != 0U; }

inline auto double_bits(double v) -> uint64_t {
  uint64_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

inline auto is_finite_f64(double v) -> bool {
  return (double_bits(v) & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

inline auto write_all_fsync(const std::filesystem::path &path, const void *data, size_t bytes)
    -> void {
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) {
    throw std::runtime_error("disk_flat_builder open failed: " + path.string() + ": " +
                             std::strerror(errno));
  }
  const auto *p = static_cast<const char *>(data);
  size_t written = 0;
  while (written < bytes) {
    ssize_t n = ::write(fd, p + written, bytes - written);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      int saved = errno;
      ::close(fd);
      throw std::runtime_error("disk_flat_builder write failed: " + path.string() + ": " +
                               std::strerror(saved));
    }
    written += static_cast<size_t>(n);
  }
  if (::fsync(fd) != 0) {
    int saved = errno;
    ::close(fd);
    throw std::runtime_error("disk_flat_builder fsync failed: " + path.string() + ": " +
                             std::strerror(saved));
  }
  if (::close(fd) != 0) {
    throw std::runtime_error("disk_flat_builder close failed: " + path.string() + ": " +
                             std::strerror(errno));
  }
}

inline auto fsync_dir(const std::filesystem::path &dir) -> void {
  // O_NOFOLLOW per design D4: parent directory open MUST refuse symlinks.
  int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error("disk_flat_builder open dir failed: " + dir.string() + ": " +
                             std::strerror(errno));
  }
  if (::fsync(fd) != 0) {
    int saved = errno;
    ::close(fd);
    throw std::runtime_error("disk_flat_builder fsync dir failed: " + dir.string() + ": " +
                             std::strerror(saved));
  }
  if (::close(fd) != 0) {
    throw std::runtime_error("disk_flat_builder close dir failed: " + dir.string() + ": " +
                             std::strerror(errno));
  }
}

inline auto rename_no_replace(const std::filesystem::path &from, const std::filesystem::path &to)
    -> void {
#if defined(__linux__) && defined(SYS_renameat2) && defined(RENAME_NOREPLACE)
  auto rc = ::syscall(SYS_renameat2,
                      AT_FDCWD,
                      from.c_str(),
                      AT_FDCWD,
                      to.c_str(),
                      static_cast<unsigned int>(RENAME_NOREPLACE));
  if (rc != 0) {
    int saved = errno;
    if (saved == EEXIST) {
      throw std::runtime_error("disk_flat_builder target segment already exists: " + to.string());
    }
    throw std::runtime_error("disk_flat_builder renameat2 failed: " + from.string() + " -> " +
                             to.string() + ": " + std::strerror(saved));
  }
#else
  struct ::stat st{};
  if (::lstat(to.c_str(), &st) == 0) {
    throw std::runtime_error("disk_flat_builder target segment already exists: " + to.string());
  }
  if (errno != ENOENT) {
    int saved = errno;
    throw std::runtime_error("disk_flat_builder rename precheck failed: " + to.string() + ": " +
                             std::strerror(saved));
  }
  if (::rename(from.c_str(), to.c_str()) != 0) {
    int saved = errno;
    throw std::runtime_error("disk_flat_builder rename failed: " + from.string() + " -> " +
                             to.string() + ": " + std::strerror(saved));
  }
#endif
}

class TmpDirGuard {
 public:
  explicit TmpDirGuard(std::filesystem::path path) : path_(std::move(path)), armed_(true) {}
  ~TmpDirGuard() {
    if (armed_) {
      std::error_code ec;
      std::filesystem::remove_all(path_, ec);
    }
  }
  TmpDirGuard(const TmpDirGuard &) = delete;
  auto operator=(const TmpDirGuard &) -> TmpDirGuard & = delete;
  TmpDirGuard(TmpDirGuard &&) = delete;
  auto operator=(TmpDirGuard &&) -> TmpDirGuard & = delete;

  void disarm() { armed_ = false; }
  auto path() const -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_;
  bool armed_;
};

}  // namespace detail

class DiskFlatBuilder {
 public:
  DiskFlatBuilder(uint32_t dim, MetricType metric) : dim_(dim), metric_(metric) {
    if (dim == 0) {
      throw std::invalid_argument("DiskFlatBuilder: dim must be > 0");
    }
    if (metric != MetricType::L2 && metric != MetricType::IP && metric != MetricType::COS) {
      throw std::invalid_argument(
          "DiskFlatBuilder: metric must be one of L2, IP, COS (got NONE or unknown)");
    }
  }

  void add_batch(const float *vectors, const uint64_t *labels, uint64_t n) {
    if (closed_) {
      throw std::runtime_error("DiskFlatBuilder: builder is closed (already finished)");
    }
    if (n == 0) {
      return;
    }
    if (vectors == nullptr || labels == nullptr) {
      throw std::invalid_argument("DiskFlatBuilder: add_batch with n>0 requires non-null buffers");
    }
    const uint64_t row_offset = labels_.size();
    // Defense in depth: check n * dim_ for overflow before any arithmetic
    // that uses the product. (Final Codex review flagged this as an
    // archive blocker — without the check, a hostile n could wrap and
    // produce a small product that bypasses cap checks downstream.)
    size_t vec_components = 0;
    if (__builtin_mul_overflow(static_cast<size_t>(n),
                               static_cast<size_t>(dim_),
                               &vec_components)) {
      throw std::invalid_argument("DiskFlatBuilder: n * dim overflows size_t (n=" +
                                  std::to_string(n) + ", dim=" + std::to_string(dim_) + ")");
    }
    for (uint64_t r = 0; r < n; ++r) {
      for (uint32_t c = 0; c < dim_; ++c) {
        const float v = vectors[r * dim_ + c];
        if (!detail::is_finite_f32(v)) {
          const uint64_t global_row = row_offset + r;
          if (detail::is_nan_f32(v)) {
            throw std::invalid_argument("DiskFlatBuilder: NaN component at row " +
                                        std::to_string(global_row) + " position " +
                                        std::to_string(c));
          }
          const std::string sign = detail::is_neg_f32(v) ? "-Inf" : "+Inf";
          throw std::invalid_argument("DiskFlatBuilder: Inf component at row " +
                                      std::to_string(global_row) + " position " +
                                      std::to_string(c) + " (" + sign + ")");
        }
      }
    }
    vectors_.insert(vectors_.end(), vectors, vectors + vec_components);
    labels_.insert(labels_.end(), labels, labels + n);
  }

  auto finish(const std::filesystem::path &segment_dir) -> SegmentManifest {
    if (closed_) {
      throw std::runtime_error("DiskFlatBuilder: builder is closed (already finished)");
    }
    if (labels_.empty()) {
      throw std::runtime_error(
          "DiskFlatBuilder: finish called with zero rows (count=0 is rejected by manifest)");
    }

    const auto parent = segment_dir.parent_path();
    if (parent.empty()) {
      throw std::invalid_argument("DiskFlatBuilder: segment_dir must have a parent: " +
                                  segment_dir.string());
    }
    const std::string seg_basename = segment_dir.filename().string();
    if (!detail::is_valid_segment_id(seg_basename)) {
      throw std::invalid_argument(
          "DiskFlatBuilder: segment_dir basename must match ^seg_[0-9]{8}$: '" + seg_basename +
          "'");
    }

    {
      std::error_code ec;
      if (std::filesystem::exists(segment_dir, ec)) {
        throw std::runtime_error("DiskFlatBuilder: target segment already exists: " +
                                 segment_dir.string());
      }
    }

    const auto pid = static_cast<int64_t>(::getpid());
    const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string tmp_name =
        ".tmp_" + seg_basename + "_" + std::to_string(pid) + "_" + std::to_string(ts);
    const auto tmp_dir = parent / tmp_name;

    {
      std::error_code ec;
      std::filesystem::create_directories(tmp_dir, ec);
      if (ec) {
        throw std::runtime_error("DiskFlatBuilder: mkdir tmp failed: " + tmp_dir.string() + ": " +
                                 ec.message());
      }
    }
    detail::TmpDirGuard guard(tmp_dir);

    detail::write_all_fsync(tmp_dir / "ids.u64.bin",
                            labels_.data(),
                            labels_.size() * sizeof(uint64_t));

    if (metric_ == MetricType::COS) {
      std::vector<float> normalized(vectors_.size());
      const uint64_t count = labels_.size();
      for (uint64_t r = 0; r < count; ++r) {
        const float *src = vectors_.data() + r * dim_;
        float *dst = normalized.data() + r * dim_;
        double sum_sq = 0.0;
        for (uint32_t c = 0; c < dim_; ++c) {
          const double v = static_cast<double>(src[c]);
          sum_sq += v * v;
        }
        if (sum_sq == 0.0) {
          throw std::invalid_argument("DiskFlatBuilder: zero-magnitude vector under COS at row " +
                                      std::to_string(r));
        }
        const double inv_norm = 1.0 / std::sqrt(sum_sq);
        if (!detail::is_finite_f64(inv_norm)) {
          throw std::runtime_error(
              "DiskFlatBuilder: non-finite inverse norm (numerical defence in depth) at row " +
              std::to_string(r));
        }
        for (uint32_t c = 0; c < dim_; ++c) {
          dst[c] = static_cast<float>(static_cast<double>(src[c]) * inv_norm);
        }
      }
      detail::write_all_fsync(tmp_dir / "vectors.f32.bin",
                              normalized.data(),
                              normalized.size() * sizeof(float));
    } else {
      detail::write_all_fsync(tmp_dir / "vectors.f32.bin",
                              vectors_.data(),
                              vectors_.size() * sizeof(float));
    }

    SegmentManifest manifest{};
    manifest.version = kManifestVersion;
    manifest.segment_id = seg_basename;
    manifest.index_type = DiskIndexType::Flat;
    manifest.metric = metric_;
    manifest.dim = dim_;
    manifest.count = labels_.size();
    manifest.ids_file = "ids.u64.bin";
    manifest.vectors_file = "vectors.f32.bin";
    manifest.save(tmp_dir / "manifest.txt");

    detail::rename_no_replace(tmp_dir, segment_dir);
    guard.disarm();
    // The segment is now published — mark builder closed BEFORE the parent
    // fsync. A subsequent fsync_dir failure is a durability concern (the
    // rename might not survive a crash), but it does not unpublish the
    // segment. Marking closed here prevents the caller from catching an
    // fsync exception and double-publishing via a retry. (Spec mandates
    // single-use builders; Codex section-5 review flagged this.)
    closed_ = true;

    // Post-rename parent-dir fsync is a durability-only step. Treat its
    // failure as a soft warning — the rename has already happened, and
    // throwing here would propagate up to DiskCollection::flush() and
    // prevent it from advancing next_segment_id, leaving the caller
    // unable to retry. (Final Codex review flagged this as the second of
    // two archive blockers.)
    try {
      detail::fsync_dir(parent);
    } catch (const std::exception &e) {
      LOG_WARN("DiskFlatBuilder: post-rename parent fsync failed (durability only): {}", e.what());
    }
    return manifest;
  }

 private:
  uint32_t dim_;
  MetricType metric_;
  bool closed_ = false;
  std::vector<float> vectors_;
  std::vector<uint64_t> labels_;
};

}  // namespace alaya::disk
