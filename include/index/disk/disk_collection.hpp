// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <io.h>
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/file.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

#include "index/disk/segment_factory.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "storage/mmap_file.hpp"
#include "utils/log.hpp"
#include "utils/metric_type.hpp"
#include "utils/platform.hpp"
#include "utils/platform_fs.hpp"

namespace alaya::disk {

namespace detail {

inline auto format_segment_id(uint64_t id) -> std::string {
  char buf[16];  // "seg_" + 8 digits + NUL = 13
  std::snprintf(buf, sizeof(buf), "seg_%08" PRIu64, id);
  return std::string(buf);
}

// Standard atomic rename: replaces the destination if it exists. Used for
// collection_manifest.txt updates where the destination intentionally exists
// (segment-publish uses atomic_replace_no_overwrite in `disk_flat_builder.hpp`).
inline auto rename_atomic_replace(const std::filesystem::path &from,
                                  const std::filesystem::path &to) -> void {
  try {
    ::alaya::platform::atomic_replace(from, to);
  } catch (const std::exception &e) {
    throw std::runtime_error("collection_manifest rename failed: " + from.string() + " -> " +
                             to.string() + ": " + e.what());
  }
}

// Atomically writes + renames the collection manifest into place. The parent
// directory fsync is split out into a separate step (fsync_collection_dir
// below) because rename atomicity does NOT depend on the parent fsync — the
// new manifest is visible to subsequent opens immediately after rename. The
// parent fsync only promotes durability across a system crash; treating it
// as a soft step allows the in-memory state to commit on rename success.
inline auto publish_collection_manifest_atomic_only(const std::filesystem::path &collection_dir,
                                                    const CollectionManifest &m) -> void {
  const auto pid = ::alaya::platform::get_pid();
  const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::string tmp_name =
      ".tmp_collection_manifest_" + std::to_string(pid) + "_" + std::to_string(ts);
  const auto tmp_path = collection_dir / tmp_name;
  m.save(tmp_path);
  rename_atomic_replace(tmp_path, collection_dir / "collection_manifest.txt");
}

// Engine-agnostic ids-file handle for cross-segment label inventory. Resolves
// `seg_dir / sm.ids_file`, mmap's it under O_NOFOLLOW (inherited from
// MMapFile), and validates the file size is exactly `count * sizeof(uint64_t)`
// — a missing or differently-sized file is treated as corruption, with the
// segment directory and reason named in the message.
struct SegmentIdsView {
  alaya::storage::MMapFile mmap;
  uint64_t count = 0;

  auto data() const -> const uint64_t * { return static_cast<const uint64_t *>(mmap.data()); }
};

// Per-platform collection lock handle. On POSIX, the int file-descriptor backs
// `flock()`; on Windows, the HANDLE backs `LockFileEx()`. Both kernels release
// the lock automatically on handle close, so close() doubles as release.
#ifndef _WIN32
struct LockFd {
  int fd = -1;

  LockFd() = default;
  explicit LockFd(int f) : fd(f) {}
  LockFd(const LockFd &) = delete;
  auto operator=(const LockFd &) -> LockFd & = delete;
  LockFd(LockFd &&other) noexcept : fd(other.fd) { other.fd = -1; }
  auto operator=(LockFd &&other) noexcept -> LockFd & {
    if (this != &other) {
      close();
      fd = other.fd;
      other.fd = -1;
    }
    return *this;
  }
  ~LockFd() { close(); }

  // POSIX flock(2) auto-releases the kernel lock on last fd close, so close()
  // doubles as the lock-release path. Name avoids std::unique_ptr::release
  // vocabulary (which means "relinquish ownership without disposal").
  void close() noexcept {
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }
};
#else
struct LockFd {
  HANDLE handle = INVALID_HANDLE_VALUE;

  LockFd() = default;
  explicit LockFd(HANDLE h) : handle(h) {}
  LockFd(const LockFd &) = delete;
  auto operator=(const LockFd &) -> LockFd & = delete;
  LockFd(LockFd &&other) noexcept : handle(other.handle) { other.handle = INVALID_HANDLE_VALUE; }
  auto operator=(LockFd &&other) noexcept -> LockFd & {
    if (this != &other) {
      close();
      handle = other.handle;
      other.handle = INVALID_HANDLE_VALUE;
    }
    return *this;
  }
  ~LockFd() { close(); }

  // LockFileEx auto-releases the kernel lock on handle close. We do not call
  // UnlockFileEx explicitly because the open handle holds the lock for the
  // duration of the file-handle lifetime.
  void close() noexcept {
    if (handle != INVALID_HANDLE_VALUE && handle != nullptr) {
      ::CloseHandle(handle);
      handle = INVALID_HANDLE_VALUE;
    }
  }
};
#endif

inline auto absolute_lock_path_string(const std::filesystem::path &lock_path) -> std::string {
  std::error_code ec;
  auto abs_path = std::filesystem::absolute(lock_path, ec);
  if (ec) {
    return lock_path.string();
  }
  return abs_path.lexically_normal().string();
}

inline auto weakly_canonical_lock_path_string(const std::filesystem::path &lock_path)
    -> std::string {
  std::error_code ec;
  auto canonical = std::filesystem::weakly_canonical(lock_path, ec);
  if (!ec) {
    return canonical.string();
  }
  return absolute_lock_path_string(lock_path);
}

// Common pid-text parser. The lock-file format is `pid=<digits>\n`; the
// reader returns the digit run as a string (no integer parsing — keeps the
// "diagnostic only" contract intact).
inline auto parse_pid_text(std::string_view content) -> std::optional<std::string> {
  const auto pos = content.find("pid=");
  if (pos == std::string_view::npos) {
    return std::nullopt;
  }
  std::size_t begin = pos + 4;
  std::size_t end = begin;
  while (end < content.size() && content[end] >= '0' && content[end] <= '9') {
    ++end;
  }
  if (end == begin) {
    return std::nullopt;
  }
  return std::string(content.substr(begin, end - begin));
}

#ifndef _WIN32
inline auto read_lock_holder_pid(int fd) -> std::optional<std::string> {
  char buf[65] = {};
  const ssize_t n = ::pread(fd, buf, sizeof(buf) - 1, 0);
  if (n <= 0) {
    return std::nullopt;
  }
  return parse_pid_text(std::string_view(buf, static_cast<size_t>(n)));
}

inline void write_lock_holder_pid_best_effort(int fd) noexcept {
  char content[64];
  const int n =
      std::snprintf(content, sizeof(content), "pid=%" PRId64 "\n", ::alaya::platform::get_pid());
  if (n <= 0 || static_cast<size_t>(n) >= sizeof(content)) {
    return;
  }
  (void)::alaya::platform::truncate_open_file(fd, 0);
  (void)::pwrite(fd, content, static_cast<size_t>(n), 0);
}

[[noreturn]] inline void throw_lock_open_failed(const std::filesystem::path &lock_path, int saved) {
  const auto path = absolute_lock_path_string(lock_path);
  if (saved == EISDIR) {
    throw std::runtime_error("DiskCollection: lock path is not a regular file: " + path);
  }
  if (saved == ENOENT) {
    throw std::runtime_error("DiskCollection: collection path does not exist for lock file: " +
                             path);
  }
  if (saved == ELOOP) {
    throw std::runtime_error("DiskCollection: lock file is a symlink or has too many symlinks: " +
                             path + ": " + std::strerror(saved));
  }
  if (saved == ENOTDIR) {
    throw std::runtime_error("DiskCollection: lock file path contains a non-directory component: " +
                             path + ": " + std::strerror(saved));
  }
  throw std::runtime_error("DiskCollection: lock open failed: " + path + ": " +
                           std::strerror(saved));
}

inline auto is_lock_contention_errno(int saved) -> bool {
  return saved == EWOULDBLOCK || saved == EAGAIN;
}
#else
inline auto read_lock_holder_pid(HANDLE handle) -> std::optional<std::string> {
  // Move file pointer to 0 and read first 64 bytes via OVERLAPPED. Using a
  // local OVERLAPPED with Offset=0 leaves any concurrent position untouched.
  OVERLAPPED ov{};
  ov.Offset = 0;
  ov.OffsetHigh = 0;
  char buf[65] = {};
  DWORD got = 0;
  if (!::ReadFile(handle, buf, sizeof(buf) - 1, &got, &ov)) {
    return std::nullopt;
  }
  if (got == 0) {
    return std::nullopt;
  }
  return parse_pid_text(std::string_view(buf, got));
}

inline void write_lock_holder_pid_best_effort(HANDLE handle) noexcept {
  char content[64];
  const int n =
      std::snprintf(content, sizeof(content), "pid=%" PRId64 "\n", ::alaya::platform::get_pid());
  if (n <= 0 || static_cast<size_t>(n) >= sizeof(content)) {
    return;
  }
  // Truncate then overwrite. Failure is best-effort (diagnostic data only).
  if (!::alaya::platform::truncate_open_file(handle, 0)) {
    return;
  }
  OVERLAPPED ov{};
  ov.Offset = 0;
  ov.OffsetHigh = 0;
  DWORD written = 0;
  (void)::WriteFile(handle, content, static_cast<DWORD>(n), &written, &ov);
}

[[noreturn]] inline void throw_lock_open_failed(const std::filesystem::path &lock_path,
                                                DWORD saved) {
  const auto path = absolute_lock_path_string(lock_path);
  if (saved == ERROR_PATH_NOT_FOUND) {
    throw std::runtime_error("DiskCollection: collection path does not exist for lock file: " +
                             path);
  }
  if (saved == ERROR_ACCESS_DENIED) {
    throw std::runtime_error("DiskCollection: lock path is not a regular file: " + path +
                             ": Win32 error " + std::to_string(saved));
  }
  throw std::runtime_error("DiskCollection: lock open failed: " + path + ": Win32 error " +
                           std::to_string(saved));
}

inline auto is_lock_contention_win32_error(DWORD err) -> bool {
  return err == ERROR_LOCK_VIOLATION || err == ERROR_SHARING_VIOLATION;
}
#endif

#ifndef _WIN32
// Post-flock inode revalidation. Compares the fd's fstat-derived
// `(st_dev, st_ino)` (captured by the caller before flock) against a fresh
// `stat(lock_path)`. Throws on mismatch or when `lock_path` no longer exists
// (the file was unlinked while we hold the fd open). Only swaps that happen
// BETWEEN the caller's fstat and this stat are catchable — a swap that
// completes before the caller's open is invisible because both readings
// observe the post-swap inode.
inline void revalidate_lock_inode_after_flock(const std::filesystem::path &lock_path,
                                              dev_t fd_st_dev,
                                              ino_t fd_st_ino) {
  struct ::stat st_path{};
  if (::stat(lock_path.c_str(), &st_path) != 0) {
    const int saved = errno;
    if (saved == ENOENT) {
      throw std::runtime_error("DiskCollection: lock file vanished after flock: " +
                               weakly_canonical_lock_path_string(lock_path));
    }
    throw std::runtime_error("DiskCollection: lock stat failed after flock: " +
                             weakly_canonical_lock_path_string(lock_path) + ": " +
                             std::strerror(saved));
  }
  if (fd_st_ino != st_path.st_ino || fd_st_dev != st_path.st_dev) {
    throw std::runtime_error("DiskCollection: lock file inode mismatch after acquire: " +
                             weakly_canonical_lock_path_string(lock_path));
  }
}
#endif

// AcquireMode selects between the open() entry (O_CREAT, tolerates an
// existing `.lock` for legacy-collection compatibility) and the ctor entry
// (O_CREAT|O_EXCL, atomic ownership of the freshly-created `.lock`).
enum class AcquireMode { ForOpen, ForCreate };

#ifndef _WIN32
inline auto acquire_collection_lock_impl(const std::filesystem::path &collection_root,
                                         AcquireMode mode) -> LockFd {
  const auto lock_path = collection_root / ".lock";
  const int base_flags = O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW;
  const int flags = (mode == AcquireMode::ForCreate) ? (base_flags | O_EXCL) : base_flags;
  const int fd = ::open(lock_path.c_str(), flags, 0600);
  if (fd < 0) {
    const int saved = errno;
    if (saved == EEXIST && mode == AcquireMode::ForCreate) {
      // O_EXCL EEXIST: another constructor (or stray `.lock`) holds the
      // namespace. Dual-substring: weakly_canonical path + literal phrase.
      throw std::runtime_error(
          "DiskCollection: target path already exists or is being created concurrently: " +
          weakly_canonical_lock_path_string(lock_path));
    }
    throw_lock_open_failed(lock_path, saved);
  }

  LockFd lock(fd);
  // Pre-flock validation. For `_for_create`, our successful O_EXCL means we
  // are the sole owner of the freshly-created `.lock` inode until something
  // else flocks it (or until our own flock attempt). A pre-flock failure
  // (fstat / !S_ISREG) is a rare-but-possible state; unlink lock_path so
  // a retry can re-EXCL. We deliberately do NOT unlink after flock failure
  // (a concurrent `_for_open` may have already grabbed flock on our
  // inode) nor after post-flock inode revalidation failure (lock_path may
  // already point to a different inode that we don't own).
  auto unlink_for_create_pre_flock = [&]() noexcept {
    if (mode == AcquireMode::ForCreate) {
      std::error_code unec;
      std::filesystem::remove(lock_path, unec);
    }
  };
  struct ::stat st{};
  if (::fstat(lock.fd, &st) != 0) {
    const int saved = errno;
    unlink_for_create_pre_flock();
    throw std::runtime_error("DiskCollection: lock fstat failed: " +
                             absolute_lock_path_string(lock_path) + ": " + std::strerror(saved));
  }
  if (!S_ISREG(st.st_mode)) {
    unlink_for_create_pre_flock();
    throw std::runtime_error("DiskCollection: lock path is not a regular file: " +
                             absolute_lock_path_string(lock_path));
  }

  if (::flock(lock.fd, LOCK_EX | LOCK_NB) != 0) {
    const int saved = errno;
    if (is_lock_contention_errno(saved)) {
      // Stable dual-substring contract: messages for contention must contain
      // the absolute `.lock` path and "collection is already open by another
      // process", matching the existing unsupported-engine matching style.
      // PID read is best-effort and only attempted on the contention path so
      // a transient pread failure on a non-contention errno path cannot
      // perturb the acquire-failure throw.
      const auto holder_pid = read_lock_holder_pid(lock.fd);
      std::string msg = "DiskCollection: collection is already open by another process: " +
                        weakly_canonical_lock_path_string(lock_path);
      if (holder_pid.has_value()) {
        msg += " (pid=" + *holder_pid + ")";
      }
      throw std::runtime_error(msg);
    }
    throw std::runtime_error(
        "DiskCollection: lock flock failed: " + weakly_canonical_lock_path_string(lock_path) +
        ": " + std::strerror(saved));
  }

  // Post-flock inode revalidation. See `revalidate_lock_inode_after_flock`
  // for the bounded detection contract; the helper throws on `unlink+touch`
  // swaps that happen between this acquire's `fstat` and `stat` calls.
  revalidate_lock_inode_after_flock(lock_path, st.st_dev, st.st_ino);

  write_lock_holder_pid_best_effort(lock.fd);
  return lock;
}
#else
inline auto acquire_collection_lock_impl(const std::filesystem::path &collection_root,
                                         AcquireMode mode) -> LockFd {
  const auto lock_path = collection_root / ".lock";
  // Sharing intentionally excludes DELETE so the lock file cannot be unlinked
  // / renamed while we hold the handle. This is the Win32 analogue of the
  // POSIX inode-swap defense — preventing the swap, rather than detecting it
  // post-acquire.
  const DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE;
  const DWORD creation = (mode == AcquireMode::ForCreate) ? static_cast<DWORD>(CREATE_NEW)
                                                          : static_cast<DWORD>(OPEN_ALWAYS);
  HANDLE handle = ::CreateFileW(lock_path.c_str(),
                                GENERIC_READ | GENERIC_WRITE,
                                share_mode,
                                nullptr,
                                creation,
                                FILE_ATTRIBUTE_NORMAL,
                                nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    const auto saved = ::GetLastError();
    if (mode == AcquireMode::ForCreate &&
        (saved == ERROR_FILE_EXISTS || saved == ERROR_ALREADY_EXISTS)) {
      throw std::runtime_error(
          "DiskCollection: target path already exists or is being created concurrently: " +
          weakly_canonical_lock_path_string(lock_path));
    }
    if (saved == ERROR_SHARING_VIOLATION) {
      // Another process opened the lock file without sharing delete — treat
      // as contention (matches the POSIX contention message contract).
      throw std::runtime_error("DiskCollection: collection is already open by another process: " +
                               weakly_canonical_lock_path_string(lock_path));
    }
    throw_lock_open_failed(lock_path, saved);
  }
  LockFd lock(handle);

  // Try-lock with LOCKFILE_FAIL_IMMEDIATELY. Lock the maximum byte range
  // (UINT64_MAX) so any concurrent partial-range lock is rejected too.
  OVERLAPPED ov{};
  if (!::LockFileEx(lock.handle,
                    LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                    0,
                    MAXDWORD,
                    MAXDWORD,
                    &ov)) {
    const auto saved = ::GetLastError();
    if (is_lock_contention_win32_error(saved)) {
      const auto holder_pid = read_lock_holder_pid(lock.handle);
      std::string msg = "DiskCollection: collection is already open by another process: " +
                        weakly_canonical_lock_path_string(lock_path);
      if (holder_pid.has_value()) {
        msg += " (pid=" + *holder_pid + ")";
      }
      throw std::runtime_error(msg);
    }
    throw std::runtime_error(
        "DiskCollection: lock LockFileEx failed: " + weakly_canonical_lock_path_string(lock_path) +
        ": Win32 error " + std::to_string(saved));
  }

  // Inode-swap revalidation has no Win32 analogue with these share flags —
  // the kernel guarantees the file cannot be renamed/unlinked while open.
  write_lock_holder_pid_best_effort(lock.handle);
  return lock;
}
#endif

// Open-entry acquire: tolerates an existing `.lock`. Used ONLY by
// `DiskCollection::open(path)`. Backwards-compatible with legacy collections
// that predate the lock (open(2) creates `.lock` on demand).
inline auto acquire_collection_lock_for_open(const std::filesystem::path &collection_root)
    -> LockFd {
  return acquire_collection_lock_impl(collection_root, AcquireMode::ForOpen);
}

// Ctor-entry acquire: atomically creates `.lock`. Used ONLY by the
// `DiskCollection` constructor. EEXIST is reflected as
// "target path already exists or is being created concurrently" so callers
// can distinguish a colliding ctor / pre-existing lock from a generic
// flock contention case.
inline auto acquire_collection_lock_for_create(const std::filesystem::path &collection_root)
    -> LockFd {
  return acquire_collection_lock_impl(collection_root, AcquireMode::ForCreate);
}

inline auto load_segment_ids_view(const std::filesystem::path &seg_dir) -> SegmentIdsView {
  const auto manifest_path = seg_dir / "manifest.txt";
  std::error_code ec;
  if (!std::filesystem::exists(manifest_path, ec) || ec) {
    throw std::runtime_error("DiskCollection: segment manifest missing for inventory: " +
                             manifest_path.string());
  }
  const auto sm = SegmentManifest::load(manifest_path);
  const auto ids_path = seg_dir / sm.ids_file;
  if (!std::filesystem::exists(ids_path, ec) || ec) {
    throw std::runtime_error("DiskCollection: segment ids_file missing for inventory: " +
                             ids_path.string());
  }
  const uint64_t expected_bytes = sm.count * sizeof(uint64_t);
  // MMapFile's O_NOFOLLOW + non-regular-file rejection covers symlink and
  // device-node attacks, with the path embedded in the resulting message.
  alaya::storage::MMapFile mmap(ids_path);
  if (mmap.size() != expected_bytes) {
    throw std::runtime_error("DiskCollection: segment ids_file size mismatch for inventory at " +
                             seg_dir.string() + " (expected " + std::to_string(expected_bytes) +
                             " bytes for count=" + std::to_string(sm.count) + ", got " +
                             std::to_string(mmap.size()) + ")");
  }
  return SegmentIdsView{std::move(mmap), sm.count};
}

inline auto parse_u32_extra(const CollectionManifest &manifest,
                            const std::string &key,
                            uint32_t fallback) -> uint32_t {
  const auto it = manifest.x_extras.find(key);
  if (it == manifest.x_extras.end()) {
    return fallback;
  }
  try {
    size_t pos = 0;
    const auto value = std::stoull(it->second, &pos);
    if (pos != it->second.size() || value > std::numeric_limits<uint32_t>::max()) {
      throw std::out_of_range("not uint32");
    }
    return static_cast<uint32_t>(value);
  } catch (const std::exception &e) {
    throw std::runtime_error("DiskCollection: invalid collection manifest " + key + "='" +
                             it->second + "': " + e.what());
  }
}

inline auto parse_u64_extra(const CollectionManifest &manifest,
                            const std::string &key,
                            uint64_t fallback) -> uint64_t {
  const auto it = manifest.x_extras.find(key);
  if (it == manifest.x_extras.end()) {
    return fallback;
  }
  try {
    size_t pos = 0;
    const auto value = std::stoull(it->second, &pos);
    if (pos != it->second.size()) {
      throw std::out_of_range("not uint64");
    }
    return value;
  } catch (const std::exception &e) {
    throw std::runtime_error("DiskCollection: invalid collection manifest " + key + "='" +
                             it->second + "': " + e.what());
  }
}

inline auto parse_float_extra(const CollectionManifest &manifest,
                              const std::string &key,
                              float fallback) -> float {
  const auto it = manifest.x_extras.find(key);
  if (it == manifest.x_extras.end()) {
    return fallback;
  }
  try {
    size_t pos = 0;
    const auto value = std::stof(it->second, &pos);
    if (pos != it->second.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return value;
  } catch (const std::exception &e) {
    throw std::runtime_error("DiskCollection: invalid collection manifest " + key + "='" +
                             it->second + "': " + e.what());
  }
}

inline auto is_finite_extra_f32(float value) -> bool {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(value));
  return (bits & 0x7F800000U) != 0x7F800000U;
}

inline auto load_vamana_params_from_manifest(const CollectionManifest &manifest)
    -> VamanaSegmentBuildParams {
  VamanaSegmentBuildParams params;
  params.R = parse_u32_extra(manifest, "x_vamana_R", params.R);
  params.L = parse_u32_extra(manifest, "x_vamana_L", params.L);
  params.alpha = parse_float_extra(manifest, "x_vamana_alpha", params.alpha);
  params.seed = parse_u32_extra(manifest, "x_vamana_seed", static_cast<uint32_t>(params.seed));
  params.num_threads = parse_u32_extra(manifest, "x_vamana_num_threads", params.num_threads);
  return params;
}

inline auto validate_vamana_params(const VamanaSegmentBuildParams &params,
                                   const std::string &context) -> void {
  if (params.R == 0) {
    throw std::runtime_error(context + ": x_vamana_R must be > 0");
  }
  if (params.L == 0) {
    throw std::runtime_error(context + ": x_vamana_L must be > 0");
  }
  if (params.L < params.R) {
    throw std::runtime_error(context + ": x_vamana_L must be >= x_vamana_R");
  }
  if (!is_finite_extra_f32(params.alpha) || params.alpha < 1.0F) {
    throw std::runtime_error(context + ": x_vamana_alpha must be finite and >= 1.0");
  }
}

inline auto validate_vamana_manifest_config(const CollectionManifest &manifest,
                                            const VamanaSegmentBuildParams &params,
                                            const std::string &context) -> void {
  if (manifest.metric != MetricType::L2) {
    throw std::runtime_error(context + ": metric must be L2 for disk_vamana");
  }
  validate_vamana_params(params, context);
}

inline auto store_vamana_params_in_manifest(CollectionManifest &manifest,
                                            const VamanaSegmentBuildParams &params) -> void {
  manifest.x_extras["x_vamana_R"] = std::to_string(params.R);
  manifest.x_extras["x_vamana_L"] = std::to_string(params.L);
  manifest.x_extras["x_vamana_alpha"] = std::to_string(params.alpha);
  manifest.x_extras["x_vamana_seed"] = std::to_string(params.seed);
  manifest.x_extras["x_vamana_num_threads"] = std::to_string(params.num_threads);
}

inline auto store_max_pending_bytes_in_manifest(CollectionManifest &manifest,
                                                size_t max_pending_bytes) -> void {
  manifest.x_extras["x_max_pending_bytes"] = std::to_string(max_pending_bytes);
}

inline auto load_max_pending_bytes_from_manifest(const CollectionManifest &manifest,
                                                 size_t fallback) -> size_t {
  const auto value =
      parse_u64_extra(manifest, "x_max_pending_bytes", static_cast<uint64_t>(fallback));
  if (value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error("DiskCollection: invalid collection manifest x_max_pending_bytes='" +
                             std::to_string(value) + "': exceeds size_t");
  }
  return static_cast<size_t>(value);
}

inline auto disk_search_distance_less(float a, float b) -> bool {
  const bool a_nan = std::isnan(a);
  const bool b_nan = std::isnan(b);
  if (a_nan || b_nan) {
    return !a_nan && b_nan;
  }
  return a < b;
}

inline auto disk_search_distance_equal_for_order(float a, float b) -> bool {
  return (std::isnan(a) && std::isnan(b)) || a == b;
}

}  // namespace detail

class DiskCollection {
 public:
  // Default soft cap on in-memory pending bytes before forcing a publish.
  static constexpr size_t kDefaultMaxPendingBytes = 512ULL * 1024 * 1024;

  // Public constructor: create-only.
  DiskCollection(const std::filesystem::path &path,
                 uint32_t dim,
                 MetricType metric,
                 DiskIndexType index_type,
                 size_t max_pending_bytes = kDefaultMaxPendingBytes,
                 VamanaSegmentBuildParams vamana_params = VamanaSegmentBuildParams{}) {
    if (dim == 0) {
      throw std::invalid_argument("DiskCollection: dim must be > 0");
    }
    if (metric != MetricType::L2 && metric != MetricType::IP && metric != MetricType::COS) {
      throw std::invalid_argument(
          "DiskCollection: metric must be one of L2, IP, COS (got NONE or unknown)");
    }
    // v1 capability gate is owned by the factory: this throws with the dual
    // substring contract ("disk_<engine>" + "not implemented in v1") so the
    // existing scenarios on DiskCollection's error message stay green and the
    // single source of truth lives in segment_factory.
    assert_engine_supported_v1(index_type);
    if (index_type == DiskIndexType::Vamana) {
      detail::validate_vamana_params(vamana_params, "DiskCollection");
    }
    // Ctor ordering:
    //   exists(path) → mkdir(path) (root only) → _for_create (O_EXCL, atomic
    //   ownership) → mkdir(path/segments) → publish_manifest.
    // The O_EXCL acquire is the kernel-level serialization point. Anything
    // that becomes externally observable before acquire (the bare `path/`
    // root) is rolled back via best-effort rmdir on acquire failure.
    {
      std::error_code ec;
      if (std::filesystem::exists(path, ec) || ec) {
        // Unified dual-substring contract with the `_for_create` EEXIST
        // path. Either `path/` is a stale collection from a previous run
        // (and the user must `rm -rf` it) or a concurrent ctor's mkdir
        // beat ours — both cases are "exists or being created
        // concurrently". The message contains `weakly_canonical(.lock
        // path)` so downstream operators can locate the offending
        // directory regardless of which branch fired.
        throw std::runtime_error(
            "DiskCollection: target path already exists or is being created concurrently: " +
            detail::weakly_canonical_lock_path_string(path / ".lock"));
      }
      std::filesystem::create_directories(path, ec);
      if (ec) {
        throw std::runtime_error("DiskCollection: mkdir failed: " + path.string() + ": " +
                                 ec.message());
      }
    }

    try {
      lock_fd_ = detail::acquire_collection_lock_for_create(path);
    } catch (...) {
      // Best-effort rollback of our just-created `path/`. rmdir succeeds
      // only when the directory is empty; if another concurrent ctor won
      // the EXCL race and dropped its `.lock` into the same dir, the rmdir
      // returns ENOTEMPTY which we silently ignore — that ctor owns the
      // path and its manifest publication is independent of ours.
      std::error_code rmec;
      std::filesystem::remove(path, rmec);
      throw;
    }
    path_ = path;
    {
      std::error_code ec;
      std::filesystem::create_directories(path / "segments", ec);
      if (ec) {
        throw std::runtime_error("DiskCollection: mkdir failed: " + (path / "segments").string() +
                                 ": " + ec.message());
      }
    }
    manifest_.version = kManifestVersion;
    manifest_.dim = dim;
    manifest_.metric = metric;
    manifest_.index_type = index_type;
    manifest_.next_segment_id = 1;
    manifest_.segment_ids.clear();
    max_pending_bytes_ = max_pending_bytes;
    vamana_params_ = vamana_params;
    detail::store_max_pending_bytes_in_manifest(manifest_, max_pending_bytes_);
    if (manifest_.index_type == DiskIndexType::Vamana) {
      detail::store_vamana_params_in_manifest(manifest_, vamana_params_);
    }

    // Constructor publish: atomic write + best-effort durability fsync.
    detail::publish_collection_manifest_atomic_only(path_, manifest_);
    try {
      detail::fsync_dir(path_);
    } catch (const std::exception &e) {
      LOG_WARN("DiskCollection: ctor fsync_dir failed (durability only): {}", e.what());
    }
  }

  static auto open(const std::filesystem::path &path) -> DiskCollection {
    {
      std::error_code ec;
      if (!std::filesystem::exists(path, ec) || ec) {
        throw std::runtime_error("DiskCollection::open: path does not exist: " + path.string());
      }
    }
    // Caller-level "collection-in-progress" precondition: when `path/`
    // exists but `.lock` is missing AND no published `collection_manifest.txt`
    // is present, we are observing a constructor that has created `path/`
    // but has not yet acquired its O_EXCL `.lock`. Throwing the documented
    // dual-substring error here gives the caller a clear retry signal
    // before we touch `_for_open`. Legacy collections (no `.lock` but a
    // valid manifest) intentionally fall through — `_for_open` will create
    // `.lock` for them.
    {
      std::error_code ec_lock;
      std::error_code ec_manifest;
      const bool lock_exists = std::filesystem::exists(path / ".lock", ec_lock);
      const bool manifest_exists =
          std::filesystem::exists(path / "collection_manifest.txt", ec_manifest);
      if (!lock_exists && !manifest_exists) {
        throw std::runtime_error(
            "DiskCollection::open: target path is a collection-in-progress, not yet published: " +
            detail::weakly_canonical_lock_path_string(path / ".lock"));
      }
    }
    DiskCollection col;
    col.path_ = path;
    // Manifest load, listed segment open, and orphan scan must run while the
    // collection-level writer lock is held.
    col.lock_fd_ = detail::acquire_collection_lock_for_open(path);
    col.manifest_ = CollectionManifest::load(path / "collection_manifest.txt");
    // Same v1 capability gate as the constructor; delegated to the factory so
    // the dual-substring message contract has a single source of truth.
    assert_engine_supported_v1(col.manifest_.index_type);
    col.max_pending_bytes_ =
        detail::load_max_pending_bytes_from_manifest(col.manifest_, kDefaultMaxPendingBytes);
    if (col.manifest_.index_type == DiskIndexType::Vamana) {
      col.vamana_params_ = detail::load_vamana_params_from_manifest(col.manifest_);
      detail::validate_vamana_manifest_config(col.manifest_,
                                              col.vamana_params_,
                                              "DiskCollection::open");
    }
    col.open_listed_segments();
    col.scan_orphans();
    return col;
  }

  DiskCollection(const DiskCollection &) = delete;
  auto operator=(const DiskCollection &) -> DiskCollection & = delete;
  DiskCollection(DiskCollection &&) = default;
  auto operator=(DiskCollection &&) -> DiskCollection & = default;
  ~DiskCollection() = default;

  void add_batch(const float *vectors, const uint64_t *labels, uint64_t n) {
    if (manifest_.index_type == DiskIndexType::Laser) {
      throw std::runtime_error(
          "DiskCollection: disk_laser add_batch not implemented in v1; use "
          "import_laser_segment");
    }
    if (n == 0) {
      return;
    }
    if (vectors == nullptr || labels == nullptr) {
      throw std::invalid_argument("DiskCollection: add_batch with n>0 requires non-null buffers");
    }
    const uint64_t per_row =
        static_cast<uint64_t>(manifest_.dim) * sizeof(float) + sizeof(uint64_t);
    const uint64_t cap = max_pending_bytes_;

    // Check n * per_row * 2 for overflow before any cap comparison. Without
    // this, a hostile n could wrap to a small value and bypass the cap check.
    uint64_t single_batch_bytes = 0;
    if (alaya_mul_overflow(uint64_t{2}, n, &single_batch_bytes) ||
        alaya_mul_overflow(single_batch_bytes, per_row, &single_batch_bytes)) {
      throw std::runtime_error("DiskCollection: pending size arithmetic overflows uint64 (n=" +
                               std::to_string(n) + ", dim=" + std::to_string(manifest_.dim) + ")");
    }
    if (single_batch_bytes > cap) {
      throw std::runtime_error("DiskCollection: single batch (" +
                               std::to_string(single_batch_bytes) +
                               " bytes) exceeds max_pending_bytes (" + std::to_string(cap) +
                               " bytes); split the batch or raise max_pending_bytes");
    }

    const uint64_t current_rows = pending_labels_.size();
    const uint64_t current_total = 2ULL * current_rows * per_row;
    uint64_t total_rows = 0;
    uint64_t new_total = 0;
    if (alaya_add_overflow(current_rows, n, &total_rows) ||
        alaya_mul_overflow(uint64_t{2}, total_rows, &new_total) ||
        alaya_mul_overflow(new_total, per_row, &new_total)) {
      throw std::runtime_error(
          "DiskCollection: pending+batch arithmetic overflows uint64 (current_rows=" +
          std::to_string(current_rows) + ", n=" + std::to_string(n) + ")");
    }
    if (new_total > cap) {
      const uint64_t remaining = (cap > current_total) ? (cap - current_total) : 0ULL;
      const uint64_t max_addable = (per_row > 0) ? (remaining / (2ULL * per_row)) : 0ULL;
      throw std::runtime_error("DiskCollection: pending buffer would overflow — current=" +
                               std::to_string(current_total) +
                               " bytes, cap=" + std::to_string(cap) +
                               " bytes, addable_rows_under_dim=" + std::to_string(max_addable));
    }

    // Strong exception safety: reserve both buffers BEFORE any insert. If
    // reserve throws bad_alloc, no state has been mutated. After successful
    // reserve, the inserts cannot allocate again so they can't throw.
    pending_vectors_.reserve(pending_vectors_.size() + n * manifest_.dim);
    pending_labels_.reserve(pending_labels_.size() + n);
    pending_vectors_.insert(pending_vectors_.end(), vectors, vectors + n * manifest_.dim);
    pending_labels_.insert(pending_labels_.end(), labels, labels + n);
  }

  void flush() {
    if (pending_labels_.empty()) {
      return;
    }
    // Within-batch label uniqueness.
    std::unordered_set<uint64_t> pending_set;
    pending_set.reserve(pending_labels_.size() * 2);
    for (auto label : pending_labels_) {
      auto [it, inserted] = pending_set.insert(label);
      if (!inserted) {
        throw std::invalid_argument("DiskCollection: duplicate label within pending batch: " +
                                    std::to_string(label));
      }
    }
    // Cross-segment uniqueness via engine-agnostic manifest.ids_file mmap —
    // no dynamic_cast against any concrete SegmentSearcher subclass. Each
    // existing segment is identified by its directory in segment_dirs_, kept
    // in lockstep with segments_.
    for (const auto &seg_dir : segment_dirs_) {
      auto ids_view = detail::load_segment_ids_view(seg_dir);
      const uint64_t *labels = ids_view.data();
      const uint64_t cnt = ids_view.count;
      for (uint64_t i = 0; i < cnt; ++i) {
        if (pending_set.contains(labels[i])) {
          throw std::invalid_argument("DiskCollection: duplicate label across segments: " +
                                      std::to_string(labels[i]));
        }
      }
    }

    if (manifest_.index_type == DiskIndexType::Vamana && pending_labels_.size() < 2) {
      throw std::runtime_error("DiskCollection: disk_vamana flush requires at least 2 rows");
    }

    const uint64_t seg_id = manifest_.next_segment_id;
    const std::string seg_basename = detail::format_segment_id(seg_id);
    const auto seg_dir = path_ / "segments" / seg_basename;

    // Drive segment construction through the factory. v1 routes Flat to the
    // existing builder; this preserves byte-level behaviour while consolidating
    // the dispatch decision in one place. The factory call also constructs
    // the searcher, so a builder-side or open-side failure is reported from
    // a single throw site.
    auto searcher = create_segment_from_pending(seg_dir,
                                                manifest_,
                                                pending_vectors_.data(),
                                                pending_labels_.data(),
                                                pending_labels_.size(),
                                                vamana_params_);
    // Segment is now on disk. Any subsequent failure in this function leaves
    // an orphan segment that will be classified as kind=complete on next open.
    // Eagerly advance in-memory next_segment_id so a retried flush() uses a
    // fresh id rather than colliding with the orphan.
    manifest_.next_segment_id = seg_id + 1;

    // Atomic rename of the collection manifest. Once this returns, the segment
    // is officially listed on disk. If this throws, segment is still orphan.
    auto new_manifest = manifest_;
    new_manifest.segment_ids.push_back(seg_basename);
    new_manifest.next_segment_id = seg_id + 1;
    detail::publish_collection_manifest_atomic_only(path_, new_manifest);

    // From this point on, the on-disk state is consistent. Commit in-memory
    // state atomically. Pending is cleared LAST so any throw above leaves it
    // intact for the caller (the fsync below is best-effort, not load-bearing).
    manifest_ = std::move(new_manifest);
    segments_.push_back(std::move(searcher));
    segment_dirs_.push_back(seg_dir);
    pending_vectors_.clear();
    pending_labels_.clear();
    pending_vectors_.shrink_to_fit();
    pending_labels_.shrink_to_fit();

    // Parent-dir fsync is a durability-only step. If it fails, the rename
    // already happened — in-memory state is correct, the only impact is that
    // the rename might not survive a power loss. Log and proceed.
    try {
      detail::fsync_dir(path_);
    } catch (const std::exception &e) {
      LOG_WARN("DiskCollection: collection_manifest fsync_dir failed (durability only): {}",
               e.what());
    }
  }

  void import_laser_segment(const std::filesystem::path &src_dir,
                            const uint64_t *labels,
                            uint64_t n) {
    if (manifest_.index_type != DiskIndexType::Laser) {
      throw std::runtime_error(
          "DiskCollection: import_laser_segment requires a disk_laser collection");
    }
    if (n == 0) {
      throw std::invalid_argument("DiskCollection: import_laser_segment requires n >= 1");
    }
    if (labels == nullptr) {
      throw std::invalid_argument(
          "DiskCollection: import_laser_segment with n>0 requires non-null labels");
    }

    std::unordered_set<uint64_t> import_set;
    if (n <= static_cast<uint64_t>(std::numeric_limits<size_t>::max() / 2)) {
      import_set.reserve(static_cast<size_t>(n * 2));
    }
    for (uint64_t i = 0; i < n; ++i) {
      auto [it, inserted] = import_set.insert(labels[i]);
      if (!inserted) {
        throw std::invalid_argument(
            "DiskCollection: duplicate label within import_laser_segment batch: " +
            std::to_string(labels[i]));
      }
    }

    for (const auto &seg_dir : segment_dirs_) {
      auto ids_view = detail::load_segment_ids_view(seg_dir);
      const uint64_t *existing = ids_view.data();
      const uint64_t cnt = ids_view.count;
      for (uint64_t i = 0; i < cnt; ++i) {
        if (import_set.contains(existing[i])) {
          throw std::invalid_argument("DiskCollection: duplicate label across segments: " +
                                      std::to_string(existing[i]));
        }
      }
    }

    const uint64_t seg_id = manifest_.next_segment_id;
    const std::string seg_basename = detail::format_segment_id(seg_id);
    const auto seg_dir = path_ / "segments" / seg_basename;

    auto searcher = import_segment_from_artifacts(seg_dir, manifest_, src_dir, labels, n);

    // Segment is now on disk. Match flush(): advance next_segment_id before
    // manifest publication so a retry after manifest-publish failure uses a
    // fresh id and leaves the published segment for orphan classification.
    manifest_.next_segment_id = seg_id + 1;

    auto new_manifest = manifest_;
    new_manifest.segment_ids.push_back(seg_basename);
    new_manifest.next_segment_id = seg_id + 1;
    detail::publish_collection_manifest_atomic_only(path_, new_manifest);

    manifest_ = std::move(new_manifest);
    segments_.push_back(std::move(searcher));
    segment_dirs_.push_back(seg_dir);

    try {
      detail::fsync_dir(path_);
    } catch (const std::exception &e) {
      LOG_WARN("DiskCollection: collection_manifest fsync_dir failed (durability only): {}",
               e.what());
    }
  }

  auto search(const float *query, const DiskSearchOptions &opts) const
      -> std::vector<DiskSearchHit> {
    if (opts.top_k == 0) {
      throw std::invalid_argument("DiskCollection: top_k must be > 0");
    }
    if (segments_.empty()) {
      return {};
    }

    if (segments_.size() == 1) {
      auto hits = segments_[0]->search(query, opts);
      if (segments_[0]->type() == DiskIndexType::Laser) {
        if (hits.size() > opts.top_k) {
          hits.resize(opts.top_k);
        }
        return hits;
      }
      std::sort(hits.begin(), hits.end(), [](const DiskSearchHit &a, const DiskSearchHit &b) {
        if (!detail::disk_search_distance_equal_for_order(a.distance, b.distance)) {
          return detail::disk_search_distance_less(a.distance, b.distance);
        }
        return a.label < b.label;
      });
      if (hits.size() > opts.top_k) {
        hits.resize(opts.top_k);
      }
      return hits;
    }

    struct Tagged {
      DiskSearchHit hit;
      uint32_t segment_index;
      size_t rank;
    };

    const bool preserve_laser_rank = segments_[0]->type() == DiskIndexType::Laser;
    std::vector<Tagged> all;
    all.reserve(segments_.size() * opts.top_k);
    for (uint32_t s = 0; s < segments_.size(); ++s) {
      auto seg_hits = segments_[s]->search(query, opts);
      for (size_t rank = 0; rank < seg_hits.size(); ++rank) {
        all.push_back(Tagged{seg_hits[rank], s, rank});
      }
    }
    // LASER segment hits use NaN distances today; keep segment-local rank as the
    // equal-distance tie-break so multi-segment search matches the single-segment
    // raw engine ordering contract.
    std::sort(all.begin(), all.end(), [preserve_laser_rank](const Tagged &a, const Tagged &b) {
      if (!detail::disk_search_distance_equal_for_order(a.hit.distance, b.hit.distance)) {
        return detail::disk_search_distance_less(a.hit.distance, b.hit.distance);
      }
      if (preserve_laser_rank) {
        if (a.rank != b.rank) {
          return a.rank < b.rank;
        }
        return a.segment_index < b.segment_index;
      }
      if (a.hit.label != b.hit.label) {
        return a.hit.label < b.hit.label;
      }
      return a.segment_index < b.segment_index;
    });

    const size_t k = std::min<size_t>(opts.top_k, all.size());
    std::vector<DiskSearchHit> out;
    out.reserve(k);
    for (size_t i = 0; i < k; ++i) {
      out.push_back(all[i].hit);
    }
    return out;
  }

  // Run `n_queries` searches against `queries` (row-major `(n_queries, dim)`
  // float buffer) and write results into the caller-owned output buffers
  // `out_labels` / `out_distances` (row-major `(n_queries, opts.top_k)`).
  //
  // Padding contract: the caller MUST pre-fill `out_labels` with
  // `UINT64_MAX` and (if non-null) `out_distances` with NaN. Per the
  // `disk-collection-batch-search` spec the implementation only overwrites
  // the slots `[0, hits.size())` for each query, so trailing slots retain
  // the caller's sentinels and the labels-only fast path is selected by
  // passing `out_distances == nullptr`.
  //
  // `n_queries == 0` is a silent noop. `num_threads == 0` and
  // `opts.top_k == 0` throw `std::invalid_argument` (the Python adapter
  // resolves `num_threads = 0` upstream).
  auto batch_search(const float *queries,
                    uint64_t n_queries,
                    const DiskSearchOptions &opts,
                    uint32_t num_threads,
                    uint64_t *out_labels,
                    float *out_distances) const -> void {
    if (opts.top_k == 0) {
      throw std::invalid_argument("DiskCollection: top_k must be > 0");
    }
    if (num_threads == 0) {
      throw std::invalid_argument(
          "DiskCollection::batch_search: num_threads must be > 0 (the Python "
          "adapter resolves num_threads = 0 before this entry)");
    }
    if (n_queries == 0) {
      return;
    }

    // Empty collection: every per-query search() would return an empty hits
    // vector, leaving the caller-pre-filled UINT64_MAX / NaN sentinels in
    // place. Spec point 7 requires "no exception, no allocation"; returning
    // here skips the worker-pool allocation and the std::thread spawns
    // entirely, so the caller's sentinels are observed without us touching
    // any heap or kernel.
    if (segments_.empty()) {
      return;
    }

    const uint64_t dim = static_cast<uint64_t>(manifest_.dim);
    const uint32_t top_k = opts.top_k;

    // Caller pre-fills the output buffers with sentinels (UINT64_MAX /
    // NaN); writing only the [0, hits.size()) prefix is what makes the
    // padding contract hold without an explicit sentinel pass. The
    // labels-only fast path is selected by `out_distances == nullptr`.
    auto write_to_output = [&](uint64_t i, const std::vector<DiskSearchHit> &hits) {
      const uint64_t base = i * top_k;
      const size_t n = std::min<size_t>(hits.size(), static_cast<size_t>(top_k));
      for (size_t j = 0; j < n; ++j) {
        out_labels[base + j] = hits[j].label;
      }
      if (out_distances != nullptr) {
        for (size_t j = 0; j < n; ++j) {
          out_distances[base + j] = hits[j].distance;
        }
      }
    };

    // num_threads == 1 runs on the calling thread and skips the std::thread
    // spawn/join + atomic dispatch overhead. The spec requires the resulting
    // out_labels / out_distances state to be byte-identical to the 1-worker
    // multi-thread path; both write the same hits in query-index order.
    if (num_threads == 1) {
      for (uint64_t i = 0; i < n_queries; ++i) {
        auto hits = search(queries + i * dim, opts);
        write_to_output(i, hits);
      }
      return;
    }

    // Multi-thread: per-query parallelism. workers share an atomic counter
    // that doles out query indices; each worker invokes the existing
    // single-query search() path on its slice. We deliberately do NOT
    // parallelize across segments inside a single search().
    //
    // Clamp workers to n_queries: a worker that immediately observes
    // fetch_add >= n_queries returns without doing useful work, so spawning
    // more than n_queries threads is wasted thread-spawn cost.
    const uint32_t worker_count = static_cast<uint32_t>(std::min<uint64_t>(num_threads, n_queries));
    std::atomic<uint64_t> next_query{0};
    std::atomic<bool> aborted{false};
    std::mutex error_mutex;
    std::exception_ptr first_error{nullptr};

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (uint32_t t = 0; t < worker_count; ++t) {
      workers.emplace_back([&, this]() {
        try {
          while (!aborted.load(std::memory_order_relaxed)) {
            const uint64_t i = next_query.fetch_add(1, std::memory_order_relaxed);
            if (i >= n_queries) {
              return;
            }
            auto hits = this->search(queries + i * dim, opts);
            write_to_output(i, hits);
          }
        } catch (...) {
          // Capture only the first exception; subsequent failures from
          // sibling workers are intentionally discarded so the caller
          // observes exactly one exception type matching the first
          // failure (spec contract 8). Setting `aborted` lets sibling
          // workers exit at the head of their next loop iteration.
          std::lock_guard<std::mutex> lk(error_mutex);
          if (!first_error) {
            first_error = std::current_exception();
          }
          aborted.store(true, std::memory_order_relaxed);
        }
      });
    }
    for (auto &w : workers) {
      w.join();
    }
    if (first_error) {
      std::rethrow_exception(first_error);
    }
  }

  // Returns the total number of FLUSHED rows. Pending rows are intentionally
  // excluded — see spec scenario "size() excludes pending rows".
  auto size() const -> uint64_t {
    uint64_t s = 0;
    for (const auto &seg : segments_) {
      s += seg->size();
    }
    return s;
  }

  auto dim() const -> uint32_t { return static_cast<uint32_t>(manifest_.dim); }

  // Tombstone API stubs (v1: not implemented).
  static void mark_deleted(uint64_t /*label*/) {
    throw std::runtime_error("DiskCollection: deletes not implemented in v1");
  }
  static auto is_deleted(uint64_t /*label*/) -> bool { return false; }

 private:
  DiskCollection() = default;

  void open_listed_segments() {
    segments_.reserve(manifest_.segment_ids.size());
    segment_dirs_.reserve(manifest_.segment_ids.size());
    for (const auto &id : manifest_.segment_ids) {
      const auto seg_dir = path_ / "segments" / id;
      // Reject segment directories that are themselves symlinks: a
      // symlink-swap of seg_*/ would otherwise redirect reads outside the
      // collection root — MMapFile's O_NOFOLLOW only protects leaf files.
      std::error_code ec;
      if (std::filesystem::is_symlink(seg_dir, ec)) {
        throw std::runtime_error("DiskCollection: segment directory is a symlink: " +
                                 seg_dir.string());
      }
      auto searcher = load_segment_from_manifest(seg_dir);
      segments_.push_back(std::move(searcher));
      segment_dirs_.push_back(seg_dir);
    }
  }

  void scan_orphans() {
    const auto segments_dir = path_ / "segments";
    std::error_code ec;
    if (!std::filesystem::is_directory(segments_dir, ec)) {
      return;
    }
    std::set<std::string> listed(manifest_.segment_ids.begin(), manifest_.segment_ids.end());
    uint64_t max_on_disk = 0;
    for (const auto &entry : std::filesystem::directory_iterator(segments_dir, ec)) {
      if (ec) {
        break;
      }
      const auto name = entry.path().filename().string();
      if (name.starts_with(".tmp_")) {
        // Stale tmp from an aborted flush — log as partial and skip.
        LOG_WARN("DiskCollection: orphan tmp dir at {} kind=partial", entry.path().string());
        continue;
      }
      if (!detail::is_valid_segment_id(name)) {
        continue;
      }
      if (listed.contains(name)) {
        // Listed; track its id for next-id calculation.
        const uint64_t id = std::stoull(name.substr(4));
        max_on_disk = std::max(max_on_disk, id);
        continue;
      }
      // Orphan segment. Classify.
      classify_and_log_orphan(entry.path());
      const uint64_t id = std::stoull(name.substr(4));
      max_on_disk = std::max(max_on_disk, id);
    }
    if (max_on_disk + 1 > manifest_.next_segment_id) {
      manifest_.next_segment_id = max_on_disk + 1;
    }
  }

  // For a disk_vamana orphan, peek at graph.index's header to read the
  // declared `expected_file_size` (bytes 0..7) without constructing a full
  // VamanaReader. The orphan path's "do not load" contract precludes the
  // reader's full structural validation; reading the 8-byte field is the
  // cheapest way to spot a truncated graph file. Returns the declared size,
  // or `std::nullopt` if the file is missing / unreadable / too short.
  static auto peek_graph_expected_size(const std::filesystem::path &graph_path)
      -> std::optional<uint64_t> {
    std::string buf;
    try {
      buf = ::alaya::platform::read_file_prefix(graph_path, sizeof(uint64_t));
    } catch (const std::exception &) {
      return std::nullopt;
    }
    uint64_t expected = 0;
    std::memcpy(&expected, buf.data(), sizeof(expected));
    return expected;
  }

  static auto stat_nonempty_regular_file(const std::filesystem::path &path) -> bool {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec) || ec) {
      return false;
    }
    const auto sz = std::filesystem::file_size(path, ec);
    return !ec && sz > 0;
  }

  static void classify_and_log_orphan(const std::filesystem::path &orphan_dir) {
    const auto manifest_path = orphan_dir / "manifest.txt";
    std::error_code ec;
    if (!std::filesystem::exists(manifest_path, ec) || ec) {
      LOG_WARN("DiskCollection: orphan segment at {} kind=partial (no manifest.txt)",
               orphan_dir.string());
      return;
    }
    SegmentManifest sm;
    try {
      sm = SegmentManifest::load(manifest_path);
    } catch (const std::exception &e) {
      LOG_WARN("DiskCollection: orphan segment at {} kind=partial (manifest unparsable: {})",
               orphan_dir.string(),
               e.what());
      return;
    }
    const auto ids_path = orphan_dir / sm.ids_file;
    const auto ids_size_actual = std::filesystem::file_size(ids_path, ec);
    if (ec) {
      LOG_WARN("DiskCollection: orphan segment at {} kind=truncated (ids stat failed)",
               orphan_dir.string());
      return;
    }
    std::optional<uint64_t> vec_size_actual;
    if (!sm.vectors_file.empty()) {
      const auto vec_path = orphan_dir / sm.vectors_file;
      vec_size_actual = std::filesystem::file_size(vec_path, ec);
      if (ec) {
        LOG_WARN("DiskCollection: orphan segment at {} kind=truncated (vectors stat failed)",
                 orphan_dir.string());
        return;
      }
    }
    const uint64_t expected_ids = sm.count * sizeof(uint64_t);
    const uint64_t expected_vec = sm.count * sm.dim * sizeof(float);
    if (ids_size_actual != expected_ids ||
        (vec_size_actual.has_value() && *vec_size_actual != expected_vec)) {
      LOG_WARN("DiskCollection: orphan segment at {} kind=truncated", orphan_dir.string());
      return;
    }
    // Engine-specific extension: Vamana requires graph.index to be present and
    // its declared expected_file_size to match the on-disk size. We read the
    // 8-byte header field rather than constructing a VamanaReader because the
    // orphan path is a "do not load" inspection — full validation is reserved
    // for load_segment_from_manifest at open time.
    if (sm.index_type == DiskIndexType::Vamana) {
      auto graph_it = sm.x_extras.find("x_graph_file");
      if (graph_it == sm.x_extras.end() || graph_it->second.empty()) {
        LOG_WARN("DiskCollection: orphan segment at {} kind=truncated (x_graph_file missing)",
                 orphan_dir.string());
        return;
      }
      const auto graph_path = orphan_dir / graph_it->second;
      const auto graph_size_actual = std::filesystem::file_size(graph_path, ec);
      if (ec) {
        LOG_WARN("DiskCollection: orphan segment at {} kind=truncated (graph.index stat failed)",
                 orphan_dir.string());
        return;
      }
      auto declared = peek_graph_expected_size(graph_path);
      if (!declared.has_value()) {
        LOG_WARN("DiskCollection: orphan segment at {} kind=truncated (graph header unreadable)",
                 orphan_dir.string());
        return;
      }
      if (*declared != static_cast<uint64_t>(graph_size_actual)) {
        LOG_WARN("DiskCollection: orphan segment at {} kind=truncated (graph size mismatch)",
                 orphan_dir.string());
        return;
      }
    }
    if (sm.index_type == DiskIndexType::Laser) {
      static constexpr const char *kRequiredLaserFiles[] = {
          "x_laser_index_file",
          "x_laser_rotator_file",
          "x_laser_cache_ids_file",
          "x_laser_cache_nodes_file",
      };
      static constexpr const char *kOptionalLaserFiles[] = {
          "x_laser_medoids_file",
          "x_laser_medoids_indices_file",
          "x_laser_pca_file",
      };
      for (const char *key : kRequiredLaserFiles) {
        const auto it = sm.x_extras.find(key);
        if (it == sm.x_extras.end() || it->second.empty() ||
            !stat_nonempty_regular_file(orphan_dir / it->second)) {
          LOG_WARN("DiskCollection: orphan segment at {} kind=truncated ({} missing or empty)",
                   orphan_dir.string(),
                   key);
          return;
        }
      }
      for (const char *key : kOptionalLaserFiles) {
        const auto it = sm.x_extras.find(key);
        if (it == sm.x_extras.end()) {
          continue;
        }
        if (it->second.empty() || !stat_nonempty_regular_file(orphan_dir / it->second)) {
          LOG_WARN("DiskCollection: orphan segment at {} kind=truncated ({} missing or empty)",
                   orphan_dir.string(),
                   key);
          return;
        }
      }
    }
    LOG_WARN("DiskCollection: orphan segment at {} kind=complete", orphan_dir.string());
  }

  std::filesystem::path path_;
  CollectionManifest manifest_;
  std::vector<std::shared_ptr<SegmentSearcher>> segments_;
  // Kept in lockstep with segments_ (same push order, same lifetime). Stored
  // explicitly so cross-segment label inventory can resolve each segment's
  // manifest.ids_file without depending on the runtime type of the searcher.
  std::vector<std::filesystem::path> segment_dirs_;
  std::vector<float> pending_vectors_;
  std::vector<uint64_t> pending_labels_;
  size_t max_pending_bytes_ = kDefaultMaxPendingBytes;
  VamanaSegmentBuildParams vamana_params_{};
  detail::LockFd lock_fd_;
};

}  // namespace alaya::disk
