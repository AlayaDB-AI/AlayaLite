// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <stdexcept>
#include <string>
#include <system_error>

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <fcntl.h>
  #include <io.h>
  #include <sys/stat.h>
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <unistd.h>
  #if defined(__linux__)
    // <linux/fs.h> defines a top-level BLOCK_SIZE macro that pollutes any TU
    // that includes us; undef immediately to avoid breaking third-party
    // headers (notably moodycamel/concurrentqueue.h which has a struct
    // member named BLOCK_SIZE).
    #include <linux/fs.h>     // RENAME_NOREPLACE
    #include <sys/syscall.h>  // SYS_renameat2
    #ifdef BLOCK_SIZE
      #undef BLOCK_SIZE
    #endif
  #endif
#endif

#include "utils/log.hpp"

namespace alaya::platform {

namespace fs = std::filesystem;

#ifdef _WIN32
using native_fd_t = HANDLE;
#else
using native_fd_t = int;
#endif

// Process id wrapper. POSIX `getpid()` returns `pid_t` (commonly 32-bit);
// Windows `GetCurrentProcessId()` returns `DWORD` (also 32-bit). Both fit in
// int64 for stable serialization across the disk-format manifests.
inline auto get_pid() noexcept -> std::int64_t {
#ifdef _WIN32
  return static_cast<std::int64_t>(::GetCurrentProcessId());
#else
  return static_cast<std::int64_t>(::getpid());
#endif
}

// Check if `path` exists, is a regular file, and is readable by the current
// process. Routes through `access(R_OK)` on POSIX and `_waccess(R_OK)` on
// Windows. Returns false on any failure (missing, directory, no permission).
inline auto is_readable_regular_file(const fs::path &path) noexcept -> bool {
  std::error_code ec;
  if (!fs::is_regular_file(path, ec) || ec) {
    return false;
  }
#ifdef _WIN32
  // _waccess mode bits: 4 = read, 2 = write, 6 = read+write.
  return _waccess(path.c_str(), 4) == 0;
#else
  return ::access(path.c_str(), R_OK) == 0;
#endif
}

inline auto create_directories_if_needed(const fs::path &path) -> void {
  if (path.empty()) {
    return;
  }
  fs::create_directories(path);
}

inline auto file_size_or(const fs::path &path, std::uintmax_t fallback) noexcept -> std::uintmax_t {
  std::error_code ec;
  const auto bytes = fs::file_size(path, ec);
  return ec ? fallback : bytes;
}

inline auto sync_file(const fs::path &path) -> void {
  if (path.empty() || !fs::exists(path)) {
    return;
  }

#ifdef _WIN32
  int fd = _wopen(path.c_str(), _O_RDONLY | _O_BINARY);
  if (fd < 0) {
    return;
  }
  [[maybe_unused]] auto result = _commit(fd);
  _close(fd);
#else
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return;
  }
  [[maybe_unused]] auto result = ::fsync(fd);
  ::close(fd);
#endif
}

// Strict variant of sync_file: throws on any syscall failure. Used by the
// segment builders where the fsync step is part of the durability contract
// — a silent best-effort fsync would let a corrupt segment be published.
inline auto sync_file_or_throw(const fs::path &path) -> void {
#ifdef _WIN32
  HANDLE h = ::CreateFileW(path.c_str(),
                           GENERIC_READ | GENERIC_WRITE,
                           FILE_SHARE_READ,
                           nullptr,
                           OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL,
                           nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("sync_file_or_throw: CreateFileW failed: " + path.string() +
                             ": Win32 error " + std::to_string(::GetLastError()));
  }
  const bool ok = ::FlushFileBuffers(h) != 0;
  const auto saved = ok ? 0UL : ::GetLastError();
  ::CloseHandle(h);
  if (!ok) {
    throw std::runtime_error("sync_file_or_throw: FlushFileBuffers failed: " + path.string() +
                             ": Win32 error " + std::to_string(saved));
  }
#else
  int flags = O_RDONLY;
  #ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
  #endif
  #ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
  #endif
  int fd = ::open(path.c_str(), flags);
  if (fd < 0) {
    throw std::runtime_error("sync_file_or_throw: open failed: " + path.string() + ": " +
                             std::strerror(errno));
  }
  if (::fsync(fd) != 0) {
    int saved = errno;
    ::close(fd);
    throw std::runtime_error("sync_file_or_throw: fsync failed: " + path.string() + ": " +
                             std::strerror(saved));
  }
  if (::close(fd) != 0) {
    throw std::runtime_error("sync_file_or_throw: close failed: " + path.string() + ": " +
                             std::strerror(errno));
  }
#endif
}

inline auto sync_directory(const fs::path &path) -> void {
  if (path.empty() || !fs::exists(path)) {
    return;
  }

#ifdef _WIN32
  LOG_INFO_ONCE(
      "platform fallback: directory sync is unavailable on Windows, continuing with best-effort "
      "semantics");
#else
  int flags = O_RDONLY;
  #ifdef O_DIRECTORY
  flags |= O_DIRECTORY;
  #endif
  int fd = ::open(path.c_str(), flags);
  if (fd < 0) {
    return;
  }
  [[maybe_unused]] auto result = ::fsync(fd);
  ::close(fd);
#endif
}

// Strict variant of sync_directory: throws on failure. POSIX opens the
// directory with O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC and fsyncs the handle.
// Windows has no equivalent operation — fsyncing a directory is a no-op
// concept under NTFS — so the Windows branch is a no-throw best-effort.
inline auto sync_directory_or_throw(const fs::path &dir) -> void {
#ifdef _WIN32
  (void)dir;
  // NTFS does not expose a fsync-equivalent for directories. Metadata is
  // journaled separately and MOVEFILE_WRITE_THROUGH on the prior rename is
  // the closest analogue. Document as a no-op rather than throw.
#else
  int flags = O_RDONLY | O_NOFOLLOW;
  #ifdef O_DIRECTORY
  flags |= O_DIRECTORY;
  #endif
  #ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
  #endif
  int fd = ::open(dir.c_str(), flags);
  if (fd < 0) {
    throw std::runtime_error("platform_fs::sync_directory_or_throw open failed: " + dir.string() +
                             ": " + std::strerror(errno));
  }
  if (::fsync(fd) != 0) {
    int saved = errno;
    ::close(fd);
    throw std::runtime_error("platform_fs::sync_directory_or_throw fsync failed: " + dir.string() +
                             ": " + std::strerror(saved));
  }
  if (::close(fd) != 0) {
    throw std::runtime_error("platform_fs::sync_directory_or_throw close failed: " + dir.string() +
                             ": " + std::strerror(errno));
  }
#endif
}

inline auto atomic_replace(const fs::path &from, const fs::path &to) -> void {
  create_directories_if_needed(to.parent_path());

#ifdef _WIN32
  if (::MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ==
      0) {
    throw std::runtime_error("Failed to atomically replace " + to.string());
  }
#else
  std::error_code ec;
  fs::rename(from, to, ec);
  if (ec) {
    throw std::runtime_error("Failed to atomically replace " + to.string() + ": " + ec.message());
  }
#endif
}

// Atomic rename that REFUSES to overwrite an existing destination. Used by
// the segment-publish path where double-publish is a correctness bug.
//
// - Linux: `renameat2(RENAME_NOREPLACE)` when available; otherwise lstat
//   pre-check + `rename()` (best-effort race-prone fallback).
// - POSIX fallback (macOS): lstat pre-check + `rename()` because the BSD
//   `renamex_np(RENAME_EXCL)` syscall is not universally available on the
//   GitHub macos-15 runners.
// - Windows: `MoveFileExW` without `MOVEFILE_REPLACE_EXISTING`. The Win32
//   call fails with `ERROR_ALREADY_EXISTS` when destination is present.
inline auto atomic_replace_no_overwrite(const fs::path &from, const fs::path &to) -> void {
#ifdef _WIN32
  if (::MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_WRITE_THROUGH) == 0) {
    const auto err = ::GetLastError();
    if (err == ERROR_ALREADY_EXISTS || err == ERROR_FILE_EXISTS) {
      throw std::runtime_error("atomic_replace_no_overwrite: target already exists: " +
                               to.string());
    }
    throw std::runtime_error("atomic_replace_no_overwrite: MoveFileExW failed: " + from.string() +
                             " -> " + to.string() + ": Win32 error " + std::to_string(err));
  }
#elif defined(__linux__) && defined(SYS_renameat2) && defined(RENAME_NOREPLACE)
  auto rc = ::syscall(SYS_renameat2,
                      AT_FDCWD,
                      from.c_str(),
                      AT_FDCWD,
                      to.c_str(),
                      static_cast<unsigned int>(RENAME_NOREPLACE));
  if (rc != 0) {
    int saved = errno;
    if (saved == EEXIST) {
      throw std::runtime_error("atomic_replace_no_overwrite: target already exists: " +
                               to.string());
    }
    throw std::runtime_error("atomic_replace_no_overwrite: renameat2 failed: " + from.string() +
                             " -> " + to.string() + ": " + std::strerror(saved));
  }
#else
  struct ::stat st{};
  if (::lstat(to.c_str(), &st) == 0) {
    throw std::runtime_error("atomic_replace_no_overwrite: target already exists: " + to.string());
  }
  if (errno != ENOENT) {
    int saved = errno;
    throw std::runtime_error("atomic_replace_no_overwrite: precheck failed: " + to.string() + ": " +
                             std::strerror(saved));
  }
  if (::rename(from.c_str(), to.c_str()) != 0) {
    int saved = errno;
    throw std::runtime_error("atomic_replace_no_overwrite: rename failed: " + from.string() +
                             " -> " + to.string() + ": " + std::strerror(saved));
  }
#endif
}

// Truncate an open file to the given length. POSIX uses ftruncate(fd);
// Windows uses SetFilePointerEx + SetEndOfFile on a HANDLE.
inline auto truncate_open_file(native_fd_t handle, std::uint64_t length) noexcept -> bool {
#ifdef _WIN32
  LARGE_INTEGER li;
  li.QuadPart = static_cast<LONGLONG>(length);
  if (!::SetFilePointerEx(handle, li, nullptr, FILE_BEGIN)) {
    return false;
  }
  return ::SetEndOfFile(handle) != 0;
#else
  return ::ftruncate(handle, static_cast<off_t>(length)) == 0;
#endif
}

// Read the first `prefix_bytes` of a regular file. Throws if the file is a
// symlink, missing, non-regular, or smaller than `prefix_bytes`. Used by
// header-style metadata reads where the consumer only needs the first few
// bytes (e.g., 8-byte u64 count) and the file body can be GB-scale.
inline auto read_file_prefix(const fs::path &path, std::size_t prefix_bytes) -> std::string {
  std::error_code ec;
  if (fs::is_symlink(path, ec)) {
    throw std::invalid_argument("read_file_prefix: refusing symlink: " + path.string());
  }
  if (!fs::is_regular_file(path, ec) || ec) {
    throw std::invalid_argument("read_file_prefix: not a regular file: " + path.string());
  }
#ifdef _WIN32
  FILE *fp = _wfopen(path.c_str(), L"rb");
  if (fp == nullptr) {
    throw std::runtime_error("read_file_prefix: _wfopen failed: " + path.string());
  }
  std::string buf(prefix_bytes, '\0');
  const auto got = std::fread(buf.data(), 1, prefix_bytes, fp);
  std::fclose(fp);
  if (got != prefix_bytes) {
    throw std::runtime_error("read_file_prefix: short read: " + path.string() + " (got " +
                             std::to_string(got) + "/" + std::to_string(prefix_bytes) + ")");
  }
  return buf;
#else
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    throw std::runtime_error("read_file_prefix: open failed: " + path.string() + ": " +
                             std::strerror(errno));
  }
  std::string buf(prefix_bytes, '\0');
  std::size_t total = 0;
  while (total < buf.size()) {
    ssize_t n = ::read(fd, buf.data() + total, buf.size() - total);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      int saved = errno;
      ::close(fd);
      throw std::runtime_error("read_file_prefix: read failed: " + path.string() + ": " +
                               std::strerror(saved));
    }
    if (n == 0) {
      break;
    }
    total += static_cast<std::size_t>(n);
  }
  ::close(fd);
  if (total != buf.size()) {
    throw std::runtime_error("read_file_prefix: short read: " + path.string());
  }
  return buf;
#endif
}

// Read a regular file into a std::string with a size cap. Refuses symlinks
// and non-regular files. Returns the file contents (no trailing data).
// Portable replacement for the POSIX-only O_NOFOLLOW + fstat + read pattern
// used by manifest readers; the symlink check is via std::filesystem which
// has a tiny TOCTOU window (acceptable for internal config files).
inline auto read_regular_file_bounded(const fs::path &path, std::size_t max_bytes) -> std::string {
  std::error_code ec;
  if (fs::is_symlink(path, ec)) {
    throw std::invalid_argument("read_regular_file_bounded: refusing symlink: " + path.string());
  }
  if (!fs::is_regular_file(path, ec) || ec) {
    throw std::invalid_argument("read_regular_file_bounded: not a regular file: " + path.string());
  }
  const auto size = fs::file_size(path, ec);
  if (ec) {
    throw std::runtime_error("read_regular_file_bounded: file_size failed: " + path.string() +
                             ": " + ec.message());
  }
  if (size > max_bytes) {
    throw std::invalid_argument("read_regular_file_bounded: file too large: " + path.string() +
                                " (" + std::to_string(size) + " bytes, max " +
                                std::to_string(max_bytes) + ")");
  }
#ifdef _WIN32
  // _wfopen + fread keeps wide-char paths intact end-to-end on Windows.
  FILE *fp = _wfopen(path.c_str(), L"rb");
  if (fp == nullptr) {
    throw std::runtime_error("read_regular_file_bounded: _wfopen failed: " + path.string());
  }
  std::string buf(size, '\0');
  const auto got = std::fread(buf.data(), 1, size, fp);
  std::fclose(fp);
  if (got != size) {
    throw std::runtime_error("read_regular_file_bounded: short read: " + path.string());
  }
  return buf;
#else
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    throw std::runtime_error("read_regular_file_bounded: open failed: " + path.string() + ": " +
                             std::strerror(errno));
  }
  std::string buf(size, '\0');
  std::size_t total = 0;
  while (total < buf.size()) {
    ssize_t n = ::read(fd, buf.data() + total, buf.size() - total);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      int saved = errno;
      ::close(fd);
      throw std::runtime_error("read_regular_file_bounded: read failed: " + path.string() + ": " +
                               std::strerror(saved));
    }
    if (n == 0) {
      break;
    }
    total += static_cast<std::size_t>(n);
  }
  ::close(fd);
  if (total != buf.size()) {
    throw std::runtime_error("read_regular_file_bounded: short read: " + path.string());
  }
  return buf;
#endif
}

// Write the full buffer to `path` (overwriting), fsync, then close. Used by
// the segment writers where durability is a hard requirement. Throws on
// any failure with a path-tagged message.
inline auto write_all_fsync(const fs::path &path, const void *data, std::size_t bytes) -> void {
#ifdef _WIN32
  HANDLE h = ::CreateFileW(path.c_str(),
                           GENERIC_WRITE,
                           0,  // exclusive
                           nullptr,
                           CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL,
                           nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("platform_fs::write_all_fsync CreateFileW failed: " + path.string() +
                             ": Win32 error " + std::to_string(::GetLastError()));
  }
  const auto *p = static_cast<const char *>(data);
  std::size_t written = 0;
  while (written < bytes) {
    DWORD chunk = static_cast<DWORD>(
        std::min<std::size_t>(bytes - written, static_cast<std::size_t>(0x7FFFFFFFU)));
    DWORD got = 0;
    if (!::WriteFile(h, p + written, chunk, &got, nullptr)) {
      const auto saved = ::GetLastError();
      ::CloseHandle(h);
      throw std::runtime_error("platform_fs::write_all_fsync WriteFile failed: " + path.string() +
                               ": Win32 error " + std::to_string(saved));
    }
    if (got == 0) {
      ::CloseHandle(h);
      throw std::runtime_error("platform_fs::write_all_fsync WriteFile returned 0 bytes: " +
                               path.string());
    }
    written += got;
  }
  if (!::FlushFileBuffers(h)) {
    const auto saved = ::GetLastError();
    ::CloseHandle(h);
    throw std::runtime_error("platform_fs::write_all_fsync FlushFileBuffers failed: " +
                             path.string() + ": Win32 error " + std::to_string(saved));
  }
  if (!::CloseHandle(h)) {
    throw std::runtime_error("platform_fs::write_all_fsync CloseHandle failed: " + path.string());
  }
#else
  int open_flags = O_WRONLY | O_CREAT | O_TRUNC;
  #ifdef O_CLOEXEC
  open_flags |= O_CLOEXEC;
  #endif
  #ifdef O_NOFOLLOW
  open_flags |= O_NOFOLLOW;
  #endif
  int fd = ::open(path.c_str(), open_flags, 0600);
  if (fd < 0) {
    throw std::runtime_error("platform_fs::write_all_fsync open failed: " + path.string() + ": " +
                             std::strerror(errno));
  }
  const auto *p = static_cast<const char *>(data);
  std::size_t written = 0;
  while (written < bytes) {
    ssize_t n = ::write(fd, p + written, bytes - written);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      int saved = errno;
      ::close(fd);
      throw std::runtime_error("platform_fs::write_all_fsync write failed: " + path.string() +
                               ": " + std::strerror(saved));
    }
    written += static_cast<std::size_t>(n);
  }
  if (::fsync(fd) != 0) {
    int saved = errno;
    ::close(fd);
    throw std::runtime_error("platform_fs::write_all_fsync fsync failed: " + path.string() + ": " +
                             std::strerror(saved));
  }
  if (::close(fd) != 0) {
    throw std::runtime_error("platform_fs::write_all_fsync close failed: " + path.string() + ": " +
                             std::strerror(errno));
  }
#endif
}

}  // namespace alaya::platform
