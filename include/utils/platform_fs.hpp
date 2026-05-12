// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <filesystem>  // NOLINT(build/c++17)
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <fcntl.h>
  #include <io.h>
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <unistd.h>
#endif

#include "utils/log.hpp"

namespace alaya::platform {

namespace fs = std::filesystem;

inline auto create_directories_if_needed(const fs::path &path) -> void {
  if (path.empty()) {
    return;
  }
  fs::create_directories(path);
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

}  // namespace alaya::platform
