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
  if (!ec) {
    return;
  }

  LOG_INFO("platform fallback: rename failed for {}, removing destination before retrying",
           to.string());
  fs::remove(to, ec);
  ec.clear();
  fs::rename(from, to, ec);
  if (ec) {
    throw std::runtime_error("Failed to atomically replace " + to.string());
  }
#endif
}

}  // namespace alaya::platform
