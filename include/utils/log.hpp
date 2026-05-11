// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>
#include <cstdarg>
#include <cstring>

/**
 * @brief Extract relative path from full path.
 *
 * Looks for common source directories (include/, src/, tests/) in the path
 * and returns the path starting from there. This ensures consistent log output
 * regardless of the build directory location.
 *
 * Examples:
 *   /tmp/build-xxx/alayalite/include/utils/log.hpp -> include/utils/log.hpp
 *   /home/user/project/src/main.cpp -> src/main.cpp
 */
inline auto extract_relative_path(const char *full_path) -> const char * {
  // Look for common source directory markers
  const char *pos = std::strstr(full_path, "/include/");
  if (pos != nullptr) {
    return pos + 1;
  }
  pos = std::strstr(full_path, "/src/");
  if (pos != nullptr) {
    return pos + 1;
  }
  pos = std::strstr(full_path, "/tests/");
  if (pos != nullptr) {
    return pos + 1;
  }
  // Fallback: return just the filename
  const char *last_slash = std::strrchr(full_path, '/');
  return last_slash != nullptr ? last_slash + 1 : full_path;
}

#define RELATIVE_FILE extract_relative_path(__FILE__)

#define CONCATENATE_STRINGS(a, b) a b
#define LOG_TRACE(msg, ...)                                                     \
  {                                                                             \
    spdlog::trace(::fmt::runtime(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", msg)), \
                  RELATIVE_FILE,                                                \
                  __LINE__,                                                     \
                  ##__VA_ARGS__);                                               \
  }
#define LOG_DEBUG(msg, ...)                                                     \
  {                                                                             \
    spdlog::debug(::fmt::runtime(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", msg)), \
                  RELATIVE_FILE,                                                \
                  __LINE__,                                                     \
                  ##__VA_ARGS__);                                               \
  }
#define LOG_INFO(msg, ...)                                                     \
  {                                                                            \
    spdlog::info(::fmt::runtime(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", msg)), \
                 RELATIVE_FILE,                                                \
                 __LINE__,                                                     \
                 ##__VA_ARGS__);                                               \
  }
#define LOG_WARN(msg, ...)                                                     \
  {                                                                            \
    spdlog::warn(::fmt::runtime(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", msg)), \
                 RELATIVE_FILE,                                                \
                 __LINE__,                                                     \
                 ##__VA_ARGS__);                                               \
  }
#define LOG_ERROR(msg, ...)                                                     \
  {                                                                             \
    spdlog::error(::fmt::runtime(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", msg)), \
                  RELATIVE_FILE,                                                \
                  __LINE__,                                                     \
                  ##__VA_ARGS__);                                               \
  }
#define LOG_CRITICAL(msg, ...)                                                     \
  {                                                                                \
    spdlog::critical(::fmt::runtime(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", msg)), \
                     RELATIVE_FILE,                                                \
                     __LINE__,                                                     \
                     ##__VA_ARGS__);                                               \
  }
