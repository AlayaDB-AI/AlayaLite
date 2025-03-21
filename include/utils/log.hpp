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
#include <spdlog/spdlog.h>
#include <cstdarg>
#include <filesystem>
#include <string>

#ifdef PROJECT_ROOT
#define RELATIVE_FILE get_relative_path(__FILE__, PROJECT_ROOT)
auto inline get_relative_path(const std::string &full_path,
                              const std::string &base_path) -> std::string {
  std::filesystem::path full(full_path);
  std::filesystem::path base(base_path);
  return std::filesystem::relative(full, base).string();
}
#else
#define RELATIVE_FILE __FILE__
#endif

#define CONCATENATE_STRINGS(a, b) a b
#define LOG_TRACE(fmt, ...)                                                              \
  {                                                                                      \
    spdlog::trace(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", fmt), RELATIVE_FILE, __LINE__, \
                  ##__VA_ARGS__);                                                        \
  }
#define LOG_DEBUG(fmt, ...)                                                              \
  {                                                                                      \
    spdlog::debug(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", fmt), RELATIVE_FILE, __LINE__, \
                  ##__VA_ARGS__);                                                        \
  }
#define LOG_INFO(fmt, ...)                                                              \
  {                                                                                     \
    spdlog::info(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", fmt), RELATIVE_FILE, __LINE__, \
                 ##__VA_ARGS__);                                                        \
  }
#define LOG_WARN(fmt, ...)                                                              \
  {                                                                                     \
    spdlog::warn(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", fmt), RELATIVE_FILE, __LINE__, \
                 ##__VA_ARGS__);                                                        \
  }
#define LOG_ERROR(fmt, ...)                                                              \
  {                                                                                      \
    spdlog::error(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", fmt), RELATIVE_FILE, __LINE__, \
                  ##__VA_ARGS__);                                                        \
  }
#define LOG_CRITICAL(fmt, ...)                                                              \
  {                                                                                         \
    spdlog::critical(CONCATENATE_STRINGS("[Alaya] [{}:{}] ", fmt), RELATIVE_FILE, __LINE__, \
                     ##__VA_ARGS__);                                                        \
  }
