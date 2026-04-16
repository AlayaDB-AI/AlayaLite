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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <thread>

namespace alaya {

inline auto system_thread_count() -> uint32_t {
  auto thread_count = std::thread::hardware_concurrency();
  return thread_count == 0 ? 1U : thread_count;
}

inline auto configured_thread_limit() -> uint32_t {
  const char *value = std::getenv("ALAYA_MAX_THREADS");
  auto system_threads = system_thread_count();
  if (value == nullptr || *value == '\0') {
    return system_threads;
  }

  char *end = nullptr;
  auto parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' || parsed == 0) {
    return system_threads;
  }

  return std::min<uint32_t>(static_cast<uint32_t>(parsed), system_threads);
}

inline auto cap_thread_count(uint32_t requested) -> uint32_t {
  auto normalized = requested == 0 ? 1U : requested;
  return std::min(normalized, configured_thread_limit());
}

}  // namespace alaya
