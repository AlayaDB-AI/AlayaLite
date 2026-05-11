// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

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
