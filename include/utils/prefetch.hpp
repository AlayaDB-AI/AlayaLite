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

#include <cstdint>
#include <cstdio>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__SSE2__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace alaya {

/**
 * @brief Prefetches data to L1 cache for faster access.
 */
inline auto prefetch_l1(const void *address) -> void {
#if defined(_MSC_VER)
    _mm_prefetch((const char *)address, _MM_HINT_T0);
#elif defined(__SSE2__)
    _mm_prefetch((const char *)address, _MM_HINT_T0);
#else
    __builtin_prefetch(address, 0, 3);
#endif
}

/**
 * @brief Prefetches data to L2 cache for faster access.
 */
inline auto prefetch_l2(const void *address) -> void {
#if defined(_MSC_VER)
    _mm_prefetch((const char *)address, _MM_HINT_T1);
#elif defined(__SSE2__)
    _mm_prefetch((const char *)address, _MM_HINT_T1);
#else
    __builtin_prefetch(address, 0, 2);
#endif
}

/**
 * @brief Prefetches data to L3 cache for faster access.
 */
inline auto prefetch_l3(const void *address) -> void {
#if defined(_MSC_VER)
    _mm_prefetch((const char *)address, _MM_HINT_T2);
#elif defined(__SSE2__)
    _mm_prefetch((const char *)address, _MM_HINT_T2);
#else
    __builtin_prefetch(address, 0, 1);
#endif
}

/**
 * @brief Prefetches a block of data to L1 cache
 */
inline auto mem_prefetch_l1(const void *address, uint32_t line) -> void {
  for (uint32_t i = 0; i < line; ++i) {
    prefetch_l1(static_cast<const char *>(address) + i * 64);
  }
}

/**
 * @brief Prefetches a block of data to L2 cache
 */
inline auto mem_prefetch_l2(const void *address, uint32_t line) -> void {
  for (uint32_t i = 0; i < line; ++i) {
    prefetch_l2(static_cast<const char *>(address) + i * 64);
  }
}

/**
 * @brief Prefetches a block of data to L3 cache
 */
inline auto mem_prefetch_l3(const void *address, uint32_t line) -> void {
  for (uint32_t i = 0; i < line; ++i) {
    prefetch_l3(static_cast<const char *>(address) + i * 64);
  }
}
};  // namespace alaya
