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

#if defined(__SSE2__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace alaya {

/**
 * @brief Prefetches data to L1 cache for faster access.
 *
 * This function issues a prefetch hint to bring data into the L1 cache. The actual mechanism
 * depends on the compiler and platform. On systems with SSE2 support, the prefetch is done
 * using the `_mm_prefetch` intrinsic, specifically targeting L1 cache. For other systems,
 * it uses the `__builtin_prefetch` function, with a hint to bring the data into the L1 cache.
 *
 * @param address A pointer to the memory address to prefetch.
 */
inline auto prefetch_l1(const void *address) -> void {
#if defined(__SSE2__)
  _mm_prefetch((const char *)address, _MM_HINT_T0);
#else
  __builtin_prefetch(address, 0, 3);
#endif
}

/**
 * @brief Prefetches data to L2 cache for faster access.
 *
 * This function issues a prefetch hint to bring data into the L2 cache. On systems with SSE2
 * support, it uses the `_mm_prefetch` intrinsic with an L2 cache hint. For other systems, the
 * `__builtin_prefetch` function is used with an L2 cache hint.
 *
 * @param address A pointer to the memory address to prefetch.
 */
inline auto prefetch_l2(const void *address) -> void {
#if defined(__SSE2__)
  _mm_prefetch((const char *)address, _MM_HINT_T1);
#else
  __builtin_prefetch(address, 0, 2);
#endif
}

/**
 * @brief Prefetches data to L3 cache for faster access.
 *
 * This function issues a prefetch hint to bring data into the L3 cache. The behavior is platform
 * dependent: systems with SSE2 support use the `_mm_prefetch` intrinsic with an L3 cache hint,
 * while other systems use the `__builtin_prefetch` function with an L3 cache hint.
 *
 * @param address A pointer to the memory address to prefetch.
 */
inline auto prefetch_l3(const void *address) -> void {
#if defined(__SSE2__)
  _mm_prefetch((const char *)address, _MM_HINT_T2);
#else
  __builtin_prefetch(address, 0, 1);
#endif
}

/**
 * @brief Prefetches a block of data to L1 cache for more convenient access.
 *
 * This function prefetches a block of memory to L1 cache by issuing prefetch hints for each
 * 64-byte aligned memory address within the provided memory range. It calls the `prefetch_l1`
 * function in a loop for each 64-byte segment in the provided memory.
 *
 * @param address A pointer to the starting memory address to prefetch.
 * @param line The number of 64-byte memory lines to prefetch.
 */
inline auto mem_prefetch_l1(const void *address, uint32_t line) -> void {
  for (uint32_t i = 0; i < line; ++i) {
    prefetch_l1(static_cast<const char *>(address) + i * 64);
  }
}

/**
 * @brief Prefetches a block of data to L2 cache for more convenient access.
 *
 * Similar to `mem_prefetch_l1`, this function prefetches memory in 64-byte chunks, but it issues
 * prefetch hints to bring data into the L2 cache. It uses the `prefetch_l2` function for each
 * memory line.
 *
 * @param address A pointer to the starting memory address to prefetch.
 * @param line The number of 64-byte memory lines to prefetch.
 */
inline auto mem_prefetch_l2(const void *address, uint32_t line) -> void {
  for (uint32_t i = 0; i < line; ++i) {
    prefetch_l2(static_cast<const char *>(address) + i * 64);
  }
}

/**
 * @brief Prefetches a block of data to L3 cache for more convenient access.
 *
 * This function prefetches memory in 64-byte chunks, issuing prefetch hints to bring data into
 * the L3 cache by calling the `prefetch_l3` function for each 64-byte segment in the memory range.
 *
 * @param address A pointer to the starting memory address to prefetch.
 * @param line The number of 64-byte memory lines to prefetch.
 */
inline auto mem_prefetch_l3(const void *address, uint32_t line) -> void {
  for (uint32_t i = 0; i < line; ++i) {
    prefetch_l3(static_cast<const char *>(address) + i * 64);
  }
}
};  // namespace alaya
