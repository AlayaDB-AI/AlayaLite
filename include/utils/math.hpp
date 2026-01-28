/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Copyright 2025 AlayaDB.AI
 */

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

// If it's C++20, use the standard library's masterpieces directly.
#if __cplusplus >= 202002L
  #include <bit>
#endif

#include "platform.hpp"

namespace alaya::math {

// ============================================================================
// 1. Bit Manipulation (Log2) - O(1)
// ============================================================================

/**
 * @brief calc floor(log2(x))
 * use hardware speedup (LZCNT / BSR)
 *
 * @param x input number
 * @return floor(log2(x))
 *
 * @example floor_log2(7) = 2, floor_log2(8) = 3
 */
template <typename T>
[[nodiscard]] constexpr auto floor_log2(T x) noexcept -> uint32_t {
  static_assert(std::is_integral_v<T>, "T must be integral");

  if (x == 0) {
    return 0;  // undefined behavior protection
  }

#if __cplusplus >= 202002L
  // C++20 standard library implementation
  return std::bit_width(static_cast<std::make_unsigned_t<T>>(x)) - 1;
#else
  // Legacy C++17/14 realization
  // GCC / Clang
  #if defined(__GNUC__) || defined(__clang__)
  if constexpr (sizeof(T) <= sizeof(unsigned int)) {
    return 31 - __builtin_clz(x);
  } else {
    return 63 - __builtin_clzll(x);
  }
  // MSVC
  #elif defined(_MSC_VER)
  unsigned long index;  // NOLINT(runtime/int) - MSVC _BitScanReverse requires unsigned long*
  if constexpr (sizeof(T) <= 4) {
    _BitScanReverse(&index, x);
  } else {
    #ifdef _WIN64
    _BitScanReverse64(&index, x);
    #else
    // 32-bit environment fallback for 64-bit integers implies manual split
    if (_BitScanReverse(&index, x >> 32)) return index + 32;
    _BitScanReverse(&index, x & 0xFFFFFFFF);
    #endif
  }
  return index;
  #else
  // Fallback: simple loop (not optimal)
  int ret = 0;
  while (x > 1) {
    ret++;
    x >>= 1;
  }
  return ret;
  #endif
#endif
}

/**
 * @brief calc ceil(log2(x))
 *
 * @param x input number
 * @return ceil(log2(x))
 *
 * @example ceil_log2(7) = 3, ceil_log2(8) = 3
 */
template <typename T>
[[nodiscard]] constexpr auto ceil_log2(T x) noexcept -> uint32_t {
  static_assert(std::is_integral_v<T>, "T must be integral");
  if (x <= 1) {
    return 0;
  }

  // if x > 1, ceil_log2(x) = floor_log2(x - 1) + 1
  return floor_log2(x - 1) + 1;
}

// ============================================================================
// 2. Alignment Utilities
// ============================================================================

/**
 * @brief general ceil division
 * Note: uses division, which is relatively slow.
 *
 * @param x dividend
 * @param divisor divisor
 * @return ceil(x / divisor)
 *
 * @throws std::invalid_argument if divisor is 0
 *
 * @example ceil_div(7, 3) = 3
 */
template <typename T>
[[nodiscard]] constexpr auto ceil_div(T x, T divisor) -> T {
  if (divisor == 0) {
    throw std::invalid_argument("Divisor cannot be 0");
  }
  return (x + divisor - 1) / divisor;
}

/**
 * @brief general round-up to the nearest multiple of divisor
 * Note: uses division, which is relatively slow.
 *
 * @param x value to be aligned
 * @param divisor alignment base
 * @return rounded-up value
 *
 * @example round_up_general(7, 3) = 9
 */
[[nodiscard]] constexpr auto round_up_general(size_t x, size_t divisor) -> size_t {
  if (divisor == 0) {
    return 0;
  }
  return ((x + divisor - 1) / divisor) * divisor;
}

/**
 * @brief high performance round-up to the nearest multiple of alignment (power of 2)
 * Compared to general modulo operations, bitwise operations are much faster.
 * @param x value to be aligned
 * @param alignment alignment base (must be a power of 2, e.g., 64, 4096)
 * @return rounded-up value
 *
 * @example round_up_pow2(70, 64) = 128
 */
template <typename T>
  requires std::is_integral_v<T>
[[nodiscard]] constexpr auto round_up_pow2(T x, size_t alignment) noexcept -> T {
  // assert((alignment & (alignment - 1)) == 0); // assert can be added in debug mode
  return (x + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief check if a number is a power of two
 *
 * @param x input number
 * @return true if x is a power of two
 *
 * @example is_power_of_two(8) = true, is_power_of_two(10) = false
 */
template <typename T>
  requires std::is_integral_v<T>
[[nodiscard]] constexpr auto is_power_of_two(T x) noexcept -> bool {
  return x > 0 && (x & (x - 1)) == 0;
}

}  // namespace alaya::math
