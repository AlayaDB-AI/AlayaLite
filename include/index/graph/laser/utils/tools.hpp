// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file tools.hpp
 * @brief General utility functions for memory alignment and random number generation.
 */

#pragma once

#include <ctime>
#include <random>
#include <thread>

namespace alaya::laser {

/** @brief Thread-safe random integer generator in range [min, max]. */
template <typename T>
inline T rand_integer(T min, T max) {
  // Brace-init avoids the most-vexing-parse: with parens, gcc tries to parse
  // the initializer as a function declarator and reads `std::random_device()()`
  // as "function returning function".
  // NOLINTNEXTLINE(whitespace/braces)
  static thread_local std::mt19937 generator{
      std::random_device{}() +  // NOLINT(whitespace/braces)
      std::hash<std::thread::id>{}(std::this_thread::get_id())};
  std::uniform_int_distribution<T> distribution(min, max);
  return distribution(generator);
}

constexpr size_t div_round_up(size_t val, size_t div) {
  return (val / div) + static_cast<size_t>((val % div) != 0);
}

constexpr size_t round_up_to_multiple(size_t val, size_t multiple_of) {
  return multiple_of * (div_round_up(val, multiple_of));
}

inline size_t ceil_log2(size_t val) {
  size_t res = 0;
  for (size_t i = 0; i < 31; ++i) {
    if ((1U << i) >= val) {
      res = i;
      break;
    }
  }
  return res;
}

/** @brief Returns hardware thread count (or 1 if unavailable). */
inline size_t total_threads() {
  const auto threads = std::thread::hardware_concurrency();
  return threads == 0 ? 1 : threads;
}
}  // namespace alaya::laser
