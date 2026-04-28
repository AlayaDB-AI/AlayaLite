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
  static thread_local std::mt19937 generator(
      std::random_device()() + std::hash<std::thread::id>()(std::this_thread::get_id()));
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
