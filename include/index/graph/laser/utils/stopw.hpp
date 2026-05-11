// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file stopw.hpp
 * @brief Simple stopwatch utility for timing code sections.
 */

#pragma once

#include <chrono>

/** @brief Stopwatch for measuring elapsed time in various units. */
class StopW {
  std::chrono::steady_clock::time_point time_begin_;

 public:
  StopW() { time_begin_ = std::chrono::steady_clock::now(); }

  [[nodiscard]] float get_elapsed_sec() const {
    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::seconds>(time_end - time_begin_).count());
  }

  [[nodiscard]] float get_elapsed_mili() const {
    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_begin_).count());
  }

  [[nodiscard]] float get_elapsed_micro() const {
    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin_).count());
  }

  [[nodiscard]] float get_elapsed_nano() const {
    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin_).count());
  }

  void reset() { time_begin_ = std::chrono::steady_clock::now(); }
};
