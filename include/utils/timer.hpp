// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// borrowed from https://gist.github.com/tzutalin/fd0340a93bb8d998abb9
#pragma once
#include <chrono>

namespace alaya {
class Timer {
  using clock_ = std::chrono::high_resolution_clock;
  std::chrono::time_point<clock_> m_beg_;

 public:
  Timer() : m_beg_(clock_::now()) {}

  void reset() { m_beg_ = clock_::now(); }

  // returns elapsed time in `us`
  [[nodiscard]] auto elapsed() const -> uint64_t {
    return std::chrono::duration_cast<std::chrono::microseconds>(clock_::now() - m_beg_).count();
  }
  [[nodiscard]] auto elapsed_us() const -> double { return static_cast<double>(elapsed()); }
  [[nodiscard]] auto elapsed_ms() const -> double {
    return static_cast<double>(elapsed()) / 1000.0;
  }
  [[nodiscard]] auto elapsed_s() const -> double {
    return static_cast<double>(elapsed()) / 1'000'000.0;
  }
};

}  // namespace alaya
