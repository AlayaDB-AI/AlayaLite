/*
 * Copyright 2025 VectorDB.NTU
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

#pragma once

#include <chrono>

namespace alaya {
class StopW {
  std::chrono::steady_clock::time_point time_begin_;

 public:
  StopW() { time_begin_ = std::chrono::steady_clock::now(); }

  [[nodiscard]] auto get_elapsed_sec() const -> float {
    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::seconds>(time_end - time_begin_).count());
  }

  [[nodiscard]] auto get_elapsed_mili() const -> float {
    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_begin_).count());
  }

  [[nodiscard]] auto get_elapsed_micro() const -> float {
    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin_).count());
  }

  [[nodiscard]] auto get_elapsed_nano() const -> float {
    std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_begin_).count());
  }

  void reset() { time_begin_ = std::chrono::steady_clock::now(); }
};
}  // namespace alaya
