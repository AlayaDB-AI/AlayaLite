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

#include <ctime>
#include <stdexcept>
#include <type_traits>

// NOLINTBEGIN
namespace alaya {
inline size_t floor_log2(size_t x) {
  size_t ret = 0;
  while (x > 1) {
    ret++;
    x >>= 1;
  }
  return ret;
}

inline size_t ceil_log2(size_t x) {
  size_t ret = floor_log2(x);
  return (1UL << ret) < x ? ret + 1 : ret;
}

template <typename T>
constexpr T ceil_round_up(T x, T divisor) {
  if (divisor <= 0) {
    throw std::invalid_argument("Invalid divisor! Divisor input should be greater than 0.");
  }
  static_assert(std::is_integral_v<T>, "T must be an integral type");
  return (x / divisor) + static_cast<T>((x % divisor) != 0);
}

template <typename T>
constexpr T round_up_to_multiple_of(size_t x, size_t multiple_of) {
  return multiple_of * (ceil_round_up(x, multiple_of));
}
// NOLINTEND
}  // namespace alaya
