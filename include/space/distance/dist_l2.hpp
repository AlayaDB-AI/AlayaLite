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

#include <cstddef>
#include <cstdint>
#include "dist_config.hpp"
#include "utils/rabitq_utils/defines.hpp"

namespace alaya {
FAST_BEGIN
template <typename DataType = float, typename DistanceType = float>
inline auto l2_sqr_rabitq(const DataType *__restrict__ x,
                          const DataType *__restrict__ y,
                          size_t dim) -> DistanceType {
  return l2_sqr(x, y, dim);
}
FAST_END

FAST_BEGIN
template <typename DataType = float, typename DistanceType = float>
inline auto l2_sqr(const DataType *x, const DataType *y, size_t dim) -> DistanceType {
  DistanceType sum = 0;
  for (size_t i = 0; i < dim; ++i) {
    DistanceType diff = static_cast<float>(x[i]) - static_cast<float>(y[i]);
    sum += diff * diff;
  }
  return sum;
}
FAST_END

FAST_BEGIN
template <typename DataType = float, typename DistanceType = float>
inline auto l2_sqr_sq4(const uint8_t *encoded_x,
                       const uint8_t *encoded_y,
                       size_t dim,
                       const DataType *min,
                       const DataType *max) -> DistanceType {
  DistanceType sum = 0;

  for (size_t i = 0; i < dim; i += 2) {
    auto x_high = (encoded_x[i / 2] >> 4) & 0x0F;
    auto y_high = (encoded_y[i / 2] >> 4) & 0x0F;
    auto diff = (x_high - y_high) * (max[i] - min[i]) / 15.0F;
    sum += diff * diff;

    if (i + 1 != dim) {
      auto x_low = encoded_x[i / 2] & 0x0F;
      auto y_low = encoded_y[i / 2] & 0x0F;
      auto diff = (x_low - y_low) * (max[i + 1] - min[i + 1]) / 15.0F;
      sum += diff * diff;
    }
  }
  return sum;
}
FAST_END

FAST_BEGIN
template <typename DataType = float, typename DistanceType = float>
inline auto l2_sqr_sq8(const uint8_t *encoded_x,
                       const uint8_t *encoded_y,
                       size_t dim,
                       const DataType *min,
                       const DataType *max) -> DistanceType {
  DistanceType sum = 0;

  for (uint32_t i = 0; i < dim; i += 1) {
    auto x = encoded_x[i];
    auto y = encoded_y[i];
    auto diff = (max[i] - min[i]) * (x - y) / 255.0F;

    sum += diff * diff;
  }

  return sum;
}
FAST_END

}  // namespace alaya
