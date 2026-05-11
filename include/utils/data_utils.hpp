// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once
#include <cmath>
#include <cstddef>

namespace alaya {

inline auto cos_dist(const float *x, const float *y, size_t dim) -> float {
  float sum = 0;
  float x_norm = 0;
  float y_norm = 0;

  for (size_t i = 0; i < dim; ++i) {
    sum += x[i] * y[i];
    x_norm += x[i] * x[i];
    y_norm += y[i] * y[i];
  }
  return -sum / std::sqrt(x_norm * y_norm);
}

template <typename DataType = float>
inline void normalize(DataType *data, size_t dim) {
  float sum = 0;
  for (size_t i = 0; i < dim; ++i) {
    sum += data[i] * data[i];
  }
  sum = 1.0 / std::sqrt(sum);
  for (size_t i = 0; i < dim; ++i) {
    data[i] *= sum;
  }
}

}  // namespace alaya
