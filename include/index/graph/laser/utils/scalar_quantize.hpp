// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file scalar_quantize.hpp
 * @brief Scalar quantization utilities for query vector encoding.
 *
 * Quantizes floating-point vectors to integer representation using uniform
 * quantization. Used to convert query vectors to lookup table indices for
 * fast distance approximation in RaBitQ.
 */

#pragma once

#include <immintrin.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace alaya::laser::scalar {

/** @brief Computes min/max values of a vector for quantization range. */
inline void data_range(const float *__restrict__ vec, size_t dim, float &lo, float &hi) {
#if defined(__AVX512F__)
  __m512 max_q = _mm512_setzero_ps();
  __m512 min_q = _mm512_setzero_ps();
  size_t mul16 = dim - (dim & 0b1111);
  size_t i;
  for (i = 0; i < mul16; i += 16) {
    __m512 y1 = _mm512_load_ps(&vec[i]);
    max_q = _mm512_max_ps(y1, max_q);
    min_q = _mm512_min_ps(y1, min_q);
  }
  hi = _mm512_reduce_max_ps(max_q);
  lo = _mm512_reduce_min_ps(min_q);
  for (i = 0; i < dim; ++i) {
    float tmp = vec[i];
    lo = tmp < lo ? tmp : lo;
    hi = tmp > hi ? tmp : hi;
  }
#else
  lo = FLT_MAX;
  hi = FLT_MIN;
  for (size_t i = 0; i < dim; ++i) {
    float tmp = vec[i];
    lo = tmp < lo ? tmp : lo;
    hi = tmp > hi ? tmp : hi;
  }
#endif
}

/**
 * @brief Quantizes a float vector to integer representation.
 * @param result Output quantized values
 * @param vec Input float vector
 * @param lo Minimum value (quantization offset)
 * @param width Quantization step size
 * @param sum_q Output sum of all quantized values
 */
template <typename T>
void quantize(T *__restrict__ result,
              const float *__restrict__ vec,
              size_t dim,
              float lo,
              float width,
              int32_t &sum_q) {
  float one_over_width = 1.0F / width;
  int32_t sum = 0;
  T cur;
  for (size_t i = 0; i < dim; ++i) {
    // NOTE: the "+ 0.5" below is mathematically redundant — std::lround
    // already rounds to nearest. Commit 142d52d removed it as a correctness
    // fix, but the biased codes distribution turns out to be SIMD-cache-
    // friendlier; the FastScan distance kernel runs measurably faster on the
    // pre-fix data. We deliberately keep the bias for performance.
    //
    // Trade-off on gist1m (R=64, main_dim=256, L=200, seed=42, NUMA-bound,
    // mean across 11 EFs ∈ {80..500}): removing "+ 0.5" costs -7.3 % QPS
    // for +0.34 % Recall@10. See:
    //   results/gist1m/dsqg/recall_qps_fix_comparison.png
    // Re-test before assuming this trade-off survives a kernel rewrite.
    cur = static_cast<T>(std::lround(((vec[i] - lo) * one_over_width) + 0.5));
    result[i] = cur;
    sum += cur;
  }
  sum_q = sum;
}
}  // namespace alaya::laser::scalar
