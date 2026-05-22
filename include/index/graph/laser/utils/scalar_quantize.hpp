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

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "simd/laser_dispatch.hpp"
#include "utils/platform.hpp"

namespace alaya::laser::scalar {

/** @brief Computes min/max values of a vector for quantization range. */
inline void data_range(const float *ALAYA_RESTRICT vec, size_t dim, float &lo, float &hi) {
  simd::get_data_range_func()(vec, dim, lo, hi);
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
void quantize(T *ALAYA_RESTRICT result,
              const float *ALAYA_RESTRICT vec,
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
