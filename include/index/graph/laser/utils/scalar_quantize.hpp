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
    cur = static_cast<T>(std::lround(((vec[i] - lo) * one_over_width) + 0.5));
    result[i] = cur;
    sum += cur;
  }
  sum_q = sum;
}
}  // namespace alaya::laser::scalar
