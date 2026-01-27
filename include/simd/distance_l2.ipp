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

// This file is included by distance_l2.hpp - do not include directly
// NOLINTBEGIN(portability-simd-intrinsics)
#include <cstddef>
#include <type_traits>
#include "cpu_features.hpp"

namespace alaya::simd {

// Type alias for function pointer (needed for IDE analysis)
using L2SqrFunc = float (*)(const float *__restrict, const float *__restrict, size_t);
using L2SqrSq8Func = float (*)(const uint8_t *__restrict,
                               const uint8_t *__restrict,
                               size_t,
                               const float *,
                               const float *);
using L2SqrSq4Func = float (*)(const uint8_t *__restrict,
                               const uint8_t *__restrict,
                               size_t,
                               const float *,
                               const float *);

// Generic Implementation (ALAYA_TARGET_SSE2 forces baseline ISA for portability)
ALAYA_NOINLINE
ALAYA_TARGET_SSE2
inline auto l2_sqr_generic(const float *__restrict x, const float *__restrict y, size_t dim)
    -> float {
  float sum = 0.0F;
  for (size_t i = 0; i < dim; ++i) {
    float diff = x[i] - y[i];
    sum += diff * diff;
  }
  return sum;
}

#ifdef ALAYA_ARCH_X86

// AVX2 + FMA Implementation (Optimized with 4 accumulators + loop unrolling)
ALAYA_NOINLINE
ALAYA_TARGET_AVX2
inline auto l2_sqr_avx2(const float *__restrict x, const float *__restrict y, size_t dim) -> float {
  // Use 4 accumulators to hide latency and improve ILP
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();
  __m256 sum2 = _mm256_setzero_ps();
  __m256 sum3 = _mm256_setzero_ps();

  size_t i = 0;
  // Process 32 floats per iteration (4 x 8)
  for (; i + 32 <= dim; i += 32) {
    __m256 vx0 = _mm256_loadu_ps(x + i);
    __m256 vy0 = _mm256_loadu_ps(y + i);
    __m256 vx1 = _mm256_loadu_ps(x + i + 8);
    __m256 vy1 = _mm256_loadu_ps(y + i + 8);
    __m256 vx2 = _mm256_loadu_ps(x + i + 16);
    __m256 vy2 = _mm256_loadu_ps(y + i + 16);
    __m256 vx3 = _mm256_loadu_ps(x + i + 24);
    __m256 vy3 = _mm256_loadu_ps(y + i + 24);

    __m256 diff0 = _mm256_sub_ps(vx0, vy0);
    __m256 diff1 = _mm256_sub_ps(vx1, vy1);
    __m256 diff2 = _mm256_sub_ps(vx2, vy2);
    __m256 diff3 = _mm256_sub_ps(vx3, vy3);

    sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
    sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
    sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);
    sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);
  }

  // Process remaining 8-float blocks
  for (; i + 8 <= dim; i += 8) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = _mm256_loadu_ps(y + i);
    __m256 diff = _mm256_sub_ps(vx, vy);
    sum0 = _mm256_fmadd_ps(diff, diff, sum0);
  }

  // Combine accumulators
  sum0 = _mm256_add_ps(sum0, sum1);
  sum2 = _mm256_add_ps(sum2, sum3);
  sum0 = _mm256_add_ps(sum0, sum2);

  // Efficient horizontal sum (no hadd)
  __m128 hi = _mm256_extractf128_ps(sum0, 1);
  __m128 lo = _mm256_castps256_ps128(sum0);
  __m128 sum128 = _mm_add_ps(lo, hi);
  // Shuffle and add
  __m128 shuf = _mm_movehdup_ps(sum128);  // [1,1,3,3]
  sum128 = _mm_add_ps(sum128, shuf);      // [0+1,1+1,2+3,3+3]
  shuf = _mm_movehl_ps(shuf, sum128);     // [2+3,3+3,...]
  sum128 = _mm_add_ss(sum128, shuf);      // [0+1+2+3,...]
  float result = _mm_cvtss_f32(sum128);

  // Tail
  for (; i < dim; ++i) {
    float diff = x[i] - y[i];
    result += diff * diff;
  }
  return result;
}

// AVX-512 Implementation (Optimized with 4 accumulators + loop unrolling)
ALAYA_NOINLINE
ALAYA_TARGET_AVX512
inline auto l2_sqr_avx512(const float *__restrict x, const float *__restrict y, size_t dim)
    -> float {
  // Use 4 accumulators to hide latency and improve ILP
  __m512 sum0 = _mm512_setzero_ps();
  __m512 sum1 = _mm512_setzero_ps();
  __m512 sum2 = _mm512_setzero_ps();
  __m512 sum3 = _mm512_setzero_ps();

  size_t i = 0;
  // Process 64 floats per iteration (4 x 16)
  for (; i + 64 <= dim; i += 64) {
    __m512 vx0 = _mm512_loadu_ps(x + i);
    __m512 vy0 = _mm512_loadu_ps(y + i);
    __m512 vx1 = _mm512_loadu_ps(x + i + 16);
    __m512 vy1 = _mm512_loadu_ps(y + i + 16);
    __m512 vx2 = _mm512_loadu_ps(x + i + 32);
    __m512 vy2 = _mm512_loadu_ps(y + i + 32);
    __m512 vx3 = _mm512_loadu_ps(x + i + 48);
    __m512 vy3 = _mm512_loadu_ps(y + i + 48);

    __m512 diff0 = _mm512_sub_ps(vx0, vy0);
    __m512 diff1 = _mm512_sub_ps(vx1, vy1);
    __m512 diff2 = _mm512_sub_ps(vx2, vy2);
    __m512 diff3 = _mm512_sub_ps(vx3, vy3);

    sum0 = _mm512_fmadd_ps(diff0, diff0, sum0);
    sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);
    sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);
    sum3 = _mm512_fmadd_ps(diff3, diff3, sum3);
  }

  // Process remaining 16-float blocks
  for (; i + 16 <= dim; i += 16) {
    __m512 vx = _mm512_loadu_ps(x + i);
    __m512 vy = _mm512_loadu_ps(y + i);
    __m512 diff = _mm512_sub_ps(vx, vy);
    sum0 = _mm512_fmadd_ps(diff, diff, sum0);
  }

  // Combine accumulators and reduce
  sum0 = _mm512_add_ps(sum0, sum1);
  sum2 = _mm512_add_ps(sum2, sum3);
  sum0 = _mm512_add_ps(sum0, sum2);
  float result = _mm512_reduce_add_ps(sum0);

  // Tail with mask
  size_t remaining = dim - i;
  if (remaining > 0) {
    auto mask = static_cast<__mmask16>((1U << remaining) - 1);
    __m512 vx = _mm512_maskz_loadu_ps(mask, x + i);
    __m512 vy = _mm512_maskz_loadu_ps(mask, y + i);
    __m512 diff = _mm512_sub_ps(vx, vy);
    result += _mm512_mask_reduce_add_ps(mask, _mm512_mul_ps(diff, diff));
  }
  return result;
}

#endif  // ALAYA_ARCH_X86

// SQ8 L2 Distance Implementation

// Generic SQ8 implementation
// Dequantize: value = min + (x / 255.0) * (max - min)
// L2: sum((x_val - y_val)^2) = sum(((x - y) * scale)^2)
ALAYA_NOINLINE
ALAYA_TARGET_SSE2
inline auto l2_sqr_sq8_generic(const uint8_t *__restrict x,
                               const uint8_t *__restrict y,
                               size_t dim,
                               const float *min,
                               const float *max) -> float {
  constexpr float kInv255 = 1.0F / 255.0F;
  float sum = 0.0F;
  for (size_t i = 0; i < dim; ++i) {
    float scale = (max[i] - min[i]) * kInv255;
    float diff = static_cast<float>(x[i]) - static_cast<float>(y[i]);
    float scaled_diff = diff * scale;
    sum += scaled_diff * scaled_diff;
  }
  return sum;
}

// SQ4 L2 Distance Implementation
// SQ4 stores 2 values per byte (4 bits each): low nibble = even index, high nibble = odd index
// Dequantize: value = min + (q / 15.0) * (max - min)
// L2: sum((x_val - y_val)^2) = sum(((x - y) * scale)^2)
ALAYA_NOINLINE
ALAYA_TARGET_SSE2
inline auto l2_sqr_sq4_generic(const uint8_t *__restrict x,
                               const uint8_t *__restrict y,
                               size_t dim,
                               const float *min,
                               const float *max) -> float {
  constexpr float kInv15 = 1.0F / 15.0F;
  float sum = 0.0F;
  size_t byte_idx = 0;
  for (size_t i = 0; i < dim; i += 2, ++byte_idx) {
    // Extract low nibble (even index)
    uint8_t x_lo = x[byte_idx] & 0x0F;
    uint8_t y_lo = y[byte_idx] & 0x0F;
    float scale_lo = (max[i] - min[i]) * kInv15;
    float diff_lo = static_cast<float>(x_lo) - static_cast<float>(y_lo);
    float scaled_diff_lo = diff_lo * scale_lo;
    sum += scaled_diff_lo * scaled_diff_lo;

    // Extract high nibble (odd index) - only if within dim
    if (i + 1 < dim) {
      uint8_t x_hi = (x[byte_idx] >> 4) & 0x0F;
      uint8_t y_hi = (y[byte_idx] >> 4) & 0x0F;
      float scale_hi = (max[i + 1] - min[i + 1]) * kInv15;
      float diff_hi = static_cast<float>(x_hi) - static_cast<float>(y_hi);
      float scaled_diff_hi = diff_hi * scale_hi;
      sum += scaled_diff_hi * scaled_diff_hi;
    }
  }
  return sum;
}

#ifdef ALAYA_ARCH_X86

// AVX2 SQ8 implementation
ALAYA_NOINLINE
ALAYA_TARGET_AVX2
inline auto l2_sqr_sq8_avx2(const uint8_t *__restrict x,
                            const uint8_t *__restrict y,
                            size_t dim,
                            const float *min,
                            const float *max) -> float {
  const __m256 kInv255 = _mm256_set1_ps(1.0F / 255.0F);
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();

  size_t i = 0;
  // Process 16 elements per iteration (2 x 8)
  for (; i + 16 <= dim; i += 16) {
    // Load 16 uint8_t values and convert to float
    __m128i x_u8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + i));
    __m128i y_u8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y + i));

    // Convert first 8 uint8 to int32, then to float
    __m256i x_i32_0 = _mm256_cvtepu8_epi32(x_u8);
    __m256i y_i32_0 = _mm256_cvtepu8_epi32(y_u8);
    __m256 x_f0 = _mm256_cvtepi32_ps(x_i32_0);
    __m256 y_f0 = _mm256_cvtepi32_ps(y_i32_0);

    // Convert second 8 uint8 to int32, then to float
    __m128i x_u8_hi = _mm_srli_si128(x_u8, 8);
    __m128i y_u8_hi = _mm_srli_si128(y_u8, 8);
    __m256i x_i32_1 = _mm256_cvtepu8_epi32(x_u8_hi);
    __m256i y_i32_1 = _mm256_cvtepu8_epi32(y_u8_hi);
    __m256 x_f1 = _mm256_cvtepi32_ps(x_i32_1);
    __m256 y_f1 = _mm256_cvtepi32_ps(y_i32_1);

    // Load min/max and compute scale = (max - min) / 255
    __m256 min0 = _mm256_loadu_ps(min + i);
    __m256 max0 = _mm256_loadu_ps(max + i);
    __m256 min1 = _mm256_loadu_ps(min + i + 8);
    __m256 max1 = _mm256_loadu_ps(max + i + 8);
    __m256 scale0 = _mm256_mul_ps(_mm256_sub_ps(max0, min0), kInv255);
    __m256 scale1 = _mm256_mul_ps(_mm256_sub_ps(max1, min1), kInv255);

    // Compute diff = (x - y) * scale
    __m256 diff0 = _mm256_mul_ps(_mm256_sub_ps(x_f0, y_f0), scale0);
    __m256 diff1 = _mm256_mul_ps(_mm256_sub_ps(x_f1, y_f1), scale1);

    // Accumulate diff^2
    sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
    sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);
  }

  // Process remaining 8-element block
  for (; i + 8 <= dim; i += 8) {
    __m128i x_u8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(x + i));
    __m128i y_u8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(y + i));
    __m256i x_i32 = _mm256_cvtepu8_epi32(x_u8);
    __m256i y_i32 = _mm256_cvtepu8_epi32(y_u8);
    __m256 x_f = _mm256_cvtepi32_ps(x_i32);
    __m256 y_f = _mm256_cvtepi32_ps(y_i32);

    __m256 vmin = _mm256_loadu_ps(min + i);
    __m256 vmax = _mm256_loadu_ps(max + i);
    __m256 scale = _mm256_mul_ps(_mm256_sub_ps(vmax, vmin), kInv255);
    __m256 diff = _mm256_mul_ps(_mm256_sub_ps(x_f, y_f), scale);
    sum0 = _mm256_fmadd_ps(diff, diff, sum0);
  }

  // Combine accumulators
  sum0 = _mm256_add_ps(sum0, sum1);

  // Horizontal sum
  __m128 hi = _mm256_extractf128_ps(sum0, 1);
  __m128 lo = _mm256_castps256_ps128(sum0);
  __m128 sum128 = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(sum128);
  sum128 = _mm_add_ps(sum128, shuf);
  shuf = _mm_movehl_ps(shuf, sum128);
  sum128 = _mm_add_ss(sum128, shuf);
  float result = _mm_cvtss_f32(sum128);

  // Tail
  constexpr float kInv255Scalar = 1.0F / 255.0F;
  for (; i < dim; ++i) {
    float scale = (max[i] - min[i]) * kInv255Scalar;
    float diff = static_cast<float>(x[i]) - static_cast<float>(y[i]);
    float scaled_diff = diff * scale;
    result += scaled_diff * scaled_diff;
  }
  return result;
}

// AVX-512 SQ8 implementation
ALAYA_NOINLINE
ALAYA_TARGET_AVX512
inline auto l2_sqr_sq8_avx512(const uint8_t *__restrict x,
                              const uint8_t *__restrict y,
                              size_t dim,
                              const float *min,
                              const float *max) -> float {
  const __m512 kInv255 = _mm512_set1_ps(1.0F / 255.0F);
  __m512 sum0 = _mm512_setzero_ps();
  __m512 sum1 = _mm512_setzero_ps();

  size_t i = 0;
  // Process 32 elements per iteration (2 x 16)
  for (; i + 32 <= dim; i += 32) {
    // Load 16 uint8_t and convert to float
    __m128i x_u8_0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + i));
    __m128i y_u8_0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y + i));
    __m128i x_u8_1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + i + 16));
    __m128i y_u8_1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y + i + 16));

    __m512i x_i32_0 = _mm512_cvtepu8_epi32(x_u8_0);
    __m512i y_i32_0 = _mm512_cvtepu8_epi32(y_u8_0);
    __m512i x_i32_1 = _mm512_cvtepu8_epi32(x_u8_1);
    __m512i y_i32_1 = _mm512_cvtepu8_epi32(y_u8_1);

    __m512 x_f0 = _mm512_cvtepi32_ps(x_i32_0);
    __m512 y_f0 = _mm512_cvtepi32_ps(y_i32_0);
    __m512 x_f1 = _mm512_cvtepi32_ps(x_i32_1);
    __m512 y_f1 = _mm512_cvtepi32_ps(y_i32_1);

    // Load min/max and compute scale
    __m512 min0 = _mm512_loadu_ps(min + i);
    __m512 max0 = _mm512_loadu_ps(max + i);
    __m512 min1 = _mm512_loadu_ps(min + i + 16);
    __m512 max1 = _mm512_loadu_ps(max + i + 16);
    __m512 scale0 = _mm512_mul_ps(_mm512_sub_ps(max0, min0), kInv255);
    __m512 scale1 = _mm512_mul_ps(_mm512_sub_ps(max1, min1), kInv255);

    // Compute diff = (x - y) * scale
    __m512 diff0 = _mm512_mul_ps(_mm512_sub_ps(x_f0, y_f0), scale0);
    __m512 diff1 = _mm512_mul_ps(_mm512_sub_ps(x_f1, y_f1), scale1);

    // Accumulate
    sum0 = _mm512_fmadd_ps(diff0, diff0, sum0);
    sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);
  }

  // Process remaining 16-element block
  for (; i + 16 <= dim; i += 16) {
    __m128i x_u8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + i));
    __m128i y_u8 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y + i));
    __m512i x_i32 = _mm512_cvtepu8_epi32(x_u8);
    __m512i y_i32 = _mm512_cvtepu8_epi32(y_u8);
    __m512 x_f = _mm512_cvtepi32_ps(x_i32);
    __m512 y_f = _mm512_cvtepi32_ps(y_i32);

    __m512 vmin = _mm512_loadu_ps(min + i);
    __m512 vmax = _mm512_loadu_ps(max + i);
    __m512 scale = _mm512_mul_ps(_mm512_sub_ps(vmax, vmin), kInv255);
    __m512 diff = _mm512_mul_ps(_mm512_sub_ps(x_f, y_f), scale);
    sum0 = _mm512_fmadd_ps(diff, diff, sum0);
  }

  // Combine and reduce
  sum0 = _mm512_add_ps(sum0, sum1);
  float result = _mm512_reduce_add_ps(sum0);

  // Tail
  constexpr float kInv255Scalar = 1.0F / 255.0F;
  for (; i < dim; ++i) {
    float scale = (max[i] - min[i]) * kInv255Scalar;
    float diff = static_cast<float>(x[i]) - static_cast<float>(y[i]);
    float scaled_diff = diff * scale;
    result += scaled_diff * scaled_diff;
  }
  return result;
}

// AVX2 SQ4 implementation
ALAYA_NOINLINE
ALAYA_TARGET_AVX2
inline auto l2_sqr_sq4_avx2(const uint8_t *__restrict x,
                            const uint8_t *__restrict y,
                            size_t dim,
                            const float *min,
                            const float *max) -> float {
  const __m256 kInv15 = _mm256_set1_ps(1.0F / 15.0F);
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();

  size_t i = 0;
  // Process 32 elements per iteration (16 bytes -> 32 4-bit values)
  for (; i + 32 <= dim; i += 32) {
    size_t byte_idx = i / 2;
    // Load 16 bytes containing 32 4-bit values
    __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + byte_idx));
    __m128i packed_y = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y + byte_idx));

    // Extract low nibbles (even indices: 0, 2, 4, ...)
    __m128i x_lo_8 = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
    __m128i y_lo_8 = _mm_and_si128(packed_y, _mm_set1_epi8(0x0F));

    // Extract high nibbles (odd indices: 1, 3, 5, ...)
    __m128i x_hi_8 = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));
    __m128i y_hi_8 = _mm_and_si128(_mm_srli_epi16(packed_y, 4), _mm_set1_epi8(0x0F));

    // Process first 8 pairs (16 values: indices 0-15)
    // Convert first 8 low nibbles to float
    __m256i x_lo_i32_0 = _mm256_cvtepu8_epi32(x_lo_8);
    __m256i y_lo_i32_0 = _mm256_cvtepu8_epi32(y_lo_8);
    __m256 x_lo_f0 = _mm256_cvtepi32_ps(x_lo_i32_0);
    __m256 y_lo_f0 = _mm256_cvtepi32_ps(y_lo_i32_0);

    // Convert first 8 high nibbles to float
    __m256i x_hi_i32_0 = _mm256_cvtepu8_epi32(x_hi_8);
    __m256i y_hi_i32_0 = _mm256_cvtepu8_epi32(y_hi_8);
    __m256 x_hi_f0 = _mm256_cvtepi32_ps(x_hi_i32_0);
    __m256 y_hi_f0 = _mm256_cvtepi32_ps(y_hi_i32_0);

    // Load min/max for even indices (0, 2, 4, 6, 8, 10, 12, 14)
    // We need to gather because min/max are not interleaved
    __m256 min_lo_0 = _mm256_set_ps(min[i + 14],
                                    min[i + 12],
                                    min[i + 10],
                                    min[i + 8],
                                    min[i + 6],
                                    min[i + 4],
                                    min[i + 2],
                                    min[i + 0]);
    __m256 max_lo_0 = _mm256_set_ps(max[i + 14],
                                    max[i + 12],
                                    max[i + 10],
                                    max[i + 8],
                                    max[i + 6],
                                    max[i + 4],
                                    max[i + 2],
                                    max[i + 0]);
    __m256 min_hi_0 = _mm256_set_ps(min[i + 15],
                                    min[i + 13],
                                    min[i + 11],
                                    min[i + 9],
                                    min[i + 7],
                                    min[i + 5],
                                    min[i + 3],
                                    min[i + 1]);
    __m256 max_hi_0 = _mm256_set_ps(max[i + 15],
                                    max[i + 13],
                                    max[i + 11],
                                    max[i + 9],
                                    max[i + 7],
                                    max[i + 5],
                                    max[i + 3],
                                    max[i + 1]);

    __m256 scale_lo_0 = _mm256_mul_ps(_mm256_sub_ps(max_lo_0, min_lo_0), kInv15);
    __m256 scale_hi_0 = _mm256_mul_ps(_mm256_sub_ps(max_hi_0, min_hi_0), kInv15);

    __m256 diff_lo_0 = _mm256_mul_ps(_mm256_sub_ps(x_lo_f0, y_lo_f0), scale_lo_0);
    __m256 diff_hi_0 = _mm256_mul_ps(_mm256_sub_ps(x_hi_f0, y_hi_f0), scale_hi_0);

    sum0 = _mm256_fmadd_ps(diff_lo_0, diff_lo_0, sum0);
    sum1 = _mm256_fmadd_ps(diff_hi_0, diff_hi_0, sum1);

    // Process second 8 pairs (16 values: indices 16-31)
    __m128i x_lo_8_hi = _mm_srli_si128(x_lo_8, 8);
    __m128i y_lo_8_hi = _mm_srli_si128(y_lo_8, 8);
    __m128i x_hi_8_hi = _mm_srli_si128(x_hi_8, 8);
    __m128i y_hi_8_hi = _mm_srli_si128(y_hi_8, 8);

    __m256i x_lo_i32_1 = _mm256_cvtepu8_epi32(x_lo_8_hi);
    __m256i y_lo_i32_1 = _mm256_cvtepu8_epi32(y_lo_8_hi);
    __m256 x_lo_f1 = _mm256_cvtepi32_ps(x_lo_i32_1);
    __m256 y_lo_f1 = _mm256_cvtepi32_ps(y_lo_i32_1);

    __m256i x_hi_i32_1 = _mm256_cvtepu8_epi32(x_hi_8_hi);
    __m256i y_hi_i32_1 = _mm256_cvtepu8_epi32(y_hi_8_hi);
    __m256 x_hi_f1 = _mm256_cvtepi32_ps(x_hi_i32_1);
    __m256 y_hi_f1 = _mm256_cvtepi32_ps(y_hi_i32_1);

    __m256 min_lo_1 = _mm256_set_ps(min[i + 30],
                                    min[i + 28],
                                    min[i + 26],
                                    min[i + 24],
                                    min[i + 22],
                                    min[i + 20],
                                    min[i + 18],
                                    min[i + 16]);
    __m256 max_lo_1 = _mm256_set_ps(max[i + 30],
                                    max[i + 28],
                                    max[i + 26],
                                    max[i + 24],
                                    max[i + 22],
                                    max[i + 20],
                                    max[i + 18],
                                    max[i + 16]);
    __m256 min_hi_1 = _mm256_set_ps(min[i + 31],
                                    min[i + 29],
                                    min[i + 27],
                                    min[i + 25],
                                    min[i + 23],
                                    min[i + 21],
                                    min[i + 19],
                                    min[i + 17]);
    __m256 max_hi_1 = _mm256_set_ps(max[i + 31],
                                    max[i + 29],
                                    max[i + 27],
                                    max[i + 25],
                                    max[i + 23],
                                    max[i + 21],
                                    max[i + 19],
                                    max[i + 17]);

    __m256 scale_lo_1 = _mm256_mul_ps(_mm256_sub_ps(max_lo_1, min_lo_1), kInv15);
    __m256 scale_hi_1 = _mm256_mul_ps(_mm256_sub_ps(max_hi_1, min_hi_1), kInv15);

    __m256 diff_lo_1 = _mm256_mul_ps(_mm256_sub_ps(x_lo_f1, y_lo_f1), scale_lo_1);
    __m256 diff_hi_1 = _mm256_mul_ps(_mm256_sub_ps(x_hi_f1, y_hi_f1), scale_hi_1);

    sum0 = _mm256_fmadd_ps(diff_lo_1, diff_lo_1, sum0);
    sum1 = _mm256_fmadd_ps(diff_hi_1, diff_hi_1, sum1);
  }

  // Combine accumulators
  sum0 = _mm256_add_ps(sum0, sum1);

  // Horizontal sum
  __m128 hi = _mm256_extractf128_ps(sum0, 1);
  __m128 lo = _mm256_castps256_ps128(sum0);
  __m128 sum128 = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(sum128);
  sum128 = _mm_add_ps(sum128, shuf);
  shuf = _mm_movehl_ps(shuf, sum128);
  sum128 = _mm_add_ss(sum128, shuf);
  float result = _mm_cvtss_f32(sum128);

  // Tail (scalar fallback for remaining elements)
  constexpr float kInv15Scalar = 1.0F / 15.0F;
  for (; i < dim; i += 2) {
    size_t byte_idx = i / 2;
    uint8_t x_lo = x[byte_idx] & 0x0F;
    uint8_t y_lo = y[byte_idx] & 0x0F;
    float scale_lo = (max[i] - min[i]) * kInv15Scalar;
    float diff_lo = static_cast<float>(x_lo) - static_cast<float>(y_lo);
    result += diff_lo * scale_lo * diff_lo * scale_lo;

    if (i + 1 < dim) {
      uint8_t x_hi = (x[byte_idx] >> 4) & 0x0F;
      uint8_t y_hi = (y[byte_idx] >> 4) & 0x0F;
      float scale_hi = (max[i + 1] - min[i + 1]) * kInv15Scalar;
      float diff_hi = static_cast<float>(x_hi) - static_cast<float>(y_hi);
      result += diff_hi * scale_hi * diff_hi * scale_hi;
    }
  }
  return result;
}

// AVX-512 SQ4 implementation
ALAYA_NOINLINE
ALAYA_TARGET_AVX512
inline auto l2_sqr_sq4_avx512(const uint8_t *__restrict x,
                              const uint8_t *__restrict y,
                              size_t dim,
                              const float *min,
                              const float *max) -> float {
  const __m512 kInv15 = _mm512_set1_ps(1.0F / 15.0F);
  __m512 sum0 = _mm512_setzero_ps();
  __m512 sum1 = _mm512_setzero_ps();

  size_t i = 0;
  // Process 32 elements per iteration (16 bytes -> 32 4-bit values)
  for (; i + 32 <= dim; i += 32) {
    size_t byte_idx = i / 2;
    // Load 16 bytes containing 32 4-bit values
    __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + byte_idx));
    __m128i packed_y = _mm_loadu_si128(reinterpret_cast<const __m128i *>(y + byte_idx));

    // Extract low nibbles (even indices)
    __m128i x_lo_8 = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
    __m128i y_lo_8 = _mm_and_si128(packed_y, _mm_set1_epi8(0x0F));

    // Extract high nibbles (odd indices)
    __m128i x_hi_8 = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));
    __m128i y_hi_8 = _mm_and_si128(_mm_srli_epi16(packed_y, 4), _mm_set1_epi8(0x0F));

    // Convert all 16 low nibbles to float using AVX-512
    __m512i x_lo_i32 = _mm512_cvtepu8_epi32(x_lo_8);
    __m512i y_lo_i32 = _mm512_cvtepu8_epi32(y_lo_8);
    __m512 x_lo_f = _mm512_cvtepi32_ps(x_lo_i32);
    __m512 y_lo_f = _mm512_cvtepi32_ps(y_lo_i32);

    // Convert all 16 high nibbles to float
    __m512i x_hi_i32 = _mm512_cvtepu8_epi32(x_hi_8);
    __m512i y_hi_i32 = _mm512_cvtepu8_epi32(y_hi_8);
    __m512 x_hi_f = _mm512_cvtepi32_ps(x_hi_i32);
    __m512 y_hi_f = _mm512_cvtepi32_ps(y_hi_i32);

    // Load min/max for even indices using gather (indices 0, 2, 4, ..., 30)
    __m512i even_idx = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m512i odd_idx = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);

    __m512 min_lo = _mm512_i32gather_ps(even_idx, min + i, 4);
    __m512 max_lo = _mm512_i32gather_ps(even_idx, max + i, 4);
    __m512 min_hi = _mm512_i32gather_ps(odd_idx, min + i, 4);
    __m512 max_hi = _mm512_i32gather_ps(odd_idx, max + i, 4);

    __m512 scale_lo = _mm512_mul_ps(_mm512_sub_ps(max_lo, min_lo), kInv15);
    __m512 scale_hi = _mm512_mul_ps(_mm512_sub_ps(max_hi, min_hi), kInv15);

    __m512 diff_lo = _mm512_mul_ps(_mm512_sub_ps(x_lo_f, y_lo_f), scale_lo);
    __m512 diff_hi = _mm512_mul_ps(_mm512_sub_ps(x_hi_f, y_hi_f), scale_hi);

    sum0 = _mm512_fmadd_ps(diff_lo, diff_lo, sum0);
    sum1 = _mm512_fmadd_ps(diff_hi, diff_hi, sum1);
  }

  // Combine and reduce
  sum0 = _mm512_add_ps(sum0, sum1);
  float result = _mm512_reduce_add_ps(sum0);

  // Tail (scalar fallback)
  constexpr float kInv15Scalar = 1.0F / 15.0F;
  for (; i < dim; i += 2) {
    size_t byte_idx = i / 2;
    uint8_t x_lo = x[byte_idx] & 0x0F;
    uint8_t y_lo = y[byte_idx] & 0x0F;
    float scale_lo = (max[i] - min[i]) * kInv15Scalar;
    float diff_lo = static_cast<float>(x_lo) - static_cast<float>(y_lo);
    result += diff_lo * scale_lo * diff_lo * scale_lo;

    if (i + 1 < dim) {
      uint8_t x_hi = (x[byte_idx] >> 4) & 0x0F;
      uint8_t y_hi = (y[byte_idx] >> 4) & 0x0F;
      float scale_hi = (max[i + 1] - min[i + 1]) * kInv15Scalar;
      float diff_hi = static_cast<float>(x_hi) - static_cast<float>(y_hi);
      result += diff_hi * scale_hi * diff_hi * scale_hi;
    }
  }
  return result;
}

#endif  // ALAYA_ARCH_X86

// ============================================================================
// Runtime Dispatch
// ============================================================================

inline auto get_l2_sqr_func() -> L2SqrFunc {
  static const L2SqrFunc kFunc = []() -> L2SqrFunc {
#ifdef ALAYA_ARCH_X86
    const auto &f = get_cpu_features();
    if (f.avx512f_) {
      return l2_sqr_avx2;  // because avx2 performs better than avx-512 in most cases
    }
    if (f.avx2_ && f.fma_) {
      return l2_sqr_avx2;
    }
#endif
    return l2_sqr_generic;
  }();
  return kFunc;
}

inline auto get_l2_sqr_sq8_func() -> L2SqrSq8Func {
  static const L2SqrSq8Func kFunc = []() -> L2SqrSq8Func {
#ifdef ALAYA_ARCH_X86
    const auto &f = get_cpu_features();
    if (f.avx512f_) {
      return l2_sqr_sq8_avx512;
    }
    if (f.avx2_ && f.fma_) {
      return l2_sqr_sq8_avx2;
    }
#endif
    return l2_sqr_sq8_generic;
  }();
  return kFunc;
}

inline auto get_l2_sqr_sq4_func() -> L2SqrSq4Func {
  static const L2SqrSq4Func kFunc = []() -> L2SqrSq4Func {
#ifdef ALAYA_ARCH_X86
    const auto &f = get_cpu_features();
    if (f.avx512f_) {
      return l2_sqr_sq4_avx2;
    }
    if (f.avx2_ && f.fma_) {
      return l2_sqr_sq4_avx2;
    }
#endif
    return l2_sqr_sq4_generic;
  }();
  return kFunc;
}

// ============================================================================
// Public API
// ============================================================================
template <typename DataType, typename DistanceType>
inline auto l2_sqr(const DataType *__restrict x, const DataType *__restrict y, size_t dim)
    -> DistanceType {
  if constexpr (std::is_same_v<DataType, float>) {
    return static_cast<DistanceType>(get_l2_sqr_func()(x, y, dim));
  } else {
    DistanceType sum = 0;
    for (size_t i = 0; i < dim; ++i) {
      auto diff = static_cast<DistanceType>(x[i]) - static_cast<DistanceType>(y[i]);
      sum += diff * diff;
    }
    return sum;
  }
}

template <typename DataType, typename DistanceType>
inline auto l2_sqr_sq8(const uint8_t *__restrict x,
                       const uint8_t *__restrict y,
                       size_t dim,
                       const DataType *min,
                       const DataType *max) -> DistanceType {
  if constexpr (std::is_same_v<DataType, float>) {
    return static_cast<DistanceType>(get_l2_sqr_sq8_func()(x, y, dim, min, max));
  } else {
    // TODO(?): 1. min, max should always be float; 2. we can pre calculate scales in fit function
    // Generic fallback for non-float types
    constexpr float kInv255 = 1.0F / 255.0F;
    DistanceType sum = 0;
    for (size_t i = 0; i < dim; ++i) {
      auto scale =
          (static_cast<DistanceType>(max[i]) - static_cast<DistanceType>(min[i])) * kInv255;
      auto diff = (static_cast<DistanceType>(x[i]) - static_cast<DistanceType>(y[i])) * scale;
      sum += diff * diff;
    }
    return sum;
  }
}

template <typename DataType, typename DistanceType>
inline auto l2_sqr_sq4(const uint8_t *__restrict x,
                       const uint8_t *__restrict y,
                       size_t dim,
                       const DataType *min,
                       const DataType *max) -> DistanceType {
  if constexpr (std::is_same_v<DataType, float>) {
    return static_cast<DistanceType>(get_l2_sqr_sq4_func()(x, y, dim, min, max));
  } else {
    // TODO(?): 1. min, max should always be float; 2. we can pre calculate scales in fit function
    // Generic fallback for non-float types
    constexpr float kInv15 = 1.0F / 15.0F;
    DistanceType sum = 0;
    size_t byte_idx = 0;
    for (size_t i = 0; i < dim; i += 2, ++byte_idx) {
      // Low nibble (even index)
      auto x_lo = static_cast<DistanceType>(x[byte_idx] & 0x0F);
      auto y_lo = static_cast<DistanceType>(y[byte_idx] & 0x0F);
      auto scale_lo =
          (static_cast<DistanceType>(max[i]) - static_cast<DistanceType>(min[i])) * kInv15;
      auto diff_lo = (x_lo - y_lo) * scale_lo;
      sum += diff_lo * diff_lo;

      // High nibble (odd index)
      if (i + 1 < dim) {
        auto x_hi = static_cast<DistanceType>((x[byte_idx] >> 4) & 0x0F);
        auto y_hi = static_cast<DistanceType>((y[byte_idx] >> 4) & 0x0F);
        auto scale_hi =
            (static_cast<DistanceType>(max[i + 1]) - static_cast<DistanceType>(min[i + 1])) *
            kInv15;
        auto diff_hi = (x_hi - y_hi) * scale_hi;
        sum += diff_hi * diff_hi;
      }
    }
    return sum;
  }
}

}  // namespace alaya::simd
// NOLINTEND(portability-simd-intrinsics)
