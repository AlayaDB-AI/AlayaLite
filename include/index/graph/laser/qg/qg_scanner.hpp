// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file qg_scanner.hpp
 * @brief SIMD-accelerated neighbor scanning using RaBitQ quantization.
 *
 * Computes approximate distances to all neighbors of a node in a single
 * vectorized pass. Uses lookup table accumulation for inner product computation
 * between quantized query and binary-coded neighbor vectors.
 *
 * Distance formula: dist ≈ ||y||² + triple_x + factor_dq * width * result + factor_vq * vl + sqr_qr
 * where result comes from the fast scan accumulation of quantized codes.
 * Note: triple_x already includes ||x_r||^2 (residual dim norm) pre-added during build.
 */

#pragma once

#include <immintrin.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "index/graph/laser/quantization/fastscan_impl.hpp"

namespace alaya::laser {

/**
 * @brief Computes approximate distances using precomputed factors.
 *
 * Combines the fast scan result with correction factors to produce
 * accurate distance estimates for all neighbors simultaneously.
 */
static inline void appro_dist_impl(size_t num_points,
                                   float sqr_y,
                                   float width,
                                   float vl,
                                   float sqr_qr,
                                   const float *__restrict__ result,
                                   const float *__restrict__ triple_x,
                                   const float *__restrict__ fac_dq,
                                   const float *__restrict__ fac_vq,
                                   float *__restrict__ appro_dist) {
#if defined(__AVX512F__)
  const __m512 sqr_y_simd = _mm512_set1_ps(sqr_y);
  const __m512 width_simd = _mm512_set1_ps(width);
  const __m512 vl_simd = _mm512_set1_ps(vl);
  const __m512 sqr_qr_simd = _mm512_set1_ps(sqr_qr);

  __m512 result_simd;
  __m512 triple_x_simd;
  __m512 fac_dq_simd;
  __m512 fac_vq_simd;

  for (size_t i = 0; i < num_points; i += 16) {
    result_simd = _mm512_loadu_ps(&result[i]);
    triple_x_simd = _mm512_loadu_ps(&triple_x[i]);
    fac_dq_simd = _mm512_loadu_ps(&fac_dq[i]);
    fac_vq_simd = _mm512_loadu_ps(&fac_vq[i]);

    triple_x_simd = _mm512_add_ps(triple_x_simd, sqr_y_simd);

    fac_dq_simd = _mm512_mul_ps(fac_dq_simd, width_simd);
    fac_dq_simd = _mm512_mul_ps(fac_dq_simd, result_simd);

    fac_vq_simd = _mm512_fmadd_ps(fac_vq_simd, vl_simd, triple_x_simd);

    triple_x_simd = _mm512_add_ps(fac_dq_simd, fac_vq_simd);
    // Add ||q_r||^2 for improved precision (||x_r||^2 already in triple_x)
    triple_x_simd = _mm512_add_ps(triple_x_simd, sqr_qr_simd);
    _mm512_storeu_ps(&appro_dist[i], triple_x_simd);
  }
#elif defined(__AVX2__)
  const __m256 sqr_y_simd = _mm256_set1_ps(sqr_y);
  const __m256 width_simd = _mm256_set1_ps(width);
  const __m256 vl_simd = _mm256_set1_ps(vl);
  const __m256 sqr_qr_simd = _mm256_set1_ps(sqr_qr);

  __m256 result_simd;
  __m256 triple_x_simd;
  __m256 fac_dq_simd;
  __m256 fac_vq_simd;

  for (size_t i = 0; i < num_points; i += 8) {
    result_simd = _mm256_loadu_ps(&result[i]);
    triple_x_simd = _mm256_loadu_ps(&triple_x[i]);
    fac_dq_simd = _mm256_loadu_ps(&fac_dq[i]);
    fac_vq_simd = _mm256_loadu_ps(&fac_vq[i]);

    triple_x_simd = _mm256_add_ps(triple_x_simd, sqr_y_simd);

    fac_dq_simd = _mm256_mul_ps(fac_dq_simd, width_simd);
    fac_dq_simd = _mm256_mul_ps(fac_dq_simd, result_simd);

    fac_vq_simd = _mm256_mul_ps(fac_vq_simd, vl_simd);

    triple_x_simd = _mm256_add_ps(_mm256_add_ps(triple_x_simd, fac_dq_simd), fac_vq_simd);
    // Add ||q_r||^2 for improved precision (||x_r||^2 already in triple_x)
    triple_x_simd = _mm256_add_ps(triple_x_simd, sqr_qr_simd);
    _mm256_storeu_ps(&appro_dist[i], triple_x_simd);
  }
  return;
#else
  throw std::runtime_error(
      "qg_scanner: SIMD (AVX2 or AVX512) required but not detected at build time");
#endif
}

/**
 * @brief Scanner for computing approximate distances to all neighbors of a node.
 *
 * Processes quantized neighbor codes and computes distance estimates using
 * SIMD-accelerated lookup table accumulation followed by factor correction.
 */
class QGScanner {
 private:
  // func for packing lookup tables
  size_t padded_dim_;
  size_t degree_bound_;

 public:
  QGScanner() = default;

  explicit QGScanner(size_t padded_dim, size_t degree_bound)
      : padded_dim_(padded_dim), degree_bound_(degree_bound) {}

  void pack_lut(const uint8_t *__restrict__ byte_query, uint8_t *__restrict__ LUT) const {
    pack_lut_impl(padded_dim_, byte_query, LUT);
  }

  void scan_neighbors(float *__restrict__ appro_dist,
                      const uint8_t *__restrict__ LUT,
                      float sqr_y,
                      float vl,
                      float width,
                      float sqr_qr,
                      int32_t sumq,
                      const uint8_t *packed_code,
                      const float *factor) const {
    std::vector<uint16_t> result(degree_bound_);

    /* Compute block by block */
    for (size_t i = 0; i < degree_bound_; i += kBatchSize) {
      accumulate_impl(padded_dim_, packed_code, LUT, &result[i]);
      packed_code = &packed_code[padded_dim_ << 2];
    }

    /* Cast to float and multiple by 2 then minus sumq */
    std::vector<float> result_float(degree_bound_);
#if defined(__AVX512F__)
    const __m512i qq = _mm512_set1_epi32(sumq);
    for (size_t i = 0; i < degree_bound_; i += 32) {
      __m256i i16a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&result[i]));
      __m256i i16b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&result[i + 16]));
      __m512i i32a = _mm512_cvtepi16_epi32(i16a);
      __m512i i32b = _mm512_cvtepi16_epi32(i16b);

      i32a = _mm512_sub_epi32(_mm512_slli_epi32(i32a, 1), qq);
      i32b = _mm512_sub_epi32(_mm512_slli_epi32(i32b, 1), qq);
      __m512 f32a = _mm512_cvtepi32_ps(i32a);
      __m512 f32b = _mm512_cvtepi32_ps(i32b);

      _mm512_storeu_ps(&result_float[i], f32a);
      _mm512_storeu_ps(&result_float[i + 16], f32b);
    }
#else
    for (size_t i = 0; i < degree_bound_; ++i) {
      result_float[i] = static_cast<float>((static_cast<int>(result[i]) << 1) - sumq);
    }
#endif
    const float *triple_x = factor;
    const float *fac_dq = &triple_x[degree_bound_];
    const float *fac_vq = &fac_dq[degree_bound_];
    appro_dist_impl(degree_bound_,
                    sqr_y,
                    width,
                    vl,
                    sqr_qr,
                    result_float.data(),
                    triple_x,
                    fac_dq,
                    fac_vq,
                    appro_dist);
  }
};
}  // namespace alaya::laser
