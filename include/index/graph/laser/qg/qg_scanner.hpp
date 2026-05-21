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

#include <cstdint>
#include <vector>

#include "index/graph/laser/quantization/fastscan_impl.hpp"
#include "simd/laser_dispatch.hpp"

namespace alaya::laser {

/**
 * @brief Scanner for computing approximate distances to all neighbors of a node.
 *
 * Processes quantized neighbor codes and computes distance estimates using
 * SIMD-accelerated lookup table accumulation followed by factor correction.
 */
class QGScanner {
 private:
  size_t padded_dim_ = 0;
  size_t degree_bound_ = 0;
  simd::AccumulateFn accumulate_ = nullptr;
  simd::ConvertAccumFn convert_ = nullptr;
  simd::AppDistFn compute_appro_dist_ = nullptr;

 public:
  QGScanner() = default;

  explicit QGScanner(size_t padded_dim, size_t degree_bound)
      : padded_dim_(padded_dim),
        degree_bound_(degree_bound),
        accumulate_(simd::get_accumulate_func()),
        convert_(simd::get_convert_func()),
        compute_appro_dist_(simd::get_appro_dist_func()) {}

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
      accumulate_(padded_dim_, packed_code, LUT, &result[i]);
      packed_code = &packed_code[padded_dim_ << 2];
    }

    /* Cast to float, multiply by 2, subtract sumq. */
    std::vector<float> result_float(degree_bound_);
    convert_(degree_bound_, result.data(), sumq, result_float.data());
    const float *triple_x = factor;
    const float *fac_dq = &triple_x[degree_bound_];
    const float *fac_vq = &fac_dq[degree_bound_];
    compute_appro_dist_(degree_bound_,
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
