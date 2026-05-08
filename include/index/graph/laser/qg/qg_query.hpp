/*
 * Copyright 2025 AlayaDB.AI
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

/**
 * @file qg_query.hpp
 * @brief Query preprocessing for RaBitQ-based approximate nearest neighbor search.
 *
 * Handles query vector preparation including:
 * 1. Fast Hadamard Transform (FHT) rotation for distribution normalization
 * 2. Scalar quantization to 6-bit representation
 * 3. Lookup table (LUT) packing for SIMD-accelerated distance computation
 */

#pragma once

#include <cstdint>

#include "index/graph/laser/common.hpp"
#include "index/graph/laser/qg/qg_scanner.hpp"
#include "index/graph/laser/utils/memory.hpp"
#include "index/graph/laser/utils/rotator.hpp"
#include "index/graph/laser/utils/scalar_quantize.hpp"

namespace alaya::laser {

/**
 * @brief Prepares query vectors for efficient RaBitQ distance computation.
 *
 * The query preparation pipeline:
 * 1. Apply FHT rotation (same as used during indexing)
 * 2. Compute value range and quantization width
 * 3. Quantize to 6-bit integers
 * 4. Pack into lookup table format for SIMD scanning
 */
class QGQuery {
 private:
  const float *query_data_ = nullptr;
  std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> lut_;
  size_t padded_dim_ = 0;
  float width_ = 0;
  float lower_val_ = 0;
  float upper_val_ = 0;
  int32_t sumq_ = 0;
  float sqr_qr_ = 0;  // Query residual norm squared ||q_r||^2

 public:
  explicit QGQuery(const float *q, size_t padded_dim)
      : query_data_(q),
        lut_(padded_dim << 2)  // padded_dim / 4 * 16
        ,
        padded_dim_(padded_dim) {}

  /**
   * @brief Prepares query for RaBitQ distance computation.
   *
   * Steps: rotate -> quantize -> pack lookup table
   */
  void query_prepare(const FHTRotator &rotator, const QGScanner &scanner) {
    // Rotate query using Fast Hadamard Transform
    std::vector<float, memory::AlignedAllocator<float>> rd_query(padded_dim_);
    rotator.rotate(query_data_, rd_query.data());

    // quantize query
    std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> byte_query(padded_dim_);
    scalar::data_range(rd_query.data(), padded_dim_, lower_val_, upper_val_);
    width_ = (upper_val_ - lower_val_) / ((1 << QG_BQUERY) - 1);
    scalar::quantize(byte_query.data(), rd_query.data(), padded_dim_, lower_val_, width_, sumq_);

    // pack lut
    scanner.pack_lut(byte_query.data(), lut_.data());
  }

  [[nodiscard]] const float &width() const { return width_; }

  [[nodiscard]] const float &lower_val() const { return lower_val_; }

  [[nodiscard]] const int32_t &sumq() const { return sumq_; }

  [[nodiscard]] const std::vector<uint8_t, memory::AlignedAllocator<uint8_t, 64>> &lut() const {
    return lut_;
  }

  [[nodiscard]] const float *query_data() const { return query_data_; }

  void set_sqr_qr(float sqr_qr) { sqr_qr_ = sqr_qr; }

  [[nodiscard]] float sqr_qr() const { return sqr_qr_; }
};
}  // namespace alaya::laser
