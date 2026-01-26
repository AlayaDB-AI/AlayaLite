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

// Include the new SIMD L2 distance implementation
#include "simd/distance_l2.hpp"

namespace alaya {

// ============================================================================
// L2 Distance Functions (Full Precision)
// ============================================================================

/**
 * @brief Compute L2 squared distance between two float vectors.
 *
 * This function uses SIMD-optimized implementation when available.
 *
 * @tparam DataType Data type (default: float)
 * @tparam DistanceType Distance type (default: float)
 * @param x First vector
 * @param y Second vector
 * @param dim Vector dimension
 * @return L2 squared distance
 */
template <typename DataType = float, typename DistanceType = float>
inline auto l2_sqr(const DataType *x, const DataType *y, size_t dim) -> DistanceType {
  return simd::l2_sqr<DataType, DistanceType>(x, y, dim);
}

/**
 * @brief Alias for l2_sqr for RaBitQ compatibility.
 */
template <typename DataType = float, typename DistanceType = float>
inline auto l2_sqr_rabitq(const DataType *__restrict__ x,
                          const DataType *__restrict__ y,
                          size_t dim) -> DistanceType {
  return l2_sqr<DataType, DistanceType>(x, y, dim);
}

// ============================================================================
// SQ4 L2 Distance Functions (4-bit Scalar Quantization)
// ============================================================================

/**
 * @brief Compute L2 squared distance between two SQ4-encoded vectors.
 *
 * SQ4 stores 2 values per byte (4 bits each):
 *   - Low nibble (bits 0-3) = even index
 *   - High nibble (bits 4-7) = odd index
 *
 * This function uses SIMD-optimized implementation when available.
 *
 * @note Parameter order matches legacy API: (x, y, dim, min, max)
 *       The underlying SIMD function uses: (x, y, min, max, dim)
 *
 * @tparam DataType Data type for min/max (default: float)
 * @tparam DistanceType Distance type (default: float)
 * @param encoded_x First SQ4-encoded vector
 * @param encoded_y Second SQ4-encoded vector
 * @param dim Vector dimension (number of elements, not bytes)
 * @param min Per-dimension minimum values
 * @param max Per-dimension maximum values
 * @return L2 squared distance
 */
template <typename DataType = float, typename DistanceType = float>
inline auto l2_sqr_sq4(const uint8_t *encoded_x,
                       const uint8_t *encoded_y,
                       size_t dim,
                       const DataType *min,
                       const DataType *max) -> DistanceType {
  return simd::l2_sqr_sq4<DataType, DistanceType>(encoded_x, encoded_y, min, max, dim);
}

// ============================================================================
// SQ8 L2 Distance Functions (8-bit Scalar Quantization)
// ============================================================================

/**
 * @brief Compute L2 squared distance between two SQ8-encoded vectors.
 *
 * This function uses SIMD-optimized implementation when available.
 *
 * @note Parameter order matches legacy API: (x, y, dim, min, max)
 *       The underlying SIMD function uses: (x, y, min, max, dim)
 *
 * @tparam DataType Data type for min/max (default: float)
 * @tparam DistanceType Distance type (default: float)
 * @param encoded_x First SQ8-encoded vector
 * @param encoded_y Second SQ8-encoded vector
 * @param dim Vector dimension
 * @param min Per-dimension minimum values
 * @param max Per-dimension maximum values
 * @return L2 squared distance
 */
template <typename DataType = float, typename DistanceType = float>
inline auto l2_sqr_sq8(const uint8_t *encoded_x,
                       const uint8_t *encoded_y,
                       size_t dim,
                       const DataType *min,
                       const DataType *max) -> DistanceType {
  return simd::l2_sqr_sq8<DataType, DistanceType>(encoded_x, encoded_y, min, max, dim);
}

}  // namespace alaya
