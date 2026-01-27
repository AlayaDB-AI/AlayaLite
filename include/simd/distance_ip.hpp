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
#include <type_traits>
#include "cpu_features.hpp"

namespace alaya::simd {

// ============================================================================
// Type Definitions
// ============================================================================

/// Function pointer type for full-precision IP distance
using IpSqrFunc = float (*)(const float *__restrict, const float *__restrict, size_t);

/// Function pointer type for SQ8-encoded IP distance
using IpSqrSq8Func = float (*)(const uint8_t *__restrict,
                               const uint8_t *__restrict,
                               size_t,
                               const float *,
                               const float *);

/// Function pointer type for SQ4-encoded IP distance
using IpSqrSq4Func = float (*)(const uint8_t *__restrict,
                               const uint8_t *__restrict,
                               size_t,
                               const float *,
                               const float *);

// ============================================================================
// Full Precision IP Distance Declarations
// ============================================================================

auto ip_sqr_generic(const float *__restrict x, const float *__restrict y, size_t dim) -> float;

#ifdef ALAYA_ARCH_X86
auto ip_sqr_avx2(const float *__restrict x, const float *__restrict y, size_t dim) -> float;
auto ip_sqr_avx512(const float *__restrict x, const float *__restrict y, size_t dim) -> float;
#endif

// ============================================================================
// SQ8 IP Distance Declarations
// ============================================================================

auto ip_sqr_sq8_generic(const uint8_t *__restrict x,
                        const uint8_t *__restrict y,
                        size_t dim,
                        const float *min,
                        const float *max) -> float;

#ifdef ALAYA_ARCH_X86
auto ip_sqr_sq8_avx2(const uint8_t *__restrict x,
                     const uint8_t *__restrict y,
                     size_t dim,
                     const float *min,
                     const float *max) -> float;
auto ip_sqr_sq8_avx512(const uint8_t *__restrict x,
                       const uint8_t *__restrict y,
                       size_t dim,
                       const float *min,
                       const float *max) -> float;
#endif

// ============================================================================
// SQ4 IP Distance Declarations
// ============================================================================

auto ip_sqr_sq4_generic(const uint8_t *__restrict x,
                        const uint8_t *__restrict y,
                        size_t dim,
                        const float *min,
                        const float *max) -> float;

#ifdef ALAYA_ARCH_X86
auto ip_sqr_sq4_avx2(const uint8_t *__restrict x,
                     const uint8_t *__restrict y,
                     size_t dim,
                     const float *min,
                     const float *max) -> float;
auto ip_sqr_sq4_avx512(const uint8_t *__restrict x,
                       const uint8_t *__restrict y,
                       size_t dim,
                       const float *min,
                       const float *max) -> float;
#endif

// ============================================================================
// Runtime Dispatch Functions
// ============================================================================

auto get_ip_sqr_func() -> IpSqrFunc;
auto get_ip_sqr_sq8_func() -> IpSqrSq8Func;
auto get_ip_sqr_sq4_func() -> IpSqrSq4Func;

// ============================================================================
// Public API Templates
// ============================================================================

/**
 * @brief Compute negative inner product between two vectors.
 *
 * Returns -sum(x[i] * y[i]) for use as a distance metric.
 *
 * @tparam DataType Data type (default: float)
 * @tparam DistanceType Return type (default: float)
 * @param x First vector
 * @param y Second vector
 * @param dim Vector dimension
 * @return Negative inner product (smaller = more similar)
 */
template <typename DataType = float, typename DistanceType = float>
auto ip_sqr(const DataType *__restrict x, const DataType *__restrict y, size_t dim) -> DistanceType;

/**
 * @brief Compute negative inner product between two SQ8-encoded vectors.
 *
 * @tparam DataType Data type for min/max (default: float)
 * @tparam DistanceType Return type (default: float)
 * @param x First SQ8-encoded vector
 * @param y Second SQ8-encoded vector
 * @param dim Vector dimension
 * @param min Per-dimension minimum values
 * @param max Per-dimension maximum values
 * @return Negative inner product (smaller = more similar)
 */
template <typename DataType = float, typename DistanceType = float>
auto ip_sqr_sq8(const uint8_t *__restrict x,
                const uint8_t *__restrict y,
                size_t dim,
                const DataType *min,
                const DataType *max) -> DistanceType;

/**
 * @brief Compute negative inner product between two SQ4-encoded vectors.
 *
 * SQ4 stores 2 values per byte (4 bits each):
 *   - Low nibble (bits 0-3) = even index
 *   - High nibble (bits 4-7) = odd index
 *
 * @tparam DataType Data type for min/max (default: float)
 * @tparam DistanceType Return type (default: float)
 * @param x First SQ4-encoded vector
 * @param y Second SQ4-encoded vector
 * @param dim Vector dimension (number of elements, not bytes)
 * @param min Per-dimension minimum values
 * @param max Per-dimension maximum values
 * @return Negative inner product (smaller = more similar)
 */
template <typename DataType = float, typename DistanceType = float>
auto ip_sqr_sq4(const uint8_t *__restrict x,
                const uint8_t *__restrict y,
                size_t dim,
                const DataType *min,
                const DataType *max) -> DistanceType;

}  // namespace alaya::simd

// Implementation
#include "distance_ip.ipp"
