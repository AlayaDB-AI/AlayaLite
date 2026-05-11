// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <type_traits>
#include "cpu_features.hpp"

namespace alaya::simd {

// Type Definitions
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

auto l2_sqr_generic(const float *__restrict x, const float *__restrict y, size_t dim) -> float;
#ifdef ALAYA_ARCH_X86
auto l2_sqr_avx2(const float *__restrict x, const float *__restrict y, size_t dim) -> float;
auto l2_sqr_avx512(const float *__restrict x, const float *__restrict y, size_t dim) -> float;
#endif

auto l2_sqr_sq8_generic(const uint8_t *__restrict x,
                        const uint8_t *__restrict y,
                        size_t dim,
                        const float *min,
                        const float *max) -> float;

#ifdef ALAYA_ARCH_X86
auto l2_sqr_sq8_avx2(const uint8_t *__restrict x,
                     const uint8_t *__restrict y,
                     size_t dim,
                     const float *min,
                     const float *max) -> float;
auto l2_sqr_sq8_avx512(const uint8_t *__restrict x,
                       const uint8_t *__restrict y,
                       size_t dim,
                       const float *min,
                       const float *max) -> float;
#endif

auto l2_sqr_sq4_generic(const uint8_t *__restrict x,
                        const uint8_t *__restrict y,
                        size_t dim,
                        const float *min,
                        const float *max) -> float;

#ifdef ALAYA_ARCH_X86
auto l2_sqr_sq4_avx2(const uint8_t *__restrict x,
                     const uint8_t *__restrict y,
                     size_t dim,
                     const float *min,
                     const float *max) -> float;
auto l2_sqr_sq4_avx512(const uint8_t *__restrict x,
                       const uint8_t *__restrict y,
                       size_t dim,
                       const float *min,
                       const float *max) -> float;
#endif

// Dispatch
auto get_l2_sqr_func() -> L2SqrFunc;
auto get_l2_sqr_sq8_func() -> L2SqrSq8Func;
auto get_l2_sqr_sq4_func() -> L2SqrSq4Func;

// Public API
/**
 * @brief Compute L2 squared distance between two vectors.
 *
 * Returns sum((x[i] - y[i])^2).
 *
 * @tparam DataType Type of input vectors (float).
 * @tparam DistanceType Type of the returned distance (float).
 * @param x Pointer to first input vector.
 * @param y Pointer to second input vector.
 * @param dim Dimensionality of the vectors.
 * @return DistanceType L2 squared distance.
 */
template <typename DataType = float, typename DistanceType = float>
auto l2_sqr(const DataType *__restrict x, const DataType *__restrict y, size_t dim) -> DistanceType;

/**
 * @brief Compute L2 squared distance between two SQ8 quantized vectors.
 *
 * Returns sum(((x[i] - y[i]) * scale[i])^2), where scale[i] = (max[i] - min[i]) / 255.
 *
 * @tparam DataType Type of min/max arrays (float).
 * @tparam DistanceType Type of the returned distance (float).
 * @param x Pointer to first SQ8 quantized vector.
 * @param y Pointer to second SQ8 quantized vector.
 * @param dim Dimensionality of the vectors.
 * @param min Pointer to min array for dequantization.
 * @param max Pointer to max array for dequantization.
 * @return DistanceType L2 squared distance.
 */
template <typename DataType = float, typename DistanceType = float>
auto l2_sqr_sq8(const uint8_t *__restrict x,
                const uint8_t *__restrict y,
                size_t dim,
                const DataType *min,
                const DataType *max) -> DistanceType;

/**
 * @brief Compute L2 squared distance between two SQ4 quantized vectors.
 *
 * Returns sum(((x[i] - y[i]) * scale[i])^2), where scale[i] = (max[i] - min[i]) / 15.
 *
 * @tparam DataType Type of min/max arrays (float).
 * @tparam DistanceType Type of the returned distance (float).
 * @param x Pointer to first SQ4 quantized vector.
 * @param y Pointer to second SQ4 quantized vector.
 * @param dim Dimensionality of the vectors.
 * @param min Pointer to min array for dequantization.
 * @param max Pointer to max array for dequantization.
 * @return DistanceType L2 squared distance.
 */
template <typename DataType = float, typename DistanceType = float>
auto l2_sqr_sq4(const uint8_t *__restrict x,
                const uint8_t *__restrict y,
                size_t dim,
                const DataType *min,
                const DataType *max) -> DistanceType;

}  // namespace alaya::simd

// Implementation
#include "distance_l2.ipp"
