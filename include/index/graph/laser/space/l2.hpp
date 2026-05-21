// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file l2.hpp
 * @brief SIMD-optimized L2 (Euclidean) squared distance computation.
 *
 * Provides AVX512/AVX2 vectorized implementations for computing squared
 * Euclidean distance between vectors. Used for exact distance calculations
 * between query and candidate vectors during k-NN search.
 */

#pragma once

#include <cstddef>

#include "simd/distance_l2.hpp"
#include "simd/laser_dispatch.hpp"

namespace alaya::laser::space {

/**
 * @brief Computes squared L2 distance between two vectors.
 * @param vec0 First vector
 * @param vec1 Second vector
 * @param dim Vector dimension
 * @return ||vec0 - vec1||^2
 */
inline float l2_sqr(const float *__restrict__ vec0, const float *__restrict__ vec1, size_t dim) {
  return ::alaya::simd::l2_sqr<float, float>(vec0, vec1, dim);
}

/** @brief Computes squared L2 norm of a single vector: ||vec0||^2 */
inline float l2_sqr_single(const float *__restrict__ vec0, size_t dim) {
  return simd::get_l2_sqr_single_func()(vec0, dim);
}

}  // namespace alaya::laser::space
