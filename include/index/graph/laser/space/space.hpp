// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file space.hpp
 * @brief High-level vector space utilities for graph construction.
 *
 * Provides:
 * - exact_nn: Brute-force nearest neighbor search (for finding entry points)
 * - compute_centroid: Parallel centroid computation
 * - matrix_padding: Pad matrix to required dimension alignment
 */

#pragma once

#include <omp.h>

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <vector>

#include "index/graph/laser/common.hpp"
#include "index/graph/laser/utils/tools.hpp"

namespace alaya::laser::space {

/** @brief Finds exact nearest neighbor via brute-force parallel scan. */
inline PID exact_nn(const float *data,
                    const float *query,
                    size_t num_points,
                    size_t dim,
                    size_t num_threads,
                    const DistFunc<float> &dist_func_) {
  std::vector<Candidate<float>> best_entries(num_threads, Candidate(0, FLT_MAX));

  const int64_t num_points_signed = static_cast<int64_t>(num_points);
#pragma omp parallel for schedule(dynamic)
  for (int64_t ii = 0; ii < num_points_signed; ++ii) {
    const size_t i = static_cast<size_t>(ii);
    auto tid = omp_get_thread_num();
    Candidate<float> &cur_entry = best_entries[tid];
    const float *cur_data = data + (dim * i);

    float distance = dist_func_(cur_data, query, dim);
    if (distance < cur_entry.distance) {
      cur_entry.id = static_cast<PID>(i);
      cur_entry.distance = distance;
    }
  }

  PID nearest_neighbor = 0;
  float min_dist = FLT_MAX;
  for (auto &candi : best_entries) {
    if (candi.distance < min_dist) {
      nearest_neighbor = candi.id;
      min_dist = candi.distance;
    }
  }
  return nearest_neighbor;
}

/** @brief Computes centroid (mean) of all vectors in parallel. */
inline auto compute_centroid(const float *data, size_t num_points, size_t dim, size_t num_threads) {
  std::vector<std::vector<double>> all_results(num_threads, std::vector<double>(dim, 0));

  const int64_t num_points_signed = static_cast<int64_t>(num_points);
#pragma omp parallel for schedule(dynamic)
  for (int64_t ii = 0; ii < num_points_signed; ++ii) {
    const size_t i = static_cast<size_t>(ii);
    auto tid = omp_get_thread_num();
    std::vector<double> &cur_results = all_results[tid];
    const float *cur_data = data + (dim * i);
    for (size_t k = 0; k < dim; ++k) {
      cur_results[k] += cur_data[k];
    }
  }

  std::vector<double> centroid_d(dim, 0);
  std::vector<float> centroid(dim, 0);
  for (auto &one_res : all_results) {
    for (size_t i = 0; i < dim; ++i) {
      centroid_d[i] += one_res[i];
    }
  }
  double inv_num_points = 1.0 / static_cast<double>(num_points);
  for (size_t i = 0; i < dim; ++i) {
    centroid[i] = static_cast<float>(centroid_d[i] * inv_num_points);
  }

  return centroid;
}

/** @brief Pads matrix columns to required alignment (e.g., power of 2). */
template <typename T>
inline void matrix_padding(const RowMatrix<T> &mat, RowMatrix<T> &padded_mat, size_t padded_dim) {
  size_t num_rows = mat.rows();
  size_t dim = mat.cols();

  padded_mat = RowMatrix<T>(num_rows, padded_dim);
  padded_mat.setZero();

  for (size_t i = 0; i < num_rows; ++i) {
    std::copy(&mat(i, 0), &mat(i, 0) + dim, &padded_mat(i, 0));
  }
}
}  // namespace alaya::laser::space
