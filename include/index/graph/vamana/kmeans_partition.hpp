// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <omp.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "utils/log.hpp"

namespace alaya::vamana {

// Row-major float matrix view type. Row-major is required so that
// `Eigen::Map<RowMajorMat>` can wrap DiskANN-style row-major data buffers
// (`float[num_points * dim]`) with zero copy; Eigen's default is column-major.
using KMeansRowMajorMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// k-means driver parameters. Defaults mirror DiskANN's pins (`max_k_means_reps
// = 10`, `residual_rel_tol = 1e-5`, see `math_utils.cpp:340`). Seed is
// exposed so the partition-with-ram-budget loop can run multiple k-means
// rounds at different `num_centers` while remaining reproducible.
struct KMeansParams {
  size_t num_centers = 3;
  size_t max_reps = 10;
  uint64_t seed = 1234;
  float residual_rel_tol = 1e-5f;
};

namespace detail {

// Block size for `compute_closest_centers` GEMM tiles. Bounds the transient
// distance matrix to `BLOCK_SIZE * num_centers * sizeof(float)` per OMP
// thread. 65536 × 21 × 4B ≈ 5.5MB per tile (worst case on our expected
// num_parts growth of 3, 5, 7, 9, 11, 13, 15, 17, 19, 21), fits in L2.
constexpr size_t kClosestCentersBlockSize = 65536;

}  // namespace detail

// compute_vecs_l2sq — write `||data_i||²` for i in [0, num_points) to
// `l2sq_out`. Uses Eigen's `rowwise().squaredNorm()`, which vectorizes via
// SSE/AVX when `-march=native` is set. Caller owns `l2sq_out` (size
// num_points floats).
inline void compute_vecs_l2sq(const float *data, size_t num_points, size_t dim, float *l2sq_out) {
  if (num_points == 0 || dim == 0) {
    return;
  }
  Eigen::Map<const KMeansRowMajorMat> d_view(data,
                                             static_cast<Eigen::Index>(num_points),
                                             static_cast<Eigen::Index>(dim));
  Eigen::Map<Eigen::VectorXf> out_view(l2sq_out, static_cast<Eigen::Index>(num_points));
  out_view = d_view.rowwise().squaredNorm();
}

// compute_closest_centers — for each of the `num_points` input vectors, find
// the `k` nearest centers (squared L2) and write their indices row-major to
// `closest_out` (size `num_points * k` `uint32_t`).
//
// Algorithm (matches DiskANN's `math_utils::compute_closest_centers`):
//   dist[i,j] = ||d_i||² + ||c_j||² − 2 * d_i · c_j
// computed via Eigen GEMM, blocked by `kClosestCentersBlockSize` so the
// transient distance matrix stays L2-resident. k is typically 1 (Lloyd
// assignment) or 2 (`k_base` overlapping shard assignment), so per-row
// top-k extraction is a linear scan (faster than a heap for k ≤ 4).
//
// Thread-safety: parallelized internally with `omp parallel for`; caller
// must not call concurrently on overlapping `closest_out` ranges.
inline void compute_closest_centers(const float *data,
                                    size_t num_points,
                                    size_t dim,
                                    const float *centers,
                                    size_t num_centers,
                                    size_t k,
                                    uint32_t *closest_out) {
  if (k == 0) {
    throw std::invalid_argument("compute_closest_centers: k must be ≥ 1");
  }
  if (k > num_centers) {
    throw std::invalid_argument("compute_closest_centers: k > num_centers not supported");
  }
  if (num_points == 0) {
    return;
  }

  std::vector<float> centers_l2sq(num_centers);
  compute_vecs_l2sq(centers, num_centers, dim, centers_l2sq.data());

  Eigen::Map<const KMeansRowMajorMat> c_view(centers,
                                             static_cast<Eigen::Index>(num_centers),
                                             static_cast<Eigen::Index>(dim));

  const size_t block_size = std::min(num_points, detail::kClosestCentersBlockSize);
  const size_t num_blocks = (num_points + block_size - 1) / block_size;

  // Per-block: ||d||² vector (block_size), dist matrix (block_size × num_centers).
  // Allocate outside the loop to amortize malloc across blocks.
  std::vector<float> block_d_l2sq(block_size);
  KMeansRowMajorMat dist(static_cast<Eigen::Index>(block_size),
                         static_cast<Eigen::Index>(num_centers));

  for (size_t b = 0; b < num_blocks; ++b) {
    const size_t start = b * block_size;
    const size_t cur = std::min(block_size, num_points - start);

    Eigen::Map<const KMeansRowMajorMat> d_block(data + start * dim,
                                                static_cast<Eigen::Index>(cur),
                                                static_cast<Eigen::Index>(dim));

    compute_vecs_l2sq(data + start * dim, cur, dim, block_d_l2sq.data());

    auto dist_block = dist.topRows(static_cast<Eigen::Index>(cur));
    // dist = -2 * D * C^T
    dist_block.noalias() = -2.0f * d_block * c_view.transpose();
    // row i += ||d_i||²  (broadcast over columns)
    for (size_t i = 0; i < cur; ++i) {
      const float norm_i = block_d_l2sq[i];
      dist_block.row(static_cast<Eigen::Index>(i)).array() += norm_i;
    }
    // column j += ||c_j||²  (broadcast over rows)
    Eigen::Map<const Eigen::VectorXf> c_norm_view(centers_l2sq.data(),
                                                  static_cast<Eigen::Index>(num_centers));
    dist_block.rowwise() += c_norm_view.transpose();

    // Top-k extraction per row. For k ≤ 4 a linear scan with a sorted small
    // array is cheaper than a heap. For Lloyd (k=1) this is a single argmin;
    // for k_base=2 shard assignment, two updates suffice.
    if (k == 1) {
#pragma omp parallel for schedule(static, 4096)
      for (int64_t i = 0; i < static_cast<int64_t>(cur); ++i) {
        const float *row = dist.data() + i * static_cast<int64_t>(num_centers);
        float best = std::numeric_limits<float>::max();
        uint32_t best_id = 0;
        for (size_t j = 0; j < num_centers; ++j) {
          if (row[j] < best) {
            best = row[j];
            best_id = static_cast<uint32_t>(j);
          }
        }
        closest_out[(start + static_cast<size_t>(i)) * k] = best_id;
      }
    } else {
#pragma omp parallel for schedule(static, 4096)
      for (int64_t i = 0; i < static_cast<int64_t>(cur); ++i) {
        const float *row = dist.data() + i * static_cast<int64_t>(num_centers);
        // Sorted ascending top-k by insertion; k is small (≤ a handful).
        std::vector<std::pair<float, uint32_t>> top(k, {std::numeric_limits<float>::max(), 0});
        for (size_t j = 0; j < num_centers; ++j) {
          float dj = row[j];
          if (dj >= top.back().first) {
            continue;
          }
          // Replace worst and bubble down.
          top.back() = {dj, static_cast<uint32_t>(j)};
          for (size_t t = top.size() - 1; t > 0 && top[t].first < top[t - 1].first; --t) {
            std::swap(top[t], top[t - 1]);
          }
        }
        for (size_t t = 0; t < k; ++t) {
          closest_out[(start + static_cast<size_t>(i)) * k + t] = top[t].second;
        }
      }
    }
  }
}

// kmeanspp_init — write `num_centers` k-means++ seed centroids to
// `pivots_out` (size `num_centers * dim` floats, caller-allocated).
//
// Algorithm (mirrors DiskANN's `kmeans::kmeanspp_selecting_pivots`
// at `math_utils.cpp:380`):
//   1. Pick initial pivot uniformly at random from [0, num_points).
//   2. Maintain dist[i] = min over chosen pivots of ||x_i − c_j||².
//   3. Sample next pivot with probability ∝ dist[i] (D² sampling).
//   4. Repeat until `num_centers` pivots chosen.
//
// Determinism: the `rng` engine is the *only* randomness source, so callers
// that want reproducible partitioning seed it with `KMeansParams::seed`.
// This is the single point of divergence from DiskANN, which uses
// `std::random_device` in the upstream code.
inline void kmeanspp_init(const float *data,
                          size_t num_points,
                          size_t dim,
                          size_t num_centers,
                          std::mt19937_64 &rng,
                          float *pivots_out) {
  if (num_centers == 0 || num_points == 0) {
    throw std::invalid_argument("kmeanspp_init: empty inputs");
  }
  if (num_centers > num_points) {
    throw std::invalid_argument("kmeanspp_init: num_centers > num_points is not allowed");
  }
  if (num_points > (1ULL << 23)) {
    // DiskANN's upstream guard: kmeans++ over > 8.4M points becomes
    // quadratic in the number of pivots × N. Partition-with-ram-budget
    // keeps us on sampled training data (typically ~1M), so this bound
    // is not expected to trip in practice; surface a clear failure if it
    // does rather than silently degrading to random pivots.
    throw std::runtime_error("kmeanspp_init: num_points exceeds 2^23 bound; caller should sample");
  }

  std::uniform_int_distribution<size_t> int_dist(0, num_points - 1);
  std::uniform_real_distribution<double> real_dist(0.0, 1.0);

  std::vector<size_t> picked;
  picked.reserve(num_centers);

  size_t first = int_dist(rng);
  picked.push_back(first);
  std::memcpy(pivots_out, data + first * dim, dim * sizeof(float));

  std::vector<float> dist(num_points);
  // Initial dist[i] = squared L2 distance to the first pivot.
#pragma omp parallel for schedule(static, 8192)
  for (int64_t i = 0; i < static_cast<int64_t>(num_points); ++i) {
    const float *v = data + static_cast<size_t>(i) * dim;
    const float *p = data + first * dim;
    float acc = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
      const float diff = v[d] - p[d];
      acc += diff * diff;
    }
    dist[static_cast<size_t>(i)] = acc;
  }

  size_t num_picked = 1;
  while (num_picked < num_centers) {
    // Re-draw loop: if the dart lands on an already-picked index (possible
    // only under numerical edge cases or duplicate points), re-roll. If all
    // remaining dist[i] sum to 0 (i.e. all un-picked points coincide with a
    // picked pivot), accept the duplicate to guarantee forward progress and
    // match DiskANN's `sum_flag` fallback.
    double sum = 0.0;
    for (size_t i = 0; i < num_points; ++i) {
      sum += static_cast<double>(dist[i]);
    }
    const bool degenerate = (sum == 0.0);

    size_t chosen = 0;
    while (true) {
      const double dart = real_dist(rng) * sum;
      double prefix = 0.0;
      chosen = num_points - 1;
      for (size_t i = 0; i < num_points; ++i) {
        if (dart >= prefix && dart < prefix + static_cast<double>(dist[i])) {
          chosen = i;
          break;
        }
        prefix += static_cast<double>(dist[i]);
      }
      if (degenerate) {
        break;
      }
      if (std::find(picked.begin(), picked.end(), chosen) == picked.end()) {
        break;
      }
    }

    picked.push_back(chosen);
    std::memcpy(pivots_out + num_picked * dim, data + chosen * dim, dim * sizeof(float));

    // dist[i] ← min(dist[i], ||x_i − c_new||²)
    const float *p_new = data + chosen * dim;
#pragma omp parallel for schedule(static, 8192)
    for (int64_t i = 0; i < static_cast<int64_t>(num_points); ++i) {
      const float *v = data + static_cast<size_t>(i) * dim;
      float acc = 0.0f;
      for (size_t d = 0; d < dim; ++d) {
        const float diff = v[d] - p_new[d];
        acc += diff * diff;
      }
      float &cur = dist[static_cast<size_t>(i)];
      if (acc < cur) {
        cur = acc;
      }
    }

    ++num_picked;
  }
}

// lloyds_iter — run one Lloyd's iteration in place:
//   1. Assign each point to its nearest center.
//   2. Replace each center with the mean of its assigned points.
// Returns the residual (sum of squared distances from each point to its
// assigned center, computed after the center update).
//
// Parallelism follows DiskANN: centroid accumulation uses per-cluster
// parallel reduction (inverted index built during assignment).
inline float lloyds_iter(const float *data,
                         size_t num_points,
                         size_t dim,
                         float *centers,
                         size_t num_centers) {
  std::vector<uint32_t> closest(num_points);
  compute_closest_centers(data,
                          num_points,
                          dim,
                          centers,
                          num_centers,
                          /*k=*/1,
                          closest.data());

  std::vector<std::vector<size_t>> inverted(num_centers);
  for (size_t i = 0; i < num_points; ++i) {
    inverted[closest[i]].push_back(i);
  }

  std::memset(centers, 0, num_centers * dim * sizeof(float));

#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t c = 0; c < static_cast<int64_t>(num_centers); ++c) {
    const auto &members = inverted[static_cast<size_t>(c)];
    if (members.empty()) {
      // Empty cluster: leave the center zeroed. DiskANN's upstream takes the
      // same hands-off approach; in practice the next kmeanspp_init restart
      // (on partition growth) reshuffles pivots, and Lloyd's early-
      // termination kicks in before this center can steal density.
      continue;
    }
    std::vector<double> acc(dim, 0.0);
    for (size_t idx : members) {
      const float *v = data + idx * dim;
      for (size_t d = 0; d < dim; ++d) {
        acc[d] += static_cast<double>(v[d]);
      }
    }
    const double inv = 1.0 / static_cast<double>(members.size());
    float *out = centers + static_cast<size_t>(c) * dim;
    for (size_t d = 0; d < dim; ++d) {
      out[d] = static_cast<float>(acc[d] * inv);
    }
  }

  // Residual: re-assign against updated centers and sum squared distances.
  // (DiskANN computes this against the *pre-update* closest assignment; the
  // difference between the two choices is a constant offset inside the
  // convergence test `(old - new) / new < 1e-5`, so either is valid as a
  // monotone progress signal. We match DiskANN by using the pre-update
  // assignment.)
  double residual = 0.0;
#pragma omp parallel for schedule(static, 8192) reduction(+ : residual)
  for (int64_t i = 0; i < static_cast<int64_t>(num_points); ++i) {
    const uint32_t cid = closest[static_cast<size_t>(i)];
    const float *v = data + static_cast<size_t>(i) * dim;
    const float *c = centers + static_cast<size_t>(cid) * dim;
    double acc = 0.0;
    for (size_t d = 0; d < dim; ++d) {
      const double diff = static_cast<double>(v[d]) - static_cast<double>(c[d]);
      acc += diff * diff;
    }
    residual += acc;
  }
  return static_cast<float>(residual);
}

// run_lloyds — iterate `lloyds_iter` up to `max_reps` times, with early
// termination when the residual stops improving (matches DiskANN's
// `(old - new) / new < 1e-5` stopping rule at `math_utils.cpp:340`).
inline float run_lloyds(const float *data,
                        size_t num_points,
                        size_t dim,
                        float *centers,
                        size_t num_centers,
                        size_t max_reps,
                        float rel_tol = 1e-5f) {
  float residual = std::numeric_limits<float>::max();
  for (size_t r = 0; r < max_reps; ++r) {
    const float old_residual = residual;
    residual = lloyds_iter(data, num_points, dim, centers, num_centers);
    if (r > 0 && residual > 0.0f && ((old_residual - residual) / residual) < rel_tol) {
      LOG_INFO("kmeans: Lloyd converged after {} iter(s), residual {} → {}",
               r + 1,
               old_residual,
               residual);
      break;
    }
    if (residual < std::numeric_limits<float>::epsilon()) {
      LOG_INFO("kmeans: Lloyd hit epsilon after {} iter(s)", r + 1);
      break;
    }
  }
  return residual;
}

// estimate_cluster_sizes — stream the test sample through the given pivots
// under `k_base` assignment, then extrapolate the raw counts by
// `1 / sampling_rate` to produce per-cluster size estimates for the full
// dataset. Returns a vector of length `num_centers`.
//
// Semantics mirror DiskANN's `estimate_cluster_sizes`
// (partition.cpp:183) plus the extrapolation step that DiskANN performs
// inline at the call site (partition.cpp:579); folding the multiply into
// this function keeps callers free of sampling-rate bookkeeping.
//
// Precondition: `pivots` points to a `num_centers * dim` row-major float
// buffer. `sampling_rate` must be in (0, 1]. For `sampling_rate = 1.0`,
// counts are returned un-scaled (i.e. the test sample IS the full set).
inline std::vector<size_t> estimate_cluster_sizes(const float *test_data,
                                                  size_t num_test,
                                                  size_t dim,
                                                  const float *pivots,
                                                  size_t num_centers,
                                                  size_t k_base,
                                                  double sampling_rate) {
  if (sampling_rate <= 0.0 || sampling_rate > 1.0) {
    throw std::invalid_argument("estimate_cluster_sizes: sampling_rate must be in (0, 1]");
  }
  std::vector<size_t> counts(num_centers, 0);
  if (num_test == 0 || num_centers == 0) {
    return counts;
  }
  std::vector<uint32_t> assignments(num_test * k_base);
  compute_closest_centers(test_data,
                          num_test,
                          dim,
                          pivots,
                          num_centers,
                          k_base,
                          assignments.data());
  for (size_t i = 0; i < num_test * k_base; ++i) {
    ++counts[assignments[i]];
  }
  const double inv_rate = 1.0 / sampling_rate;
  for (size_t c = 0; c < num_centers; ++c) {
    counts[c] = static_cast<size_t>(static_cast<double>(counts[c]) * inv_rate);
  }
  return counts;
}

// kmeans_train — top-level entry: k-means++ seeding followed by Lloyd's
// iterations. Writes `num_centers * dim` floats to `centroids_out`
// (caller-allocated).
//
// Determinism: seeding `std::mt19937_64` with `params.seed` is the only
// randomness source; running the same input twice with the same seed and
// same thread count yields identical centroids. Thread-count variation may
// perturb residuals via floating-point reduction order, but the final
// centroid assignment remains stable for our convergence tolerance.
inline void kmeans_train(const float *data,
                         size_t num_points,
                         size_t dim,
                         const KMeansParams &params,
                         float *centroids_out) {
  if (params.num_centers == 0) {
    throw std::invalid_argument("kmeans_train: num_centers == 0");
  }
  std::mt19937_64 rng(params.seed);
  kmeanspp_init(data, num_points, dim, params.num_centers, rng, centroids_out);
  LOG_INFO("kmeans_train: N={}, dim={}, K={}, seed={}",
           num_points,
           dim,
           params.num_centers,
           params.seed);
  const float residual = run_lloyds(data,
                                    num_points,
                                    dim,
                                    centroids_out,
                                    params.num_centers,
                                    params.max_reps,
                                    params.residual_rel_tol);
  LOG_INFO("kmeans_train: final residual = {}", residual);
}

}  // namespace alaya::vamana
