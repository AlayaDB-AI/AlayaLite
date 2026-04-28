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
#include <mutex>
#include <stdexcept>
#include <vector>

#include "index/graph/vamana/kmeans_partition.hpp"
#include "utils/log.hpp"

namespace alaya::vamana {

// DiskANN pinned constants, replicated here so callers don't need to know
// the upstream header layout. OVERHEAD_FACTOR is the 10% allocator/scratch
// slack from `DiskANN/include/index.h:28`. GRAPH_SLACK_FACTOR matches the
// 1.3× transient over-degree cap also used in `vamana_builder.hpp`.
inline constexpr double kOverheadFactor = 1.1;
inline constexpr float kBudgetGraphSlackFactor = 1.3f;

// Round `x` up to the nearest multiple of 8. Equivalent to DiskANN's
// `ROUND_UP(X, 8)` macro. Captures the 8-float / 32-byte cache-line pad
// of the in-memory data store.
inline constexpr uint32_t round_up_8(uint32_t x) noexcept { return ((x + 7U) / 8U) * 8U; }

// estimate_ram_usage_bytes — DiskANN's `estimate_ram_usage` formula,
// returning bytes. Inputs:
//   num_points    : N
//   dim           : D (unpadded)
//   dtype_size    : sizeof(T) for the vector element (4 for float)
//   graph_degree  : R
//
// Breakdown:
//   size_of_data         = N * round_up_8(D) * dtype_size
//   size_of_graph        = N * R * 4 * GRAPH_SLACK_FACTOR
//   size_of_locks        = N * sizeof(std::mutex)
//   size_of_outer_vector = N * sizeof(ptrdiff_t)
// RAM = 1.1 * (sum)
//
// The `size_of_locks` term captures the per-node mutex array DiskANN
// allocates for concurrent `inter_insert` writes. At 100M × ~40-byte mutex
// this is ~4GB — not negligible, and must be included for our shard-count
// growth sequence to match DiskANN's reference partitioning.
inline double estimate_ram_usage_bytes(size_t num_points,
                                       uint32_t dim,
                                       uint32_t dtype_size,
                                       uint32_t graph_degree) {
  const double n = static_cast<double>(num_points);
  const double size_of_data =
      n * static_cast<double>(round_up_8(dim)) * static_cast<double>(dtype_size);
  const double size_of_graph = n * static_cast<double>(graph_degree) *
                               static_cast<double>(sizeof(uint32_t)) *
                               static_cast<double>(kBudgetGraphSlackFactor);
  const double size_of_locks = n * static_cast<double>(sizeof(std::mutex));
  const double size_of_outer = n * static_cast<double>(sizeof(std::ptrdiff_t));
  return kOverheadFactor * (size_of_data + size_of_graph + size_of_locks + size_of_outer);
}

// Convenience wrapper: budget expressed in gibibytes.
inline double estimate_ram_usage_gib(size_t num_points,
                                     uint32_t dim,
                                     uint32_t dtype_size,
                                     uint32_t graph_degree) {
  return estimate_ram_usage_bytes(num_points, dim, dtype_size, graph_degree) /
         (1024.0 * 1024.0 * 1024.0);
}

// Parameters for `determine_num_parts_with_ram_budget`. Kept as a small
// aggregate so the driver signature stays readable (design heuristic:
// more than 4 positional args → options struct).
struct BudgetLoopParams {
  uint32_t graph_degree = 64;  // Vamana R
  uint32_t dtype_size = sizeof(float);
  size_t k_base = 2;             // overlap factor for shard assignment
  double sampling_rate = 0.01;   // fraction of base data used for train/test
  double ram_budget_gib = 32.0;  // per-shard build budget (GiB)
  KMeansParams base_kmeans{};    // .seed, .max_reps carry through; .num_centers is overridden
  // Safety cap on the growth loop. 1024 is enough for GIST 1M at 0.1 GiB
  // budget (which lands near ~250 shards) plus comfortable headroom for
  // BIGANN-100M-class experiments. Raise further only if a real workload
  // needs it; a runaway loop at > 1024 almost always indicates a
  // misconfigured budget (too small) or sampling rate (too low → kmeans
  // can't balance).
  size_t max_num_parts = 1024;
};

// determine_num_parts_with_ram_budget — DiskANN's
// `partition_with_ram_budget` growth loop, expressed as a self-contained
// orchestrator. Starts at `num_parts = 3`, runs k-means++ + Lloyd, uses
// `estimate_cluster_sizes` on the test sample to estimate per-shard
// counts, evaluates max RAM via `estimate_ram_usage_bytes`, and grows
// `num_parts += 2` until every shard fits within `ram_budget_gib`.
//
// Inputs:
//   train_data   : row-major sample (num_train × dim) for k-means training
//   test_data    : row-major sample (num_test × dim) for size estimation.
//                  Independent draw from `train_data` in DiskANN.
//   dim          : vector dimensionality
//   p            : loop configuration (see BudgetLoopParams)
//   pivots_out   : resized on each iteration to `num_parts * dim`; final
//                  contents are the frozen centroid matrix.
//
// Returns the chosen `num_parts`. Throws if `max_num_parts` is hit without
// the budget being satisfied (almost always indicates a misconfigured
// budget — too low, or graph_degree too high for the dataset scale).
//
// Determinism: `kmeans_train` reseeds `std::mt19937_64(base_kmeans.seed)`
// on each growth iteration, so the growth trajectory is fully reproducible
// given the same (seed, train_data, test_data, budget) tuple.
inline size_t determine_num_parts_with_ram_budget(const float *train_data,
                                                  size_t num_train,
                                                  const float *test_data,
                                                  size_t num_test,
                                                  size_t dim,
                                                  const BudgetLoopParams &p,
                                                  std::vector<float> &pivots_out) {
  if (p.ram_budget_gib <= 0.0) {
    throw std::invalid_argument("determine_num_parts_with_ram_budget: ram_budget_gib must be > 0");
  }
  if (num_train == 0 || num_test == 0) {
    throw std::invalid_argument("determine_num_parts_with_ram_budget: empty train/test sample");
  }
  if (p.sampling_rate <= 0.0 || p.sampling_rate > 1.0) {
    throw std::invalid_argument("determine_num_parts_with_ram_budget: sampling_rate out of range");
  }

  const double ram_budget_bytes = p.ram_budget_gib * 1024.0 * 1024.0 * 1024.0;

  size_t num_parts = 3;
  while (num_parts <= p.max_num_parts) {
    if (num_parts > num_train) {
      throw std::runtime_error(
          "determine_num_parts_with_ram_budget: num_parts exceeds training "
          "sample size; increase sampling_rate or data size");
    }

    KMeansParams params = p.base_kmeans;
    params.num_centers = num_parts;
    pivots_out.assign(num_parts * dim, 0.0f);
    kmeans_train(train_data, num_train, dim, params, pivots_out.data());

    const std::vector<size_t> extrapolated = estimate_cluster_sizes(test_data,
                                                                    num_test,
                                                                    dim,
                                                                    pivots_out.data(),
                                                                    num_parts,
                                                                    p.k_base,
                                                                    p.sampling_rate);

    double max_shard_ram = 0.0;
    for (size_t c = 0; c < num_parts; ++c) {
      const double ram = estimate_ram_usage_bytes(extrapolated[c],
                                                  static_cast<uint32_t>(dim),
                                                  p.dtype_size,
                                                  p.graph_degree);
      if (ram > max_shard_ram) {
        max_shard_ram = ram;
      }
    }
    const double max_shard_gib = max_shard_ram / (1024.0 * 1024.0 * 1024.0);
    LOG_INFO("budget: num_parts={} max_shard_ram={:.3f}GiB budget={:.3f}GiB",
             num_parts,
             max_shard_gib,
             p.ram_budget_gib);
    if (max_shard_ram <= ram_budget_bytes) {
      LOG_INFO("budget: freezing num_parts={}", num_parts);
      return num_parts;
    }
    num_parts += 2;
  }
  throw std::runtime_error(
      "determine_num_parts_with_ram_budget: max_num_parts hit without "
      "satisfying budget; raise --build_dram_budget or lower R");
}

}  // namespace alaya::vamana
