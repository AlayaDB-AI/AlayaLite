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

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "index/graph/vamana/robust_prune.hpp"
#include "simd/distance_l2.hpp"
#include "utils/log.hpp"
#include "utils/timer.hpp"

namespace alaya::vamana {

// Slack factor controlling the soft over-degree cap during in-flight
// neighbor accumulation. Value pinned to 1.3 to match DiskANN's
// `defaults::GRAPH_SLACK_FACTOR`; see spec requirement
// "Inter-insert with GRAPH_SLACK_FACTOR".
inline constexpr float GRAPH_SLACK_FACTOR = 1.3f;

struct VamanaBuildParams {
  uint32_t R = 64;           // graph degree bound
  uint32_t L = 200;          // build-time beam width
  float alpha = 1.2f;        // α-RNG pruning parameter
  uint32_t num_threads = 0;  // 0 → omp_get_num_procs()
  uint32_t maxc = 750;       // occlude_list pool cap
  uint64_t seed = 1234;      // reserved for optional shuffles; medoid is deterministic by data
};

// VamanaBuilder — single-shard in-memory Vamana graph construction on
// float32 L2 data.
//
// Ownership: builder borrows `data` (caller keeps it alive during build()).
// Data layout: row-major `num_points × dim`, contiguous, no padding.
// Output: adjacency as `std::vector<std::vector<uint32_t>>`, accessible
// after build() via `graph()`. Medoid id accessible via `medoid()`.
//
// Why `std::vector<std::vector<uint32_t>>` rather than AlayaLite's
// fixed-arity `Graph`: in-flight degrees temporarily exceed R by up to
// 1.3× (GRAPH_SLACK_FACTOR) before the cleanup pass, so the storage must
// be dynamically sized per node. The writer truncates the header's
// `max_observed_degree` to R on output.
class VamanaBuilder {
 public:
  VamanaBuilder(const float *data, size_t num_points, uint32_t dim, VamanaBuildParams params)
      : data_(data),
        num_points_(num_points),
        dim_(dim),
        params_(params),
        graph_(num_points),
        locks_(num_points),
        l2_(alaya::simd::get_l2_sqr_func()) {
    if (params_.num_threads == 0) {
      params_.num_threads = static_cast<uint32_t>(omp_get_num_procs());
    }
    for (auto &adj : graph_) {
      adj.reserve(static_cast<size_t>(std::ceil(GRAPH_SLACK_FACTOR * params_.R)));
    }
  }

  void build() {
    omp_set_num_threads(static_cast<int>(params_.num_threads));
    calculate_entry_point();
    LOG_INFO("Vamana build: N={}, dim={}, R={}, L={}, alpha={}, threads={}, medoid={}",
             num_points_,
             dim_,
             params_.R,
             params_.L,
             params_.alpha,
             params_.num_threads,
             medoid_);

    // Single-pass link aligned with DiskANN v0.7.0 (df225d3) — the α ramp
    // (cur_alpha ∈ {1.0, ..., alpha}) lives inside occlude_list, not as an
    // outer loop. Previous two-pass dispatch (link(1.0) then link(alpha))
    // left ~8x more orphans because Pass 1's α=1.0 pruning yielded a graph
    // from which Pass 2's greedy search could no longer reach the isolated
    // nodes. See graph_diff log in full_test/stage1_smoke/.
    //
    // Do NOT switch to 2-pass without re-reading the validation study at
    // openspec/changes/archive/2026-04-22-sharded-vamana-validation/design.md.
    // Upstream DiskANN post-v0.7.0 (f198a8a5) added a 2-pass wrapper in
    // build_with_data_populated (index.cpp:1561-1575). Our empirical
    // measurement on GIST 1M (3 shard-count regimes, 2 search paths, NUMA-
    // pinned) shows 2-pass helps only in an extreme-sharding + low-L_search
    // corner that Laser users don't hit; and 1-pass is 2.7× faster to build.
    // Changing this without new data is a regression.
    LOG_INFO("Link pass: alpha={}", params_.alpha);
    link(params_.alpha);
  }

  const std::vector<std::vector<uint32_t>> &graph() const { return graph_; }
  uint32_t medoid() const { return medoid_; }

 private:
  inline float l2_dist(uint32_t a, uint32_t b) const {
    return l2_(data_ + static_cast<size_t>(a) * dim_, data_ + static_cast<size_t>(b) * dim_, dim_);
  }
  inline float l2_dist_to(uint32_t a, const float *q) const {
    return l2_(data_ + static_cast<size_t>(a) * dim_, q, dim_);
  }

  // Per-thread scratch: DiskANN's `InMemQueryScratch` equivalent.
  // Reused across nodes visited by the same thread to avoid repeated
  // heap traffic on the beam search and occlude buffers.
  struct Scratch {
    std::vector<Neighbor> pool;
    NeighborPriorityQueue best_l_nodes;
    std::vector<uint32_t> id_scratch;
    std::vector<float> dist_scratch;
    std::vector<uint8_t> visited_bitset;    // sized num_points, 0/1 flags
    std::vector<uint32_t> visited_touched;  // ids to reset at clear
    std::vector<float> occlude_factor;
  };

  void init_scratches() {
    scratches_.resize(params_.num_threads);
    for (auto &s : scratches_) {
      s.best_l_nodes.reserve(params_.L);
      s.pool.reserve(static_cast<size_t>(std::ceil(1.05 * GRAPH_SLACK_FACTOR * params_.R)) +
                     params_.L);
      s.id_scratch.reserve(params_.L * 4);
      s.dist_scratch.reserve(params_.L * 4);
      s.visited_bitset.assign(num_points_, 0);
      s.visited_touched.reserve(params_.L * 4);
      s.occlude_factor.reserve(params_.maxc);
    }
  }

  static void clear_visited(Scratch &s) {
    for (uint32_t id : s.visited_touched) {
      s.visited_bitset[id] = 0;
    }
    s.visited_touched.clear();
  }

  // medoid = argmin_i sum_j (x_i - centroid)_j^2, where centroid is the
  // arithmetic mean of all vectors. Matches DiskANN's
  // `InMemDataStore::calculate_medoid` (src/in_mem_data_store.cpp:341).
  // Deterministic given the data (no random sampling).
  void calculate_entry_point() {
    std::vector<double> center(dim_, 0.0);
    for (size_t i = 0; i < num_points_; ++i) {
      const float *v = data_ + i * dim_;
      for (uint32_t j = 0; j < dim_; ++j) {
        center[j] += static_cast<double>(v[j]);
      }
    }
    const double inv_n = 1.0 / static_cast<double>(num_points_);
    std::vector<float> centroid(dim_);
    for (uint32_t j = 0; j < dim_; ++j) {
      centroid[j] = static_cast<float>(center[j] * inv_n);
    }

    double best_dist = std::numeric_limits<double>::max();
    uint32_t best_id = 0;
#pragma omp parallel num_threads(static_cast<int>(params_.num_threads))
    {
      double local_best = std::numeric_limits<double>::max();
      uint32_t local_id = 0;
#pragma omp for nowait schedule(static, 65536)
      for (int64_t i = 0; i < static_cast<int64_t>(num_points_); ++i) {
        double d = 0.0;
        const float *v = data_ + static_cast<size_t>(i) * dim_;
        for (uint32_t j = 0; j < dim_; ++j) {
          double diff = static_cast<double>(centroid[j]) - static_cast<double>(v[j]);
          d += diff * diff;
        }
        if (d < local_best) {
          local_best = d;
          local_id = static_cast<uint32_t>(i);
        }
      }
#pragma omp critical
      {
        if (local_best < best_dist) {
          best_dist = local_best;
          best_id = local_id;
        }
      }
    }
    medoid_ = best_id;
  }

  // Greedy beam search from `start_id` toward `query_id`'s vector.
  // Fills `scratch.pool` with expanded nodes (the candidate pool for
  // subsequent pruning) and `scratch.best_l_nodes` with the final top-L.
  // Mirrors DiskANN's `Index::iterate_to_fixed_point` (src/index.cpp:807).
  void iterate_to_fixed_point(uint32_t query_id, uint32_t Lindex, uint32_t start_id, Scratch &s) {
    s.pool.clear();
    s.best_l_nodes.clear();
    s.best_l_nodes.reserve(Lindex);
    clear_visited(s);

    auto visit = [&](uint32_t id) {
      if (s.visited_bitset[id] == 0) {
        s.visited_bitset[id] = 1;
        s.visited_touched.push_back(id);
        return true;
      }
      return false;
    };

    if (visit(start_id)) {
      s.best_l_nodes.insert(Neighbor(start_id, l2_dist(query_id, start_id)));
    }

    while (s.best_l_nodes.has_unexpanded_node()) {
      Neighbor nbr = s.best_l_nodes.closest_unexpanded();
      uint32_t n = nbr.id;
      s.pool.emplace_back(nbr);

      // Snapshot the live adjacency under the node's lock, then release the
      // lock before computing distances (those are the expensive ops).
      std::vector<uint32_t> nbrs;
      {
        std::lock_guard<std::mutex> guard(locks_[n]);
        nbrs = graph_[n];
      }

      s.id_scratch.clear();
      s.dist_scratch.clear();
      // Do NOT skip `m == query_id` here. DiskANN lets the query itself
      // enter the candidate pool when reachable through a neighbor, which
      // populates the pruning pool with the currently-closest candidates.
      // search_for_point_and_prune strips self from the pool right before
      // prune_neighbors, so correctness is preserved. Skipping here caused
      // a ~0.17 avg-degree deficit and ~2x orphan-count excess vs the
      // DiskANN reference; see openspec/changes/port-diskann-vamana Gate 1
      // report.
      for (uint32_t m : nbrs) {
        if (visit(m)) {
          s.id_scratch.push_back(m);
        }
      }
      s.dist_scratch.resize(s.id_scratch.size());
      for (size_t i = 0; i < s.id_scratch.size(); ++i) {
        s.dist_scratch[i] = l2_dist(query_id, s.id_scratch[i]);
      }
      for (size_t i = 0; i < s.id_scratch.size(); ++i) {
        s.best_l_nodes.insert(Neighbor(s.id_scratch[i], s.dist_scratch[i]));
      }
    }
  }

  void search_for_point_and_prune(uint32_t node,
                                  uint32_t Lindex,
                                  std::vector<uint32_t> &pruned_list,
                                  Scratch &s,
                                  float alpha) {
    iterate_to_fixed_point(node, Lindex, medoid_, s);
    // Strip self-id from pool before pruning (see DiskANN index.cpp:1051).
    auto self_it = std::remove_if(s.pool.begin(), s.pool.end(), [node](const Neighbor &nn) {
      return nn.id == node;
    });
    s.pool.erase(self_it, s.pool.end());
    prune_neighbors(node,
                    s.pool,
                    alpha,
                    params_.R,
                    params_.maxc,
                    pruned_list,
                    s.occlude_factor,
                    [this](uint32_t a, uint32_t b) {
                      return l2_dist(a, b);
                    });
  }

  // For each newly minted forward edge n → des, attempt the reverse edge
  // des → n. Fast path (|des_pool| < 1.3R): append without pruning. Slow
  // path: copy des_pool + {n} under lock, prune outside lock, then
  // atomically replace des's adjacency. Mirrors DiskANN's
  // `Index::inter_insert` (src/index.cpp:1216).
  void inter_insert(uint32_t n, const std::vector<uint32_t> &pruned_list, Scratch &s, float alpha) {
    const size_t slack = static_cast<size_t>(GRAPH_SLACK_FACTOR * params_.R);
    for (uint32_t des : pruned_list) {
      std::vector<uint32_t> copy_of_neighbors;
      bool prune_needed = false;
      {
        std::lock_guard<std::mutex> guard(locks_[des]);
        auto &des_pool = graph_[des];
        if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
          if (des_pool.size() < slack) {
            des_pool.push_back(n);
          } else {
            copy_of_neighbors.reserve(des_pool.size() + 1);
            copy_of_neighbors = des_pool;
            copy_of_neighbors.push_back(n);
            prune_needed = true;
          }
        }
      }
      if (prune_needed) {
        // Build a deduplicated pool of {des's current neighbors + n} with
        // distances, then prune. Note: operates on the *snapshot*, so this
        // work happens outside des's lock (which is correct: only the final
        // set_neighbors write needs mutual exclusion).
        s.pool.clear();
        s.occlude_factor.clear();
        s.pool.reserve(copy_of_neighbors.size());
        for (uint32_t cur_nbr : copy_of_neighbors) {
          if (cur_nbr == des) {
            continue;
          }
          bool already_seen = false;
          for (const auto &prev : s.pool) {
            if (prev.id == cur_nbr) {
              already_seen = true;
              break;
            }
          }
          if (!already_seen) {
            s.pool.emplace_back(cur_nbr, l2_dist(des, cur_nbr));
          }
        }
        std::vector<uint32_t> new_neighbors;
        prune_neighbors(des,
                        s.pool,
                        alpha,
                        params_.R,
                        params_.maxc,
                        new_neighbors,
                        s.occlude_factor,
                        [this](uint32_t a, uint32_t b) {
                          return l2_dist(a, b);
                        });
        {
          std::lock_guard<std::mutex> guard(locks_[des]);
          graph_[des] = std::move(new_neighbors);
        }
      }
    }
  }

  void link(float alpha) {
    init_scratches();
    alaya::Timer link_timer;
    link_timer.reset();

#pragma omp parallel for schedule(dynamic, 2048) num_threads(static_cast<int>(params_.num_threads))
    for (int64_t node_ctr = 0; node_ctr < static_cast<int64_t>(num_points_); ++node_ctr) {
      uint32_t node = static_cast<uint32_t>(node_ctr);
      Scratch &s = scratches_[static_cast<size_t>(omp_get_thread_num())];
      std::vector<uint32_t> pruned_list;
      search_for_point_and_prune(node, params_.L, pruned_list, s, alpha);

      {
        std::lock_guard<std::mutex> guard(locks_[node]);
        graph_[node] = pruned_list;
      }

      inter_insert(node, pruned_list, s, alpha);
    }

    // Cleanup: any node whose in-flight degree exceeds R (via inter-insert
    // fast path) gets a fresh prune. Mirrors DiskANN's "final cleanup" loop
    // (src/index.cpp:1355).
#pragma omp parallel for schedule(dynamic, 2048) num_threads(static_cast<int>(params_.num_threads))
    for (int64_t node_ctr = 0; node_ctr < static_cast<int64_t>(num_points_); ++node_ctr) {
      uint32_t node = static_cast<uint32_t>(node_ctr);
      std::vector<uint32_t> snapshot;
      {
        std::lock_guard<std::mutex> guard(locks_[node]);
        if (graph_[node].size() <= params_.R) {
          continue;
        }
        snapshot = graph_[node];
      }
      Scratch &s = scratches_[static_cast<size_t>(omp_get_thread_num())];
      s.pool.clear();
      s.occlude_factor.clear();
      s.pool.reserve(snapshot.size());
      for (uint32_t cur_nbr : snapshot) {
        if (cur_nbr == node) {
          continue;
        }
        bool already_seen = false;
        for (const auto &prev : s.pool) {
          if (prev.id == cur_nbr) {
            already_seen = true;
            break;
          }
        }
        if (!already_seen) {
          s.pool.emplace_back(cur_nbr, l2_dist(node, cur_nbr));
        }
      }
      std::vector<uint32_t> new_neighbors;
      prune_neighbors(node,
                      s.pool,
                      alpha,
                      params_.R,
                      params_.maxc,
                      new_neighbors,
                      s.occlude_factor,
                      [this](uint32_t a, uint32_t b) {
                        return l2_dist(a, b);
                      });
      {
        std::lock_guard<std::mutex> guard(locks_[node]);
        graph_[node] = std::move(new_neighbors);
      }
    }

    LOG_INFO("link(alpha={}) done in {}s", alpha, link_timer.elapsed_s());
  }

  const float *data_;
  size_t num_points_;
  uint32_t dim_;
  VamanaBuildParams params_;
  std::vector<std::vector<uint32_t>> graph_;
  std::vector<std::mutex> locks_;
  std::vector<Scratch> scratches_;
  alaya::simd::L2SqrFunc l2_;
  uint32_t medoid_ = 0;
};

}  // namespace alaya::vamana
