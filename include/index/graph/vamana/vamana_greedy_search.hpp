// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// VamanaGreedySearch — header-only in-memory greedy / beam search over a
// loaded `VamanaReader` graph plus caller-owned float32 row-major vectors.
//
// v1 contract (intentionally narrow; see design.md D6 / D10):
//   * L2 only. The distance kernel is `simd::get_l2_sqr_func()`.
//   * In-memory only. Vectors and graph are provided by the caller; this
//     header does no I/O and owns no on-disk file.
//   * No coupling to the disk-segment subsystem in `include/index/disk/`
//     (no inheritance from disk searchers, no registration with disk
//     factories). The future Vamana adapter for the disk-collection
//     lifecycle is the only place that bridges this search.
//
// Inputs the caller is responsible for:
//   * `reader.graph()` — already validated by the reader.
//   * `vectors` — `num_nodes * dim` float32, row-major. Must outlive the
//     `VamanaGreedySearch` instance.
//   * `query` — `dim` float32. Read but never mutated.
//
// Output: a `std::vector<GreedyHit>` of length at most
// `min(top_k, reader.num_nodes())`, sorted by ascending L2 squared
// distance with ties broken by ascending internal id.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "index/graph/vamana/robust_prune.hpp"  // Neighbor, NeighborPriorityQueue (read-only)
#include "index/graph/vamana/vamana_reader.hpp"
#include "simd/distance_l2.hpp"

namespace alaya::vamana {

// Result of a single greedy search hit: internal node id and the L2
// squared distance to the query.
struct GreedyHit {
  uint32_t id;
  float distance;
};

class VamanaGreedySearch {
 public:
  // The constructor borrows the reader and the vectors buffer. Both must
  // outlive every `search()` call. `vectors` must point to at least
  // `reader.num_nodes() * dim` consecutive float32 values, row-major.
  VamanaGreedySearch(const VamanaReader &reader, const float *vectors, uint32_t dim)
      : reader_(reader), vectors_(vectors), dim_(dim), kernel_(alaya::simd::get_l2_sqr_func()) {}

  // Non-copyable, non-movable — keeps scratch state local-only and
  // matches the reader's ownership contract.
  VamanaGreedySearch(const VamanaGreedySearch &) = delete;
  VamanaGreedySearch &operator=(const VamanaGreedySearch &) = delete;
  VamanaGreedySearch(VamanaGreedySearch &&) = delete;
  VamanaGreedySearch &operator=(VamanaGreedySearch &&) = delete;

  // Run a greedy / beam search starting from the reader's medoid
  // (`reader.start()`). Returns up to `min(top_k, num_nodes())` hits in
  // ascending-distance order with ascending-id tie-break.
  //
  // Length contract: on any graph that is reachable from `reader.start()`
  // — which `VamanaBuilder` always produces — the returned vector has
  // exactly `min(top_k, num_nodes())` entries. On a hand-crafted graph
  // with components unreachable from the medoid, the count may be less.
  // `VamanaReader` validates structural integrity but does not verify
  // medoid reachability, so callers consuming non-builder inputs should
  // treat `at most num_nodes()` as the upper bound.
  //
  // Throws `std::runtime_error` if any of: `top_k == 0`,
  // `search_list_size == 0`, `search_list_size < top_k`.
  std::vector<GreedyHit> search(const float *query, uint32_t top_k, uint32_t search_list_size) {
    if (top_k == 0) {
      throw std::runtime_error("VamanaGreedySearch: top_k = 0 is not supported");
    }
    if (search_list_size == 0) {
      throw std::runtime_error("VamanaGreedySearch: search_list_size = 0 is not supported");
    }
    if (search_list_size < top_k) {
      throw std::runtime_error(
          "VamanaGreedySearch: search_list_size=" + std::to_string(search_list_size) +
          " is smaller than top_k=" + std::to_string(top_k));
    }

    struct SearchScratch {
      std::vector<uint8_t> visited;
      std::vector<uint32_t> visited_touched;
      NeighborPriorityQueue pool;
    };

    thread_local SearchScratch scratch;

    // Clear only the visited entries set in the previous call so reset
    // cost is O(|visited last time|) rather than O(num_nodes).
    if (scratch.visited.size() == reader_.num_nodes()) {
      for (uint32_t id : scratch.visited_touched) {
        scratch.visited[id] = 0;
      }
    } else {
      scratch.visited.assign(reader_.num_nodes(), 0);
    }
    scratch.visited_touched.clear();

    scratch.pool.clear();
    scratch.pool.reserve(search_list_size);

    const auto &graph = reader_.graph();
    const uint32_t start_id = reader_.start();
    const size_t dim_sz = static_cast<size_t>(dim_);

    scratch.visited[start_id] = 1;
    scratch.visited_touched.push_back(start_id);
    const float start_dist =
        kernel_(query, vectors_ + static_cast<size_t>(start_id) * dim_sz, dim_sz);
    scratch.pool.insert(Neighbor(start_id, start_dist));

    while (scratch.pool.has_unexpanded_node()) {
      const Neighbor n = scratch.pool.closest_unexpanded();
      const std::vector<uint32_t> &nbrs = graph[n.id];
      for (uint32_t m : nbrs) {
        if (scratch.visited[m] == 0) {
          scratch.visited[m] = 1;
          scratch.visited_touched.push_back(m);
          const float dist = kernel_(query, vectors_ + static_cast<size_t>(m) * dim_sz, dim_sz);
          scratch.pool.insert(Neighbor(m, dist));
        }
      }
    }

    // NeighborPriorityQueue maintains ascending (distance, id) order via
    // Neighbor::operator< (see robust_prune.hpp), so the prefix is the
    // top-k in spec order without an extra sort.
    const size_t pool_size = scratch.pool.size();
    const size_t take = std::min<size_t>(static_cast<size_t>(top_k), pool_size);
    std::vector<GreedyHit> result;
    result.reserve(take);
    for (size_t i = 0; i < take; ++i) {
      result.push_back(GreedyHit{scratch.pool[i].id, scratch.pool[i].distance});
    }
    {
      std::lock_guard<std::mutex> guard(last_visited_mutex_);
      last_visited_order_ = scratch.visited_touched;
    }
    return result;
  }

  // Accessors for tests / introspection. The first-inserted candidate
  // for any non-empty graph is `reader.start()` (spec scenario "Search
  // visits the medoid first") — exposed as a convenience.
  uint32_t medoid() const { return reader_.start(); }

  // Insertion order of the visited bitset for the most recent
  // `search()` call. Element 0 is always the medoid. Used by the
  // "search starts from medoid" spec scenario; not part of the
  // production query path.
  const std::vector<uint32_t> &last_visited_order() const { return last_visited_order_; }

 private:
  const VamanaReader &reader_;
  const float *vectors_;
  uint32_t dim_;
  alaya::simd::L2SqrFunc kernel_;

  mutable std::mutex last_visited_mutex_;
  std::vector<uint32_t> last_visited_order_;
};

}  // namespace alaya::vamana
