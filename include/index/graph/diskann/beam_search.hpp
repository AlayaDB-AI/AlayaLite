// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file beam_search.hpp
 * @brief Disk search over the DiskANN layout — PQ beam search and No-PQ greedy.
 *
 * Two modes (design D2/D5), dispatched by cached_beam_search():
 *
 *   - PQ mode (use_pq): a beam that pops the closest unexpanded candidates from
 *     the frontier, serves cache hits from memory, and submits cache misses as
 *     async O_DIRECT page reads. Neighbor distances come from the per-query PQ
 *     table, so the frontier is ordered best-first cheaply. Optional rerank reads
 *     the top candidates' coordinates and re-scores them with exact L2.
 *
 *     Two scheduling variants (SearchParams::deterministic): the default
 *     pipelines I/O with compute (up to beam_width reads in flight, processed in
 *     completion order — fastest, but results vary with I/O timing); the opt-in
 *     deterministic variant uses a per-beam barrier (process in popped order,
 *     ~10-15% slower) so concurrent batch_search() matches sequential search()
 *     byte for byte.
 *
 *   - No-PQ mode: a faithful disk port of Vamana greedy search. Expanding a node
 *     reads each unvisited neighbor's coordinates and computes its *exact* L2
 *     distance, so the frontier is ordered by true distance and descends to the
 *     query. This costs more I/O than PQ (every discovered node is read) but
 *     needs no quantization and yields full-precision results. (A lazy
 *     FLT_MAX / parent-distance placeholder was measured to converge far too
 *     slowly — recall < 0.5 even reading 20% of the index — so the neighbor
 *     distances are computed eagerly, matching the in-memory VamanaGreedySearch
 *     model that design D5 names.)
 *
 *     Like PQ, No-PQ has two scheduling variants (SearchParams::deterministic).
 *     Because every distance is exact and the frontier's insert is order-
 *     independent in its final contents, both yield the same recall; they differ
 *     only in how neighbor reads are issued. deterministic = true (batched
 *     barrier) reads each expansion's unvisited neighbors as one concurrent batch
 *     and is byte-for-byte identical to a one-at-a-time synchronous read, so
 *     batch_search() == search(). The default pipelines reads across expansions
 *     (up to 2*beam_width in flight), maximising I/O/compute overlap at the cost
 *     of strict-greedy ordering (only tie-ordering of equally-distant nodes may
 *     differ, not recall).
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/diskann/node_cache.hpp"
#include "index/graph/diskann/pq_table.hpp"
#include "index/graph/diskann/search_scratch.hpp"
#include "index/graph/diskann/tombstone_bitmap.hpp"
#include "index/graph/laser/utils/aligned_file_reader.hpp"
#include "index/graph/vamana/robust_prune.hpp"
#include "simd/distance_l2.hpp"

namespace alaya::diskann {

/// Tunable runtime search parameters.
struct SearchParams {
  uint32_t search_list_size = 100;  ///< L: frontier capacity (>= top_k)
  uint32_t beam_width = 4;          ///< PQ mode: max async reads per beam step
  bool use_pq = true;               ///< use PQ approx distances (needs a PQTable)
  bool rerank = true;               ///< PQ only: re-score top candidates with exact L2
  uint32_t rerank_count = 0;        ///< PQ rerank pool size; 0 => top_k*3 (spec default)
  bool deterministic = false;       ///< Reproducible batch==sequential via a per-expansion
                                    ///< barrier (PQ: per-beam). Default off = async-pipelined
                                    ///< I/O (faster). Applies to both PQ and No-PQ modes.
};

/// Immutable per-index context the search borrows (no ownership).
struct SearchContext {
  AlignedFileReader *reader = nullptr;
  const DiskLayoutGeometry *geom = nullptr;
  const NodeCache *cache = nullptr;
  const PQTable *pq = nullptr;  ///< nullptr => index has no PQ
  uint32_t medoid = 0;
  uint64_t num_points = 0;

  // Tombstone-aware search (null for static indices).
  const TombstoneBitmap *tombstone = nullptr;
};

/// Optional instrumentation for tests / profiling.
struct SearchStats {
  uint64_t n_ios = 0;                ///< page reads issued to disk
  uint64_t n_cache_hits = 0;         ///< nodes served from the BFS cache
  uint64_t n_nodes_processed = 0;    ///< nodes whose exact distance was computed
  uint64_t n_rerank_reads = 0;       ///< extra synchronous reads for PQ rerank
  std::vector<uint32_t> read_order;  ///< ids in the order first processed
};

/**
 * @brief Scan a node's neighbors and insert the unvisited ones into the frontier
 *        with their PQ approximate distances (PQ mode).
 *
 * The frontier's bounded insert prunes neighbors that cannot beat the current
 * worst entry. Already-visited neighbors and out-of-range ids are skipped.
 * Extracted as a free function so the contract is unit-testable in isolation.
 */
inline void scan_and_insert_neighbors(alaya::vamana::NeighborPriorityQueue &retset,
                                      std::unordered_set<uint32_t> &visited,
                                      const uint32_t *nbrs,
                                      uint32_t n_nbrs,
                                      const PQTable &pq,
                                      const float *pq_table,
                                      uint64_t num_points) {
  for (uint32_t k = 0; k < n_nbrs; ++k) {
    const uint32_t m = nbrs[k];
    if (m >= num_points) {
      continue;  // defensive: ignore corrupt neighbor ids
    }
    if (!visited.insert(m).second) {
      continue;
    }
    retset.insert(alaya::vamana::Neighbor(m, pq.pq_distance(m, pq_table)));
  }
}

/**
 * @brief No-PQ disk greedy search: a disk port of Vamana greedy search.
 *
 * Expanding a node reads each unvisited neighbor's coordinates and computes its
 * exact L2 distance, so the frontier (NeighborPriorityQueue of capacity L) is
 * ordered by true distance and the search descends to the query. Each node is
 * read at most once; its neighbor list is cached for when it is expanded.
 *
 * Two I/O scheduling variants share this code (SearchParams::deterministic):
 * a batched per-expansion barrier (byte-exact, batch == sequential) and the
 * default async pipeline (reads overlap across expansions). Because all
 * distances are exact and the frontier insert is order-independent, both give
 * identical recall — see the file header for the contract.
 */
inline std::vector<std::pair<uint32_t, float>> disk_greedy_search(const SearchContext &ctx,
                                                                  const float *query,
                                                                  uint32_t top_k,
                                                                  const SearchParams &params,
                                                                  ThreadData &td,
                                                                  SearchStats *stats) {
  AlignedFileReader &reader = *ctx.reader;
  const DiskLayoutGeometry &geom = *ctx.geom;
  const NodeCache &cache = *ctx.cache;
  const uint64_t dim = geom.dim;
  const uint64_t page_size = geom.page_size;
  const auto l2 = alaya::simd::get_l2_sqr_func();
  // Page slots available in the sector scratch (alloc_scratch sizes it for
  // 2 * beam_width pages); both schedulers cap their in-flight reads at this.
  const uint64_t n_slots = std::max<uint64_t>(1, td.sector_scratch_bytes / page_size);

  const size_t list_size = std::max<size_t>(params.search_list_size, top_k);
  td.reset_query(list_size);
  auto &frontier = td.retset;
  auto &visited = td.visited;

  // Neighbor lists of read nodes, needed when a node is later expanded.
  std::unordered_map<uint32_t, std::vector<uint32_t>> nbrs_cache;

  // Absorb a freshly-read node: cache its neighbor list and insert it into the
  // frontier with its exact L2 distance (coords are co-located in the record).
  auto absorb = [&](uint32_t id, const char *rec) {
    NodeRecordView view{rec, dim};
    nbrs_cache[id].assign(view.nbrs(), view.nbrs() + view.n_nbrs());
    frontier.insert(alaya::vamana::Neighbor(id, l2(query, view.coords(), dim)));
    if (stats != nullptr) {
      stats->read_order.push_back(id);
      stats->n_nodes_processed++;
    }
  };

  // Synchronously read one node into scratch slot 0 (used only to seed the
  // medoid); returns a pointer to its record (cache or scratch).
  auto read_seed = [&](uint32_t id) -> const char * {
    const char *rec = cache.lookup(id);
    if (rec != nullptr) {
      if (stats != nullptr) {
        stats->n_cache_hits++;
      }
      return rec;
    }
    std::vector<AlignedRead> r1;
    r1.emplace_back(geom.get_page_offset(id), page_size, id, td.sector_scratch);
    reader.read(r1, td.ctx_);
    if (stats != nullptr) {
      stats->n_ios++;
    }
    return td.sector_scratch + geom.offset_to_node(id);
  };

  // IP-DiskANN: tombstoned nodes are skipped (graph repaired at delete time).
  const TombstoneBitmap *tomb = ctx.tombstone;
  auto consider = [&](uint32_t m, auto &&emit) {
    if (m >= ctx.num_points || !visited.insert(m).second) {
      return;
    }
    if (tomb != nullptr && tomb->is_deleted(m)) {
      return;
    }
    emit(m);
  };

  visited.insert(ctx.medoid);
  absorb(ctx.medoid, read_seed(ctx.medoid));

  std::vector<AlignedRead> reqs;

  if (params.deterministic) {
    // Batched barrier: expand strictly closest-first. For each expansion, read
    // ALL unvisited neighbors concurrently (chunked to n_slots), wait for the
    // chunk, then absorb in neighbor-list order. Identical insert sequence to a
    // one-at-a-time synchronous read, so concurrent batch_search() reproduces
    // sequential search() byte for byte — it only overlaps the up-to-R reads of
    // each expansion instead of serialising them.
    std::vector<uint32_t> todo;
    std::vector<uint32_t> chunk_ids;
    std::vector<const char *> chunk_recs;
    while (frontier.has_unexpanded_node()) {
      const uint32_t x = frontier.closest_unexpanded().id;
      const auto it = nbrs_cache.find(x);
      if (it == nbrs_cache.end()) {
        continue;  // defensive: a frontier node is always absorbed (and so cached)
      }
      // Copy: absorb() below may rehash nbrs_cache and invalidate it->second.
      const std::vector<uint32_t> nbrs = it->second;
      todo.clear();
      for (const uint32_t m : nbrs) {
        consider(m, [&](uint32_t id) {
          todo.push_back(id);
        });
      }
      for (size_t off = 0; off < todo.size(); off += n_slots) {
        const size_t end = std::min<size_t>(off + n_slots, todo.size());
        reqs.clear();
        chunk_ids.clear();
        chunk_recs.clear();
        uint64_t slot = 0;
        for (size_t i = off; i < end; ++i) {
          const uint32_t m = todo[i];
          const char *crec = cache.lookup(m);
          if (crec != nullptr) {
            if (stats != nullptr) {
              stats->n_cache_hits++;
            }
            chunk_ids.push_back(m);
            chunk_recs.push_back(crec);
            continue;
          }
          char *buf = td.sector_scratch + (slot++) * page_size;
          chunk_ids.push_back(m);
          chunk_recs.push_back(buf + geom.offset_to_node(m));
          reqs.emplace_back(geom.get_page_offset(m), page_size, m, buf);
        }
        if (!reqs.empty()) {
          reader.read(reqs, td.ctx_);  // blocking; kernel runs the batch concurrently
          if (stats != nullptr) {
            stats->n_ios += reqs.size();
          }
        }
        for (size_t j = 0; j < chunk_ids.size(); ++j) {
          absorb(chunk_ids[j], chunk_recs[j]);
        }
      }
    }
  } else {
    // Async pipeline (default): keep up to n_slots reads in flight across
    // expansions. `pending` holds discovered (visited-marked) ids awaiting a slot;
    // `inflight` maps an in-flight id to its (slot, record pointer). Processing
    // follows I/O completion order, so the strict-greedy expansion order is
    // relaxed (results may differ only in tie-ordering of equally-distant nodes).
    std::vector<uint64_t> free_slots;
    free_slots.reserve(n_slots);
    for (uint64_t s = 0; s < n_slots; ++s) {
      free_slots.push_back(s);
    }
    std::unordered_map<uint32_t, std::pair<uint64_t, const char *>> inflight;
    std::deque<uint32_t> pending;
    std::vector<AlignedReadEvent> evts;

    // Expand closest-unexpanded nodes into `pending` until there is enough queued
    // to fill the free slots (relaxed greedy: may expand before a closer in-flight
    // node lands). closest_unexpanded() advances the cursor monotonically, so this
    // always makes progress and cannot loop forever.
    auto refill_pending = [&]() {
      while (pending.size() < free_slots.size() && frontier.has_unexpanded_node()) {
        const uint32_t x = frontier.closest_unexpanded().id;
        const auto it = nbrs_cache.find(x);
        if (it == nbrs_cache.end()) {
          continue;
        }
        for (const uint32_t m : it->second) {
          consider(m, [&](uint32_t id) {
            pending.push_back(id);
          });
        }
      }
    };

    // Drain `pending` into free slots: cache hits absorbed inline (no slot), misses
    // submitted as O_DIRECT reads.
    auto submit_pending = [&]() {
      reqs.clear();
      while (!free_slots.empty() && !pending.empty()) {
        const uint32_t m = pending.front();
        pending.pop_front();
        const char *crec = cache.lookup(m);
        if (crec != nullptr) {
          if (stats != nullptr) {
            stats->n_cache_hits++;
          }
          absorb(m, crec);
          continue;
        }
        const uint64_t slot = free_slots.back();
        free_slots.pop_back();
        char *buf = td.sector_scratch + slot * page_size;
        inflight.emplace(m, std::make_pair(slot, buf + geom.offset_to_node(m)));
        reqs.emplace_back(geom.get_page_offset(m), page_size, m, buf);
      }
      if (!reqs.empty()) {
        reader.submit_reqs(reqs, td.ctx_);
        if (stats != nullptr) {
          stats->n_ios += reqs.size();
        }
      }
    };

    refill_pending();
    submit_pending();
    // Continue while any work remains: a read in flight, a queued read, or an
    // unexpanded frontier node (the latter matters when cache hits add nodes
    // without ever entering `inflight`).
    while (!inflight.empty() || !pending.empty() || frontier.has_unexpanded_node()) {
      if (!inflight.empty()) {
        reader.get_events(td.ctx_, 1, evts);  // get_events clears + fills `evts`
        for (const auto &e : evts) {
          if (e.result != static_cast<int64_t>(page_size)) {
            throw std::runtime_error("disk_greedy_search: short/failed read, result=" +
                                     std::to_string(e.result));
          }
          const auto it = inflight.find(static_cast<uint32_t>(e.id));
          if (it == inflight.end()) {
            continue;  // defensive: completion for an id we are not tracking
          }
          const uint64_t slot = it->second.first;
          const char *rec = it->second.second;
          absorb(static_cast<uint32_t>(e.id), rec);
          free_slots.push_back(slot);
          inflight.erase(it);
        }
      }
      refill_pending();
      submit_pending();
    }
  }

  std::vector<std::pair<uint32_t, float>> out;
  out.reserve(std::min<size_t>(frontier.size(), top_k));
  for (size_t i = 0; i < frontier.size() && out.size() < top_k; ++i) {
    const uint32_t id = frontier[i].id;
    if (tomb != nullptr && tomb->is_deleted(id)) {
      continue;
    }
    out.emplace_back(id, frontier[i].distance);
  }
  return out;
}

/**
 * @brief Run the disk search for one query (PQ beam or No-PQ greedy).
 *
 * @return Up to @p top_k (internal_id, distance) pairs sorted by ascending
 *         distance. Distances are exact L2-squared except for PQ no-rerank
 *         results on nodes that were never read (those carry their PQ
 *         approximation).
 */
inline std::vector<std::pair<uint32_t, float>> cached_beam_search(const SearchContext &ctx,
                                                                  const float *query,
                                                                  uint32_t top_k,
                                                                  const SearchParams &params,
                                                                  ThreadData &td,
                                                                  SearchStats *stats = nullptr) {
  const bool use_pq = params.use_pq && (ctx.pq != nullptr);
  if (!use_pq) {
    return disk_greedy_search(ctx, query, top_k, params, td, stats);
  }

  AlignedFileReader &reader = *ctx.reader;
  const DiskLayoutGeometry &geom = *ctx.geom;
  const NodeCache &cache = *ctx.cache;
  const PQTable &pq = *ctx.pq;
  const uint64_t dim = geom.dim;
  const uint64_t page_size = geom.page_size;
  const auto l2 = alaya::simd::get_l2_sqr_func();

  const size_t list_size = std::max<size_t>(params.search_list_size, top_k);
  td.reset_query(list_size);
  auto &retset = td.retset;
  auto &visited = td.visited;
  auto &exact_by_id = td.exact_by_id;

  pq.preprocess_query(query, td.pq_table.data());
  const float *pq_table = td.pq_table.data();

  // Seed from the medoid (first node expanded — spec scenario).
  visited.insert(ctx.medoid);
  retset.insert(alaya::vamana::Neighbor(ctx.medoid, pq.pq_distance(ctx.medoid, pq_table)));

  const uint64_t beam = std::max<uint64_t>(1, params.beam_width);

  // Process a freshly-read node: compute its exact L2 (the coords are co-located
  // on the sector we just read) and insert its unvisited neighbors into the
  // frontier with PQ approximate distances. Shared by both search paths.
  auto process_node = [&](uint32_t id, const char *rec) {
    NodeRecordView view{rec, dim};
    exact_by_id[id] = l2(query, view.coords(), dim);
    if (stats != nullptr) {
      stats->read_order.push_back(id);
      stats->n_nodes_processed++;
    }
    scan_and_insert_neighbors(retset,
                              visited,
                              view.nbrs(),
                              view.n_nbrs(),
                              pq,
                              pq_table,
                              ctx.num_points);
  };

  std::vector<AlignedRead> reqs;
  std::vector<AlignedReadEvent> evts;

  if (params.deterministic) {
    // Deterministic per-beam barrier: pop the closest unexpanded candidates, read
    // the cache misses together, wait for ALL of them, then process the batch in
    // popped order. Processing order is independent of I/O completion timing, so
    // concurrent batch_search() matches sequential search() byte for byte. Opt-in
    // (DiskANNSearchParams::deterministic) — it forgoes the cross-beam I/O/compute
    // overlap of the default path and so runs ~10-15% slower.
    std::vector<uint32_t> batch;
    std::unordered_map<uint32_t, const char *> rec_of;
    while (retset.has_unexpanded_node()) {
      batch.clear();
      reqs.clear();
      rec_of.clear();
      uint64_t slot = 0;
      while (retset.has_unexpanded_node() && reqs.size() < beam) {
        const uint32_t id = retset.closest_unexpanded().id;
        batch.push_back(id);
        const char *crec = cache.lookup(id);
        if (crec != nullptr) {
          rec_of[id] = crec;
          if (stats != nullptr) {
            stats->n_cache_hits++;
          }
        } else {
          char *buf = td.sector_scratch + (slot++) * page_size;
          rec_of[id] = buf + geom.offset_to_node(id);
          reqs.emplace_back(geom.get_page_offset(id), page_size, id, buf);
        }
      }

      if (!reqs.empty()) {
        reader.submit_reqs(reqs, td.ctx_);
        reader.get_events(td.ctx_, static_cast<int>(reqs.size()), evts);
        if (stats != nullptr) {
          stats->n_ios += reqs.size();
        }
        for (const auto &e : evts) {
          if (e.result != static_cast<int64_t>(page_size)) {
            throw std::runtime_error("cached_beam_search: short/failed read, result=" +
                                     std::to_string(e.result));
          }
        }
      }

      for (const uint32_t id : batch) {
        process_node(id, rec_of[id]);
      }
    }
  } else {
    // Async pipelined beam (default): keep up to `beam` reads in flight and
    // process each as it completes, refilling the pipe immediately, so disk I/O
    // overlaps with distance computation. Processing follows I/O completion order,
    // so results are NOT bitwise-reproducible across runs/threads (set
    // deterministic for that). Thread-safety, correctness and recall are
    // unaffected — only the tie-ordering of equally-good candidates differs.
    std::vector<uint64_t> free_slots;
    free_slots.reserve(beam);
    for (uint64_t s = 0; s < beam; ++s) {
      free_slots.push_back(s);
    }
    // In-flight reads: node id -> (scratch slot, record pointer within that page).
    std::unordered_map<uint32_t, std::pair<uint64_t, const char *>> inflight;

    // Pop closest-unexpanded candidates, serving cache hits inline and submitting
    // cache misses into free scratch slots. Stops when the slots are full or the
    // frontier has no unexpanded node left.
    auto fill_pipe = [&]() {
      reqs.clear();
      while (!free_slots.empty() && retset.has_unexpanded_node()) {
        const uint32_t id = retset.closest_unexpanded().id;
        const char *crec = cache.lookup(id);
        if (crec != nullptr) {
          if (stats != nullptr) {
            stats->n_cache_hits++;
          }
          process_node(id, crec);
          continue;
        }
        const uint64_t slot = free_slots.back();
        free_slots.pop_back();
        char *buf = td.sector_scratch + slot * page_size;
        inflight.emplace(id, std::make_pair(slot, buf + geom.offset_to_node(id)));
        reqs.emplace_back(geom.get_page_offset(id), page_size, id, buf);
      }
      if (!reqs.empty()) {
        reader.submit_reqs(reqs, td.ctx_);
        if (stats != nullptr) {
          stats->n_ios += reqs.size();
        }
      }
    };

    fill_pipe();
    while (!inflight.empty()) {
      // Wait for exactly one completion (get_events clears and fills `evts`). The
      // other in-flight reads keep the disk busy while we process this one — that
      // is the I/O/compute overlap. NOTE: poll_events/get_events both clear their
      // out-vector, so they must not share one; we use get_events alone here.
      reader.get_events(td.ctx_, 1, evts);
      for (const auto &e : evts) {
        if (e.result != static_cast<int64_t>(page_size)) {
          throw std::runtime_error("cached_beam_search: short/failed read, result=" +
                                   std::to_string(e.result));
        }
        const auto it = inflight.find(static_cast<uint32_t>(e.id));
        if (it == inflight.end()) {
          continue;  // defensive: completion for an id we are not tracking
        }
        const uint64_t slot = it->second.first;
        const char *rec = it->second.second;
        process_node(static_cast<uint32_t>(e.id), rec);
        free_slots.push_back(slot);
        inflight.erase(it);
      }
      fill_pipe();
    }
  }

  // ---------------- Result extraction (PQ) ----------------
  std::vector<std::pair<uint32_t, float>> out;
  std::vector<AlignedRead> rerank_req(1);

  auto read_exact_sync = [&](uint32_t id) -> float {
    const auto it = exact_by_id.find(id);
    if (it != exact_by_id.end()) {
      return it->second;
    }
    char *buf = td.sector_scratch;
    rerank_req[0] = {geom.get_page_offset(id), page_size, id, buf};
    reader.read(rerank_req, td.ctx_);
    NodeRecordView view{buf + geom.offset_to_node(id), dim};
    const float ex = l2(query, view.coords(), dim);
    exact_by_id[id] = ex;
    if (stats != nullptr) {
      stats->n_rerank_reads++;
    }
    return ex;
  };

  if (params.rerank) {
    // Re-score the top PQ candidates with exact L2.
    const size_t want =
        params.rerank_count > 0 ? params.rerank_count : static_cast<size_t>(top_k) * 3;
    const size_t n_cand = std::min<size_t>(retset.size(), want);
    out.reserve(n_cand);
    for (size_t i = 0; i < n_cand; ++i) {
      const uint32_t id = retset[i].id;
      out.emplace_back(id, read_exact_sync(id));
    }
  } else {
    // Top candidates ranked by PQ distance (exact where known during traversal).
    const size_t n_cand = std::min<size_t>(retset.size(), static_cast<size_t>(top_k));
    out.reserve(n_cand);
    for (size_t i = 0; i < n_cand; ++i) {
      const uint32_t id = retset[i].id;
      const auto it = exact_by_id.find(id);
      out.emplace_back(id, it != exact_by_id.end() ? it->second : retset[i].distance);
    }
  }

  std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  });
  if (out.size() > top_k) {
    out.resize(top_k);
  }
  return out;
}

}  // namespace alaya::diskann
