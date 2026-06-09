// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file search_scratch.hpp
 * @brief Per-thread scratch state for the DiskANN cached beam search.
 *
 * Each search thread borrows one @c ThreadData from a @c ConcurrentQueue for
 * the duration of a query (LASER pattern, design D8). It bundles:
 *   - the visited set (dedup of popped nodes),
 *   - the @c retset frontier (NeighborPriorityQueue, reused from Vamana),
 *   - @c exact_by_id: exact L2 distances of nodes actually read from disk/cache,
 *   - @c pq_table: the per-query @c n_chunks x 256 PQ distance table (PQ mode),
 *   - @c sector_scratch: a sector-aligned double buffer for async page reads,
 *   - @c ctx_: the thread's AlignedFileReader I/O context.
 *
 * The scratch is owned by the index and reused across queries; reset_query()
 * clears only the per-query state.
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/laser/utils/aligned_file_reader.hpp"
#include "index/graph/laser/utils/memory.hpp"
#include "index/graph/vamana/robust_prune.hpp"

namespace alaya::diskann {

struct ThreadData {
  // --- Per-query mutable search state ---
  std::unordered_set<uint32_t> visited;             ///< ids popped/seeded
  alaya::vamana::NeighborPriorityQueue retset;      ///< exploration frontier
  std::unordered_map<uint32_t, float> exact_by_id;  ///< read node -> exact L2 sqr
  std::vector<float> pq_table;                      ///< n_chunks*256 (empty if no PQ)

  // --- I/O scratch (allocated once, reused) ---
  char *sector_scratch = nullptr;  ///< n_page_slots * page_size bytes, sector-aligned.
                                   ///< One page slot per in-flight read; the No-PQ pipeline
                                   ///< depth equals the slot count (PQ caps in-flight at
                                   ///< beam_width, so it ignores the surplus).
  uint64_t sector_scratch_bytes = 0;
  IOContext ctx_{};  ///< AIO context (owned via reader.register_thread())

  /// Reset only the per-query state; keeps allocated buffers.
  void reset_query(size_t search_list_size) {
    visited.clear();
    exact_by_id.clear();
    retset.reserve(search_list_size);
    retset.clear();
  }

  /// Allocate the sector page buffer (@p n_page_slots pages, one per concurrent
  /// read) and (optionally) the PQ table.
  void alloc_scratch(uint64_t n_page_slots, uint64_t page_size, uint32_t pq_table_entries) {
    sector_scratch_bytes = n_page_slots * page_size;
    sector_scratch = reinterpret_cast<char *>(
        alaya::laser::memory::align_allocate<kSectorLen>(sector_scratch_bytes));
    if (pq_table_entries > 0) {
      pq_table.assign(pq_table_entries, 0.0f);
    }
  }

  /// Release the sector buffer (PQ table frees with the vector).
  void free_scratch() {
    if (sector_scratch != nullptr) {
      alaya::laser::memory::align_free(sector_scratch);
      sector_scratch = nullptr;
    }
    sector_scratch_bytes = 0;
  }
};

}  // namespace alaya::diskann
