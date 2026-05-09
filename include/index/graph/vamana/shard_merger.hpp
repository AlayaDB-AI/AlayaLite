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
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "utils/log.hpp"
#include "utils/timer.hpp"

namespace alaya::vamana {

namespace detail {

// Read a DiskANN-format idmap file written by shard_assigner:
//   uint32_t count
//   uint32_t stride (== 1)
//   uint32_t ids[count]
inline std::vector<uint32_t> read_idmap(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("read_idmap: cannot open " + path.string());
  }
  uint32_t count = 0;
  uint32_t stride = 0;
  in.read(reinterpret_cast<char *>(&count), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&stride), sizeof(uint32_t));
  if (!in.good() || stride != 1) {
    throw std::runtime_error("read_idmap: corrupt header or stride != 1: " + path.string());
  }
  std::vector<uint32_t> ids(count);
  if (count > 0) {
    in.read(reinterpret_cast<char *>(ids.data()),
            static_cast<std::streamsize>(count) * sizeof(uint32_t));
    if (static_cast<size_t>(in.gcount()) != count * sizeof(uint32_t)) {
      throw std::runtime_error("read_idmap: short read: " + path.string());
    }
  }
  return ids;
}

}  // namespace detail

// compute_medoid_streaming — two-pass streaming medoid of a `.fbin` file.
// Pass 1: accumulate the per-dimension sum across all N vectors to derive
// the centroid; pass 2: find `argmin_i ||x_i − centroid||²`. Matches the
// semantics of `VamanaBuilder::calculate_entry_point` but operates on the
// file (for datasets too large to hold in memory).
//
// block_size bounds peak RAM to `block_size * dim * 4B`. At 1M × 128d = 512MB.
inline uint32_t compute_medoid_streaming(const std::filesystem::path &fbin_path,
                                         size_t block_size = 1'000'000) {
  std::ifstream in(fbin_path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("compute_medoid_streaming: cannot open " + fbin_path.string());
  }
  uint32_t num_points_u32 = 0;
  uint32_t dim_u32 = 0;
  in.read(reinterpret_cast<char *>(&num_points_u32), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&dim_u32), sizeof(uint32_t));
  if (!in.good() || num_points_u32 == 0 || dim_u32 == 0) {
    throw std::runtime_error("compute_medoid_streaming: corrupt .fbin header: " +
                             fbin_path.string());
  }
  const size_t N = num_points_u32;
  const uint32_t dim = dim_u32;

  // Pass 1: streaming centroid.
  std::vector<double> centroid_accum(dim, 0.0);
  std::vector<float> buf(std::min(block_size, N) * dim);
  const std::streamoff data_start = static_cast<std::streamoff>(2 * sizeof(uint32_t));
  in.seekg(data_start);
  size_t read_so_far = 0;
  while (read_so_far < N) {
    const size_t cur = std::min(block_size, N - read_so_far);
    in.read(reinterpret_cast<char *>(buf.data()),
            static_cast<std::streamsize>(cur) * dim * sizeof(float));
    for (size_t p = 0; p < cur; ++p) {
      const float *v = buf.data() + p * dim;
      for (uint32_t d = 0; d < dim; ++d) {
        centroid_accum[d] += static_cast<double>(v[d]);
      }
    }
    read_so_far += cur;
  }
  std::vector<float> centroid(dim);
  const double inv_n = 1.0 / static_cast<double>(N);
  for (uint32_t d = 0; d < dim; ++d) {
    centroid[d] = static_cast<float>(centroid_accum[d] * inv_n);
  }

  // Pass 2: streaming argmin of ||x_i − centroid||².
  in.clear();
  in.seekg(data_start);
  double best_dist = std::numeric_limits<double>::max();
  uint32_t best_id = 0;
  read_so_far = 0;
  while (read_so_far < N) {
    const size_t cur = std::min(block_size, N - read_so_far);
    in.read(reinterpret_cast<char *>(buf.data()),
            static_cast<std::streamsize>(cur) * dim * sizeof(float));

    double local_best = std::numeric_limits<double>::max();
    uint32_t local_id = 0;
#pragma omp parallel
    {
      double th_best = std::numeric_limits<double>::max();
      uint32_t th_id = 0;
#pragma omp for nowait
      for (int64_t p = 0; p < static_cast<int64_t>(cur); ++p) {
        const float *v = buf.data() + static_cast<size_t>(p) * dim;
        double acc = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
          const double diff = static_cast<double>(v[d]) - static_cast<double>(centroid[d]);
          acc += diff * diff;
        }
        if (acc < th_best) {
          th_best = acc;
          th_id = static_cast<uint32_t>(read_so_far + static_cast<size_t>(p));
        }
      }
#pragma omp critical
      {
        if (th_best < local_best) {
          local_best = th_best;
          local_id = th_id;
        }
      }
    }
    if (local_best < best_dist) {
      best_dist = local_best;
      best_id = local_id;
    }
    read_so_far += cur;
  }
  return best_id;
}

// merge_shards — read per-shard Vamana `.index` files + idmaps and produce
// a single merged `.index` by unioning neighbor lists (translated to global
// ids), shuffling with a seeded `std::mt19937`, and truncating to R.
//
// DiskANN reference: `disk_utils.cpp:241`. Per spec D5 the merge uses only
// `union + shuffle + cut` — no RobustPrune — so that the recall drift
// envelope tracks the reference implementation.
//
// Inputs:
//   shard_graph_paths[i]   per-shard `.index` output from VamanaBuilder + save_graph
//   shard_idmap_paths[i]   shard-local id → global id map (uint32 array)
//   output_path            merged `.index` destination
//   R                      truncation target for each merged adjacency
//   medoid                 global id to write into the output header's `start`
//   seed                   mt19937 seed for the shuffle step
//   frozen_pts             propagated to the output header (0 for this port)
//
// Preconditions:
//   - Each shard's idmap is sorted ascending by global id. `shard_assigner`
//     satisfies this by construction (it streams input in id order and
//     appends to shard files).
//   - Shard graph files use AlayaLite's `save_graph` layout
//     (24-byte header + per-node [k, nbrs[k]] records in shard-local id
//     order 0..shard_size-1).
//
// Determinism: same (seed, shard inputs, medoid) yields byte-identical
// output. `std::shuffle` with `std::mt19937(seed)` is reproducible across
// compilers and platforms (the algorithm is mandated by the C++ standard).
inline uint64_t merge_shards(const std::vector<std::filesystem::path> &shard_graph_paths,
                             const std::vector<std::filesystem::path> &shard_idmap_paths,
                             const std::filesystem::path &output_path,
                             uint32_t R,
                             uint32_t medoid,
                             uint64_t seed,
                             uint64_t frozen_pts = 0) {
  const size_t nshards = shard_graph_paths.size();
  if (nshards == 0 || shard_idmap_paths.size() != nshards) {
    throw std::invalid_argument("merge_shards: require ≥ 1 shard and paired graph/idmap paths");
  }

  alaya::Timer merge_timer;
  merge_timer.reset();

  // Load idmaps up front. For 100M × k_base=2 total shard entries, this is
  // ~800MB — acceptable vs the alternative of random-access seek inside
  // idmap files during the merge loop.
  std::vector<std::vector<uint32_t>> idmaps(nshards);
  size_t max_global_id = 0;
  size_t total_shard_entries = 0;
  for (size_t s = 0; s < nshards; ++s) {
    idmaps[s] = detail::read_idmap(shard_idmap_paths[s]);
    for (uint32_t gid : idmaps[s]) {
      if (static_cast<size_t>(gid) > max_global_id) {
        max_global_id = gid;
      }
    }
    total_shard_entries += idmaps[s].size();
    if (!std::is_sorted(idmaps[s].begin(), idmaps[s].end())) {
      throw std::runtime_error("merge_shards: idmap is not sorted ascending (shard " +
                               std::to_string(s) + "); shard_assigner invariant violated");
    }
  }
  const size_t N = max_global_id + 1;
  LOG_INFO("merge_shards: N={}, nshards={}, total_shard_entries={} (avg k_base={:.2f})",
           N,
           nshards,
           total_shard_entries,
           static_cast<double>(total_shard_entries) / static_cast<double>(N));

  // Open per-shard graph readers. Skip the 24-byte header; subsequent reads
  // advance through per-node `[k, nbrs[k]]` records. Because each shard's
  // idmap is sorted ascending and we iterate global ids ascending, each
  // shard's reader is consumed in shard-local order 0..shard_size-1.
  //
  // As we open each reader, also harvest the shard-local medoid from the
  // header (offset 12, uint32) and translate to a global id via the shard's
  // idmap — DiskANN's `merge_shards` (disk_utils.cpp:389) uses these to
  // populate the `_medoids.bin` auxiliary file written at the end.
  std::vector<std::ifstream> readers(nshards);
  std::vector<size_t> shard_cursor(nshards, 0);
  std::vector<uint32_t> shard_medoids_global(nshards, 0);
  for (size_t s = 0; s < nshards; ++s) {
    readers[s].open(shard_graph_paths[s], std::ios::binary);
    if (!readers[s].is_open()) {
      throw std::runtime_error("merge_shards: cannot open shard graph " +
                               shard_graph_paths[s].string());
    }
    readers[s].seekg(12, std::ios::beg);
    uint32_t shard_local_medoid = 0;
    readers[s].read(reinterpret_cast<char *>(&shard_local_medoid), sizeof(uint32_t));
    if (!readers[s].good()) {
      throw std::runtime_error("merge_shards: cannot read shard medoid: " +
                               shard_graph_paths[s].string());
    }
    if (shard_local_medoid >= idmaps[s].size()) {
      throw std::runtime_error("merge_shards: shard " + std::to_string(s) + " medoid (local " +
                               std::to_string(shard_local_medoid) +
                               ") is out of range for idmap size " +
                               std::to_string(idmaps[s].size()));
    }
    shard_medoids_global[s] = idmaps[s][shard_local_medoid];
    readers[s].seekg(24, std::ios::beg);  // skip to per-node records
  }

  // Open output. Reserve 24 bytes for the header; patch bytes 0..7 after
  // we know the final file size. Matches `vamana_writer::save_graph`.
  const auto output_parent = output_path.parent_path();
  if (!output_parent.empty()) {
    std::filesystem::create_directories(output_parent);
  }
  std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("merge_shards: cannot open output " + output_path.string());
  }
  uint64_t expected_file_size = 24;
  out.write(reinterpret_cast<const char *>(&expected_file_size), sizeof(uint64_t));
  out.write(reinterpret_cast<const char *>(&R), sizeof(uint32_t));
  out.write(reinterpret_cast<const char *>(&medoid), sizeof(uint32_t));
  out.write(reinterpret_cast<const char *>(&frozen_pts), sizeof(uint64_t));

  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::unordered_set<uint32_t> seen;
  seen.reserve(static_cast<size_t>(R) * 4);  // typical k_base*R, rounded
  std::vector<uint32_t> nhood;
  nhood.reserve(static_cast<size_t>(R) * 4);

  size_t missing_nodes = 0;
  for (uint32_t gid = 0; gid < N; ++gid) {
    seen.clear();
    nhood.clear();

    // Gather neighbor contributions from every shard that contains this gid.
    // Uses sequential reads: each shard_cursor[s] advances by 1 when we
    // consume a record. Because idmap[s] is sorted ascending, the next
    // record in the shard's file corresponds to the shard-local position
    // shard_cursor[s], which maps to global id `idmaps[s][shard_cursor[s]]`.
    bool any_shard = false;
    for (size_t s = 0; s < nshards; ++s) {
      if (shard_cursor[s] >= idmaps[s].size()) {
        continue;
      }
      if (idmaps[s][shard_cursor[s]] != gid) {
        continue;
      }
      any_shard = true;
      uint32_t k = 0;
      readers[s].read(reinterpret_cast<char *>(&k), sizeof(uint32_t));
      if (!readers[s].good()) {
        throw std::runtime_error("merge_shards: read error on shard " + std::to_string(s) +
                                 " at shard-local id " + std::to_string(shard_cursor[s]));
      }
      if (k > 0) {
        std::vector<uint32_t> local_nbrs(k);
        readers[s].read(reinterpret_cast<char *>(local_nbrs.data()),
                        static_cast<std::streamsize>(k) * sizeof(uint32_t));
        for (uint32_t lnb : local_nbrs) {
          if (lnb >= idmaps[s].size()) {
            throw std::runtime_error("merge_shards: out-of-range shard-local neighbor " +
                                     std::to_string(lnb) + " in shard " + std::to_string(s));
          }
          const uint32_t gnb = idmaps[s][lnb];
          if (gnb == gid) {
            continue;  // drop self-loops
          }
          if (seen.insert(gnb).second) {
            nhood.push_back(gnb);
          }
        }
      }
      ++shard_cursor[s];
    }
    if (!any_shard) {
      ++missing_nodes;
    }

    // Shuffle (mandated seed) then truncate to R. Mirrors DiskANN
    // `disk_utils.cpp:421-422`.
    std::shuffle(nhood.begin(), nhood.end(), rng);
    uint32_t out_k = static_cast<uint32_t>(std::min<size_t>(nhood.size(), R));
    out.write(reinterpret_cast<const char *>(&out_k), sizeof(uint32_t));
    if (out_k > 0) {
      out.write(reinterpret_cast<const char *>(nhood.data()),
                static_cast<std::streamsize>(out_k) * sizeof(uint32_t));
    }
    expected_file_size += sizeof(uint32_t) * (static_cast<uint64_t>(out_k) + 1);
  }

  // Patch the expected_file_size header.
  out.seekp(0, std::ios::beg);
  out.write(reinterpret_cast<const char *>(&expected_file_size), sizeof(uint64_t));
  if (!out.good()) {
    throw std::runtime_error("merge_shards: write error on output");
  }
  out.close();

  // DiskANN-format medoids file: uint32 nshards, uint32 1 (stride),
  // nshards × uint32 global_medoid. Path = `<output>_medoids.bin`,
  // matching disk_utils.cpp:376's destination. Consumers (DiskANN SSD
  // search, PQ flash index) use this for multi-entry-point init.
  const std::filesystem::path medoids_path(output_path.string() + "_medoids.bin");
  {
    std::ofstream mf(medoids_path, std::ios::binary | std::ios::trunc);
    if (!mf.is_open()) {
      throw std::runtime_error("merge_shards: cannot open medoids file: " + medoids_path.string());
    }
    const uint32_t nshards_u32 = static_cast<uint32_t>(nshards);
    const uint32_t one_stride = 1;
    mf.write(reinterpret_cast<const char *>(&nshards_u32), sizeof(uint32_t));
    mf.write(reinterpret_cast<const char *>(&one_stride), sizeof(uint32_t));
    mf.write(reinterpret_cast<const char *>(shard_medoids_global.data()),
             static_cast<std::streamsize>(nshards) * sizeof(uint32_t));
    if (!mf.good()) {
      throw std::runtime_error("merge_shards: write error on " + medoids_path.string());
    }
  }
  LOG_INFO("merge_shards: wrote medoids to {} ({} entries)", medoids_path.string(), nshards);

  if (missing_nodes > 0) {
    LOG_WARN("merge_shards: {} node id(s) absent from all shards (wrote k=0)", missing_nodes);
  }
  LOG_INFO("merge_shards: wrote {} bytes to {} in {:.2f}s, medoid={}, R={}",
           expected_file_size,
           output_path.string(),
           merge_timer.elapsed_s(),
           medoid,
           R);
  return expected_file_size;
}

}  // namespace alaya::vamana
