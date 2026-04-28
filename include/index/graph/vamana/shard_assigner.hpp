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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "index/graph/vamana/kmeans_partition.hpp"
#include "utils/log.hpp"
#include "utils/timer.hpp"

namespace alaya::vamana {

// Default streaming block size (number of base vectors held in RAM per read).
// 1M × 128d × 4B = 512 MB per block, which is comfortable on the 64+GB hosts
// we target. Per-block transient distance matrix at num_parts=20 is
// 1M × 20 × 4B = 80 MB — fits in L3-adjacent DRAM without spilling to swap.
// Exposed as a function parameter so the caller can shrink for constrained
// hosts (e.g. BIGANN-100M on 32GB budget with 8 OMP threads may want 256k).
inline constexpr size_t kShardAssignBlockSize = 1'000'000;

// Naming convention mirrors DiskANN's `partition.cpp:262-263`:
//   <prefix>_subshard-<i>.bin                 shard data file (.fbin layout)
//   <prefix>_subshard-<i>_ids_uint32.bin      shard idmap (uint32 global ids)
//
// Using the same names lets downstream tooling (including DiskANN's
// `search_memory_index` for cross-verification in the alignment harness)
// pick up the shards without a naming shim.
inline std::string shard_data_path(const std::filesystem::path &prefix, size_t i) {
  return prefix.string() + "_subshard-" + std::to_string(i) + ".bin";
}
inline std::string shard_idmap_path(const std::filesystem::path &prefix, size_t i) {
  return prefix.string() + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
}

namespace detail {

// A lightweight pair of output streams for one shard. Both files begin with
// a (count, stride) header — stride is `dim` for data files and `1` for
// idmap files (mirrors DiskANN's `.fbin` / `.bin` convention). The count
// field starts as 0 and is rewritten with the final value when the writer
// is finalized.
struct ShardOutputStreams {
  std::ofstream data_os;
  std::ofstream idmap_os;
  uint32_t count = 0;

  ShardOutputStreams(const std::filesystem::path &data_path,
                     const std::filesystem::path &idmap_path,
                     uint32_t dim)
      : data_os(data_path, std::ios::binary | std::ios::trunc),
        idmap_os(idmap_path, std::ios::binary | std::ios::trunc) {
    if (!data_os.is_open()) {
      throw std::runtime_error("shard_assigner: cannot open " + data_path.string());
    }
    if (!idmap_os.is_open()) {
      throw std::runtime_error("shard_assigner: cannot open " + idmap_path.string());
    }
    const uint32_t zero = 0;
    const uint32_t one = 1;
    data_os.write(reinterpret_cast<const char *>(&zero), sizeof(uint32_t));
    data_os.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
    idmap_os.write(reinterpret_cast<const char *>(&zero), sizeof(uint32_t));
    idmap_os.write(reinterpret_cast<const char *>(&one), sizeof(uint32_t));
  }

  void append(const float *vec, uint32_t dim, uint32_t global_id) {
    data_os.write(reinterpret_cast<const char *>(vec),
                  static_cast<std::streamsize>(dim) * sizeof(float));
    idmap_os.write(reinterpret_cast<const char *>(&global_id), sizeof(uint32_t));
    ++count;
  }

  void finalize() {
    data_os.seekp(0, std::ios::beg);
    data_os.write(reinterpret_cast<const char *>(&count), sizeof(uint32_t));
    data_os.close();
    idmap_os.seekp(0, std::ios::beg);
    idmap_os.write(reinterpret_cast<const char *>(&count), sizeof(uint32_t));
    idmap_os.close();
  }
};

}  // namespace detail

// ShardAssignmentResult — returned by `shard_data_by_centroids` so the
// caller can log stats and chain into per-shard Vamana builds.
struct ShardAssignmentResult {
  std::vector<uint32_t> counts;          // per-shard point count (replicated: sum = N * k_base)
  std::vector<std::string> data_paths;   // per-shard data file
  std::vector<std::string> idmap_paths;  // per-shard idmap file
  uint32_t dim = 0;                      // propagated from input for convenience
};

// shard_data_by_centroids — stream a `.fbin` base file, compute the top
// `k_base` nearest centroids for each point, and append the vector +
// global id to the owning shards. Ports DiskANN's
// `shard_data_into_clusters` (partition.cpp:236) with Eigen replacing MKL.
//
// Layout of output files (per shard):
//   <prefix>_subshard-<i>.bin:              uint32 count, uint32 dim, float vectors[count*dim]
//   <prefix>_subshard-<i>_ids_uint32.bin:   uint32 count, uint32 1,   uint32 global_ids[count]
//
// Memory usage is bounded by `block_size * dim * 4B` (input read buffer)
// plus `block_size * num_centers * 4B` (distance matrix), independent of
// the full dataset size. k_base=2 doubles the total bytes written across
// shards versus k_base=1, which is accepted overhead for recall-at-merge.
//
// Return value fields:
//   counts[i]       — exact number of points written to shard i
//   data_paths[i]   — absolute (or prefix-relative) output path for shard data
//   idmap_paths[i]  — output path for shard idmap
//   dim             — dimensionality read from the input `.fbin` header
//
// Throws on any I/O error or header mismatch. On throw, partially written
// shard files are left on disk for post-mortem inspection (caller is
// responsible for cleanup of the prefix directory).
inline ShardAssignmentResult shard_data_by_centroids(const std::filesystem::path &fbin_path,
                                                     const float *centroids,
                                                     size_t num_centers,
                                                     size_t k_base,
                                                     const std::filesystem::path &prefix,
                                                     size_t block_size = kShardAssignBlockSize) {
  if (num_centers == 0 || k_base == 0 || k_base > num_centers) {
    throw std::invalid_argument("shard_data_by_centroids: invalid num_centers / k_base");
  }
  std::filesystem::create_directories(prefix.parent_path());

  std::ifstream in(fbin_path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("shard_data_by_centroids: cannot open " + fbin_path.string());
  }
  uint32_t num_points_u32 = 0;
  uint32_t dim_u32 = 0;
  in.read(reinterpret_cast<char *>(&num_points_u32), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&dim_u32), sizeof(uint32_t));
  if (!in.good() || num_points_u32 == 0 || dim_u32 == 0) {
    throw std::runtime_error("shard_data_by_centroids: corrupt .fbin header at " +
                             fbin_path.string());
  }
  const size_t num_points = num_points_u32;
  const uint32_t dim = dim_u32;

  LOG_INFO("shard_assigner: input={} N={} dim={} num_centers={} k_base={} block={}",
           fbin_path.string(),
           num_points,
           dim,
           num_centers,
           k_base,
           block_size);

  // Open one (data, idmap) pair per shard upfront. std::ofstream is not
  // copyable/movable pre-C++11; use direct vector construction via emplace.
  std::vector<std::unique_ptr<detail::ShardOutputStreams>> shards;
  shards.reserve(num_centers);
  ShardAssignmentResult result;
  result.counts.assign(num_centers, 0);
  result.data_paths.resize(num_centers);
  result.idmap_paths.resize(num_centers);
  result.dim = dim;
  for (size_t i = 0; i < num_centers; ++i) {
    result.data_paths[i] = shard_data_path(prefix, i);
    result.idmap_paths[i] = shard_idmap_path(prefix, i);
    shards.emplace_back(std::make_unique<detail::ShardOutputStreams>(result.data_paths[i],
                                                                     result.idmap_paths[i],
                                                                     dim));
  }

  const size_t effective_block = std::min(block_size, num_points);
  std::vector<float> block_buf(effective_block * dim);
  std::vector<uint32_t> closest(effective_block * k_base);

  alaya::Timer assign_timer;
  assign_timer.reset();

  size_t processed = 0;
  while (processed < num_points) {
    const size_t cur = std::min(effective_block, num_points - processed);
    in.read(reinterpret_cast<char *>(block_buf.data()),
            static_cast<std::streamsize>(cur) * dim * sizeof(float));
    if (static_cast<size_t>(in.gcount()) != cur * dim * sizeof(float)) {
      throw std::runtime_error("shard_data_by_centroids: short read on input .fbin");
    }

    compute_closest_centers(block_buf.data(),
                            cur,
                            dim,
                            centroids,
                            num_centers,
                            k_base,
                            closest.data());

    // Scatter to shards. The writes are sequential per shard (each shard's
    // `ofstream` appends in order), so no intra-shard locking is needed.
    // The outer loop is serial; parallelizing across shards would require
    // per-shard locks on the single ofstream, which typically offers no
    // throughput gain over sequential append (disk-bandwidth bound).
    for (size_t p = 0; p < cur; ++p) {
      const uint32_t gid = static_cast<uint32_t>(processed + p);
      const float *vec = block_buf.data() + p * dim;
      for (size_t t = 0; t < k_base; ++t) {
        const uint32_t shard_id = closest[p * k_base + t];
        shards[shard_id]->append(vec, dim, gid);
        ++result.counts[shard_id];
      }
    }

    processed += cur;
    LOG_INFO("shard_assigner: {}/{} points assigned ({:.1f}%)",
             processed,
             num_points,
             100.0 * static_cast<double>(processed) / static_cast<double>(num_points));
  }
  in.close();

  for (auto &s : shards) {
    s->finalize();
  }

  size_t total_written = 0;
  for (size_t i = 0; i < num_centers; ++i) {
    total_written += result.counts[i];
  }
  LOG_INFO("shard_assigner: done in {:.2f}s, {} × {} k-base → {} total shard entries",
           assign_timer.elapsed_s(),
           num_points,
           k_base,
           total_written);
  if (total_written != num_points * k_base) {
    throw std::runtime_error(
        "shard_data_by_centroids: total shard entries != N * k_base "
        "(post-condition violation)");
  }
  return result;
}

}  // namespace alaya::vamana
