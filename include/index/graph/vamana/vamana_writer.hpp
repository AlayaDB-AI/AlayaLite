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
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace alaya::vamana {

// save_graph — write a Vamana adjacency to a single-file DiskANN `.index`
// binary. Layout (native byte order, no padding), matching
// `InMemGraphStore::save_graph` in DiskANN (src/in_mem_graph_store.cpp:200):
//
//   offset 0   size 8   uint64_t expected_file_size
//   offset 8   size 4   uint32_t max_observed_degree  (= R)
//   offset 12  size 4   uint32_t start                (medoid id)
//   offset 16  size 8   uint64_t frozen_pts           (0 for this port)
//   offset 24  ...      per-node records in id order 0..N-1:
//                         uint32_t k
//                         uint32_t neighbors[k]
//
// The header is written first with a placeholder `expected_file_size`, then
// the per-node records append to the stream. After the stream is closed we
// `tellp` for the final size and seek back to rewrite bytes 0..7 with the
// true size (two-phase write mirrors DiskANN's implementation).
inline void save_graph(const std::vector<std::vector<uint32_t>> &graph,
                       const std::filesystem::path &path,
                       uint32_t max_degree,
                       uint32_t start,
                       uint64_t frozen_pts = 0) {
  const auto parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("save_graph: cannot open for writing: " + path.string());
  }

  uint64_t index_size = 24;  // header bytes; updated after per-node write
  const uint32_t max_degree_written = max_degree;

  out.write(reinterpret_cast<const char *>(&index_size), sizeof(uint64_t));
  out.write(reinterpret_cast<const char *>(&max_degree_written), sizeof(uint32_t));
  out.write(reinterpret_cast<const char *>(&start), sizeof(uint32_t));
  out.write(reinterpret_cast<const char *>(&frozen_pts), sizeof(uint64_t));

  for (const auto &adj : graph) {
    const uint32_t k = static_cast<uint32_t>(adj.size());
    out.write(reinterpret_cast<const char *>(&k), sizeof(uint32_t));
    if (k > 0) {
      out.write(reinterpret_cast<const char *>(adj.data()),
                static_cast<std::streamsize>(k) * sizeof(uint32_t));
    }
    index_size += sizeof(uint32_t) * (static_cast<size_t>(k) + 1);
  }

  // Rewrite the expected_file_size at offset 0 now that we know the total.
  out.seekp(0, std::ios::beg);
  out.write(reinterpret_cast<const char *>(&index_size), sizeof(uint64_t));
  if (!out.good()) {
    throw std::runtime_error("save_graph: write error on " + path.string());
  }
  out.close();
}

}  // namespace alaya::vamana
