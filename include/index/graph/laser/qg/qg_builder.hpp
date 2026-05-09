/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <sys/types.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "index/graph/laser/common.hpp"
#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/space/space.hpp"
#include "index/graph/laser/utils/aligned_file_reader.hpp"
#include "index/graph/laser/utils/tools.hpp"
#include "third_party/ngt/hashset.hpp"

namespace alaya::laser {
constexpr size_t kMaxBsIter = 5;
using CandidateList = std::vector<Candidate<float>>;

class QGBuilder {
 private:
  QuantizedGraph &qg_;
  size_t ef_build_;
  size_t num_threads_;
  size_t num_nodes_;
  size_t dim_;
  size_t degree_bound_;
  size_t max_candidate_pool_size_ = 750;
  size_t max_pruned_size_ = 300;
  DistFunc<float> dist_func_;
  std::vector<CandidateList> new_neighbors_;
  std::vector<CandidateList> pruned_neighbors_;
  std::vector<HashBasedBooleanSet> visited_list_;
  std::vector<uint32_t> degrees_;
  void random_init();
  void init_from_vamana(const std::string &filename);
  void search_new_neighbors(bool refine);
  void heuristic_prune(PID, CandidateList &, CandidateList &, bool);
  void add_reverse_edges(PID data_id, std::vector<std::mutex> &, bool);
  void add_pruned_edges(const CandidateList &, const CandidateList &, CandidateList &, float);
  void graph_refine();
  void iter(bool);

 public:
  explicit QGBuilder(QuantizedGraph &index, uint32_t ef_build, size_t num_threads)
      : qg_{index},
        ef_build_{ef_build},
        num_threads_{std::min(num_threads, total_threads() * 4)},
        num_nodes_{qg_.num_vertices()},
        dim_{qg_.dimension()},
        degree_bound_(qg_.degree_bound()),
        dist_func_{space::l2_sqr},
        new_neighbors_(qg_.num_vertices()),
        pruned_neighbors_(qg_.num_vertices()),
        visited_list_(num_threads_,
                      HashBasedBooleanSet(std::min(ef_build_ * ef_build_, num_nodes_ / 10))),
        degrees_(qg_.num_vertices(), degree_bound_) {}

  /**
   * @brief Builds a disk-based quantized graph index from a Vamana graph and vector data.
   *
   * This function constructs the final quantized graph index by combining the graph topology
   * from a pre-built Vamana index with vector data, computing RaBitQ quantization codes
   * for efficient approximate distance computation during search.
   *
   * ============================================================================
   * MEMORY-EFFICIENT OUT-OF-CORE INDEX CONSTRUCTION
   * ============================================================================
   *
   * This implementation is specifically designed to build indices for datasets that are
   * MUCH LARGER than available RAM. The key insight is that we never need to hold the
   * entire dataset in memory - we only need to process one node at a time.
   *
   * How It Works:
   * 1. PHASE 1 - Prepare Aligned Vector File:
   *    - Read vectors sequentially from input file
   *    - Write to temporary file with sector-aligned padding (4KB alignment)
   *    - This enables O_DIRECT reads without page cache overhead
   *    - Memory usage: O(1) - just one vector buffer
   *
   * 2. PHASE 2 - Parallel Node Processing:
   *    - Each thread acquires a small scratch buffer from a shared pool
   *    - For each node to process:
   *      a) Read the node's vector and all neighbor vectors from disk (async I/O)
   *      b) Compute RaBitQ quantization codes in the scratch buffer
   *      c) Write the completed node directly to the output index file
   *      d) Return scratch buffer to pool for reuse
   *    - Writes are sequential per-node, enabling efficient disk I/O
   *
   * 3. PHASE 3 - Build Cache File:
   *    - Read high in-degree nodes from the completed index
   *    - Write to separate cache file for fast loading at search time
   *
   * @param vamana_file  Path to the pre-built Vamana graph file containing graph topology
   * @param filename     Base path for output files. The function generates:
   *                     - {filename}_R{degree}_MD{dim}.index: Main index file
   *                     - {filename}_R{degree}_MD{dim}.index_rotator: FHT rotation matrix
   *                     - {filename}_R{degree}_MD{dim}.index_cache_ids: Cached node IDs
   *                     - {filename}_R{degree}_MD{dim}.index_cache_nodes: Cached node data
   *
   * @note The input vector file must be at path: {filename}_pca_base.fbin
   * @note A temporary file {filename}_tmp.fbin is created and deleted after processing
   */
  void build(const char *vamana_file, const char *filename) {
    // ==================== PHASE 0: Load Graph Topology ====================
    // Load the Vamana graph structure (neighbor lists) into memory.
    // This is the only large in-memory data structure: O(N * degree) bytes.
    // The actual vectors are NOT loaded here - they remain on disk.
    init_from_vamana(std::string(vamana_file));

    // ==================== Setup Output Paths ====================
    std::string index_path = qg_.gen_index_path(filename);
    std::string data_path = std::string(filename) + "_pca_base.fbin";
    std::string tmp_path = std::string(filename) + "_tmp.fbin";

    std::ofstream output(index_path, std::ios::binary);
    assert(output.is_open());

    // ==================== Write Index Header (Metadata Sector) ====================
    // The first sector (4KB) stores index metadata for validation during loading.
    std::vector<uint64_t> metas(kSectorLen / sizeof(uint64_t), 0);
    metas[0] = qg_.num_points_;     // Total number of vectors
    metas[1] = qg_.dimension_;      // Vector dimension
    metas[2] = qg_.entry_point_;    // Graph entry point for search
    metas[3] = qg_.node_len_;       // Bytes per node (vector + codes + factors + neighbors)
    metas[4] = qg_.node_per_page_;  // Nodes packed per disk page

    // Calculate total file size for integrity verification
    size_t page_num = (qg_.num_points_ + qg_.node_per_page_ - 1) / qg_.node_per_page_;
    metas[8] = qg_.page_size_ * page_num + kSectorLen;

    output.write(reinterpret_cast<const char *>(metas.data()), kSectorLen);

    // ==================== PHASE 1: Create Sector-Aligned Temporary Vector File
    // ==================== Why sector alignment? Linux O_DIRECT requires 4KB-aligned reads.
    std::ofstream tmp_output(tmp_path, std::ios::binary);
    assert(tmp_output.is_open());

    std::ifstream vector_input(data_path, std::ios::binary);
    int n, d;
    vector_input.read(reinterpret_cast<char *>(&n), sizeof(int));
    vector_input.read(reinterpret_cast<char *>(&d), sizeof(int));
    assert(d == (qg_.dimension_ + qg_.residual_dimension_));

    // Round up vector size to sector boundary for O_DIRECT compatibility
    size_t vector_tmp_page_size = (d * sizeof(float) + kSectorLen - 1) / kSectorLen * kSectorLen;
    std::vector<char> buffer(vector_tmp_page_size);  // O(1) memory: single vector buffer

    std::cout << "copy vectors..." << std::endl;

    // Stream vectors one at a time: O(1) memory regardless of dataset size
    for (size_t i = 0; i < qg_.num_points_; i++) {
      // Progress bar for user feedback
      if (i % 10000 == 0) {
        float progress = static_cast<float>(i) * 100 / qg_.num_points_;
        int bar_width = 50;
        int pos = static_cast<int>(bar_width * progress / 100);
        std::cout << "\r[";
        for (int j = 0; j < bar_width; ++j) {
          if (j < pos)
            std::cout << "=";
          else if (j == pos)
            std::cout << ">";
          else
            std::cout << " ";
        }
        std::cout << "] " << std::fixed << std::setprecision(1) << progress << "%" << std::flush;
      }
      // Zero-fill buffer to ensure clean padding bytes
      std::memset(buffer.data(), 0, buffer.size());
      vector_input.read(reinterpret_cast<char *>(buffer.data()), d * sizeof(float));
      tmp_output.write(buffer.data(), static_cast<std::streamsize>(vector_tmp_page_size));
    }

    vector_input.close();
    tmp_output.close();

    // ==================== PHASE 2: Parallel Out-of-Core Index Construction ====================
    // Open the aligned vector file for direct I/O (bypasses page cache)
    LinuxAlignedFileReader vector_reader;
    vector_reader.open(tmp_path.c_str());

    // Create a bounded pool of thread-local scratch buffers.
    ConcurrentQueue<ThreadData> thread_data;
#pragma omp parallel for num_threads(static_cast<int>(num_threads_))
    for (size_t thread = 0; thread < num_threads_; thread++) {
#pragma omp critical
      {
        vector_reader.register_thread();
        ThreadData data;
        data.ctx_ = vector_reader.get_ctx();
        // Scratch for building one node's quantized representation
        data.cur_page_scratch_ =
            reinterpret_cast<char *>(memory::align_allocate<kSectorLen>(qg_.page_size_));
        // Scratch for reading neighbor vectors (degree * vector_size)
        data.neighbor_vector_scratch_ = reinterpret_cast<char *>(
            memory::align_allocate<kSectorLen>(qg_.degree_bound_ * vector_tmp_page_size));
        thread_data.push(data);
      }
    }

    std::cout << "\nupdate qg..." << std::endl;

    // Process all nodes in parallel with dynamic scheduling.
    // Each iteration processes ONE node: reads its neighbors, computes quantization, writes to
    // disk.
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < qg_.num_points_; ++i) {
      // Progress bar
      if (i % 10000 == 0) {
        float progress = static_cast<float>(i) * 100 / qg_.num_points_;
        int bar_width = 50;
        int pos = static_cast<int>(bar_width * progress / 100);
        std::cout << "\r[";
        for (int j = 0; j < bar_width; ++j) {
          if (j < pos)
            std::cout << "=";
          else if (j == pos)
            std::cout << ">";
          else
            std::cout << " ";
        }
        std::cout << "] " << std::fixed << std::setprecision(1) << progress << "%" << std::flush;
      }

      // Acquire a scratch buffer from the shared pool (blocking if none available)
      // This ensures we never exceed our memory budget
      ThreadData data = thread_data.pop();
      while (data.cur_page_scratch_ == nullptr) {
        thread_data.wait_for_push_notify();
        data = thread_data.pop();
      }

      // Core processing: read vectors from disk, compute RaBitQ codes, prepare node data
      // This function reads the node's vector and all neighbor vectors via async I/O,
      // then computes quantization codes without holding vectors in memory permanently
      qg_.update_qg_out_of_memory(i, new_neighbors_[i], vector_reader, data);

      // Write completed node directly to output file (sequential I/O pattern)
#pragma omp critical
      {
        // TODO(port-laser-disk-index followup): write/read page-layout mismatch; see proposal D8
        output.seekp(i * qg_.page_size_ + kSectorLen, std::ios::beg);
        output.write(reinterpret_cast<const char *>(data.cur_page_scratch_), qg_.page_size_);
      }

      // Return scratch buffer to pool for reuse by other threads
      thread_data.push(data);
      thread_data.push_notify_all();
    }
    output.close();

    // ==================== Save Rotation Matrix ====================
    // The Fast Hadamard Transform rotation matrix is small and saved separately
    std::string rotator_path = index_path + "_rotator";
    std::ofstream rotator_output(rotator_path, std::ios::binary);
    assert(rotator_output.is_open());
    qg_.rotator_.save(rotator_output);
    rotator_output.close();

    // ==================== Cleanup Temporary File ====================
    // The sector-aligned temporary file is no longer needed
    if (std::remove(tmp_path.c_str()) != 0) {
      std::perror("Error deleting temporary file");
    } else {
      std::cout << "\nTemporary file deleted\n";
    }

    // ==================== PHASE 3: Build Cache File ====================
    // Pre-extract high in-degree nodes to a separate file for fast loading at search time.
    // During search, these nodes are loaded into memory to reduce disk I/O.
    // Nodes are sorted by in-degree in init_from_vamana(), so we cache the most frequently
    // accessed nodes (those with highest in-degree are visited most often during search).
    std::cout << "write cache...\n";
    auto cache_num = static_cast<size_t>(static_cast<double>(qg_.num_vertices()) * kCacheRatio);
    qg_.cache_ids_.resize(cache_num);

    std::string cache_ids_file = index_path + "_cache_ids";
    std::string cache_nodes_file = index_path + "_cache_nodes";

    // Write cache node IDs
    std::ofstream cache_ids_output(cache_ids_file, std::ios::binary);
    cache_ids_output.write(reinterpret_cast<const char *>(&cache_num), sizeof(size_t));
    cache_ids_output.write(reinterpret_cast<const char *>(qg_.cache_ids_.data()),
                           static_cast<std::streamsize>(sizeof(PID) * cache_num));
    cache_ids_output.close();

    // Write cache node data (extracted from the completed index)
    std::ofstream cache_vectors_output(cache_nodes_file, std::ios::binary);
    assert(cache_vectors_output.is_open());
    cache_vectors_output.write(reinterpret_cast<const char *>(&cache_num), sizeof(size_t));
    cache_vectors_output.write(reinterpret_cast<const char *>(&qg_.node_len_), sizeof(size_t));

    // Read cache nodes in batches to balance I/O efficiency and memory usage
    size_t batch_size = 1024;
    char *cache_buffer =
        reinterpret_cast<char *>(memory::align_allocate<kSectorLen>(batch_size * qg_.page_size_));
    LinuxAlignedFileReader cache_reader;
    cache_reader.open(index_path);
    cache_reader.register_thread();
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(batch_size + 1);

    // Batch read and write cache nodes
    for (size_t i = 0; i < cache_num; i += batch_size) {
      size_t cur_batch_size = std::min(cache_num - i, batch_size);
      // Build batch read requests for this chunk
      for (size_t j = 0; j < cur_batch_size; ++j) {
        frontier_read_reqs.emplace_back(kSectorLen + (qg_.cache_ids_[i + j] * qg_.page_size_),
                                        qg_.page_size_,
                                        qg_.cache_ids_[i + j],
                                        cache_buffer + (j * qg_.page_size_));
      }
      // Execute batch read
      cache_reader.read(frontier_read_reqs, cache_reader.get_ctx());
      frontier_read_reqs.clear();

      // Write each node's data (only node_len_ bytes, not full page)
      for (size_t j = 0; j < cur_batch_size; ++j) {
        cache_vectors_output.write(reinterpret_cast<const char *>(cache_buffer +
                                                                  (j * qg_.page_size_)),
                                   static_cast<std::streamsize>(sizeof(char) * qg_.node_len_));
      }
    }
    cache_vectors_output.close();
    cache_reader.close();
    std::cout << "Done. \n";
  }
};

inline void QGBuilder::init_from_vamana(const std::string &filename) {
  size_t expected_file_size;
  size_t file_frozen_pts;
  uint32_t start;
  int64_t file_offset = 0;  // will need this for single file format support
  uint32_t max_observed_degree = 0;
  uint32_t max_range_of_graph = 0;

  std::ifstream in;
  in.exceptions(std::ios::badbit | std::ios::failbit);
  in.open(filename, std::ios::binary);
  in.seekg(file_offset, std::ios::beg);
  in.read(reinterpret_cast<char *>(&expected_file_size), sizeof(size_t));
  in.read(reinterpret_cast<char *>(&max_observed_degree), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&start), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&file_frozen_pts), sizeof(size_t));
  size_t vamana_metadata_size =
      sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);

  assert(max_observed_degree == qg_.degree_bound());

  qg_.set_ep(start);

  std::cout << "From graph header, expected_file_size: " << expected_file_size
            << ", _max_observed_degree: " << max_observed_degree << ", _start: " << start
            << ", file_frozen_pts: " << file_frozen_pts << std::endl;

  std::cout << "Loading vamana graph " << filename << "..." << std::flush;

  std::vector<uint32_t> in_degrees(num_nodes_, 0);

  size_t bytes_read = vamana_metadata_size;
  size_t cc = 0;
  uint32_t nodes_read = 0;
  while (bytes_read != expected_file_size) {
    uint32_t k;
    in.read(reinterpret_cast<char *>(&k), sizeof(uint32_t));

    if (k == 0) {
      std::cerr << "ERROR: Point found with no out-neighbours, point#" << nodes_read << std::endl;
    }

    cc += k;
    ++nodes_read;
    std::vector<uint32_t> tmp(k);
    tmp.reserve(k);
    in.read(reinterpret_cast<char *>(tmp.data()),
            static_cast<std::streamsize>(k * sizeof(uint32_t)));

    for (PID cur_neigh : tmp) {
      in_degrees[cur_neigh]++;
      new_neighbors_[nodes_read - 1].emplace_back(cur_neigh,
                                                  0.0);  // Distance will be computed later in
                                                         // parallel during graph construction
    }

    bytes_read += sizeof(uint32_t) * (static_cast<uint32_t>(k) + 1);
    if (nodes_read % 10000000 == 0) {
      std::cout << "." << std::flush;
    }
    if (k > max_range_of_graph) {
      max_range_of_graph = k;
    }
  }

  qg_.cache_ids_.resize(num_nodes_);
  std::iota(qg_.cache_ids_.begin(), qg_.cache_ids_.end(), 0);
  // Sort node IDs by in-degree in descending order (high in-degree nodes are cached first for
  // better search performance)
  std::sort(qg_.cache_ids_.begin(), qg_.cache_ids_.end(), [&](PID a, PID b) {
    return in_degrees[a] > in_degrees[b];
  });

  std::cout << "done. Index has " << nodes_read << " nodes and " << cc
            << " out-edges, _start is set to " << start << std::endl;
  assert(nodes_read == num_nodes_);
}

}  // namespace alaya::laser
