// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <libaio.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "index/graph/laser/common.hpp"
#include "index/graph/laser/qg/qg_query.hpp"
#include "index/graph/laser/qg/qg_scanner.hpp"
#include "index/graph/laser/quantization/rabitq.hpp"
#include "index/graph/laser/space/l2.hpp"
#include "index/graph/laser/utils/aligned_file_reader.hpp"
#include "index/graph/laser/utils/array.hpp"
#include "index/graph/laser/utils/buffer.hpp"
#include "index/graph/laser/utils/concurrent_queue.hpp"
#include "index/graph/laser/utils/io.hpp"
#include "index/graph/laser/utils/memory.hpp"
#include "index/graph/laser/utils/pca_transform.hpp"
#include "index/graph/laser/utils/rotator.hpp"
#include "third_party/ngt/hashset.hpp"

namespace alaya::laser {
/**
 * @brief this Factor only for illustration, the true storage is continuous
 * degree_bound_*triple_x + degree_bound_*factor_dq + degree_bound_*factor_vq
 *
 * Note: ||x_r||^2 (residual dimension norm) is pre-added to triple_x during build
 */
struct Factor {
  float triple_x;   // Sqr of distance to centroid + 2 * x * x1 / x0 + ||x_r||^2 (residual dim)
  float factor_dq;  // Factor of delta * ||q_r|| * (FastScanRes - sum_q)
  float factor_vq;  // Factor of v_l * ||q_r||
};

struct ThreadData {
  HashBasedBooleanSet visited_;
  buffer::SearchBuffer search_pool_;
  io_context_t ctx_;
  char *sector_scratch_ = nullptr;
  char *neighbor_vector_scratch_ = nullptr;
  char *cur_page_scratch_ = nullptr;
  float *pca_query_scratch_ = nullptr;  // Buffer for PCA-transformed query
  std::shared_ptr<std::vector<float>> pca_query_scratch_storage_;
};

struct ClusterStats {
  float avg_dist;
  float std_dev;
  float z_min;
  float z_max;
};

constexpr size_t kSectorLen = 4096;

class QuantizedGraph {
  friend class QGBuilder;

 private:
  size_t num_points_ = 0;    // num points
  size_t degree_bound_ = 0;  // degree bound
  size_t dimension_ = 0;     // dimension
  size_t residual_dimension_ = 0;
  size_t padded_dim_ = 0;  // padded dimension
  PID entry_point_ = 0;    // Entry point of graph

  data::Array<float,
              std::vector<size_t>,
              memory::AlignedAllocator<float,
                                       1 << 22,
                                       true>>
      data_;  // vectors + graph + quantization codes
  QGScanner scanner_;
  FHTRotator rotator_;
  PCATransform pca_transform_;  // PCA transform for online query transformation
  LinuxAlignedFileReader aligned_file_reader_;
  ConcurrentQueue<ThreadData> thread_data_;
  int dc_count_;
  size_t ef_search_ = 200;

  size_t node_len_;
  size_t page_size_;
  size_t node_per_page_;

  // workspace for disk-based quantized graph
  size_t min_beam_width_ = 2;
  size_t max_beam_width_ = 16;
  std::string index_file_name_;

  std::vector<PID> medoids_;
  std::vector<float> medoids_vector_;

  std::vector<ClusterStats> cluster_stats_;

  std::vector<PID> cache_ids_;
  std::vector<char> cache_nodes_;
  std::unordered_map<PID, char *> caches_;

  int query_time_ = 0;
  float total_io_time_ = 0;
  float total_io_time1_ = 0;
  int64_t total_cpu_time_ = 0;
  float total_read_num_ = 0;
  float total_iter_num_ = 0;
  float total_cache_num_ = 0;
  int64_t total_ks_time_ = 0;
  float total_query_latency_ = 0;
  float total_n_hops_ = 0;

  std::vector<PID> mem_graph_enter_points_;

  size_t nthreads_ = 1;

  /*
   * Position of different data in each row
   *      RawData + QuantizationCodes + Factors + neighborIDs
   * Since we guarantee the degree for each vertex equals degree_bound (multiple of 32),
   * we do not need to store the degree for each vertex
   */
  size_t res_dim_offset_ = 0;
  size_t code_offset_ = 0;      // pos of packed code
  size_t factor_offset_ = 0;    // pos of Factor
  size_t neighbor_offset_ = 0;  // pos of Neighbors
  size_t row_offset_ = 0;       // length of entire row

  void initialize();
  void allocate_data();
  void init_workspace();

  // search on disk-based quantized graph
  void disk_search_qg(const float *__restrict__ query,
                      uint32_t knn,
                      uint32_t *__restrict__ results);

  void copy_vectors(const float *);

  // for beam search
  [[nodiscard]] uint64_t get_page_offset(uint64_t node_id) const {
    return kSectorLen + page_size_ * (node_id / node_per_page_);
  }

  [[nodiscard]] uint64_t offset_to_node(uint64_t node_id) const {
    return (node_id % node_per_page_) * node_len_;
  }

  [[nodiscard]] std::string gen_index_path(const char *prefix) const {
    return std::string(prefix) + "_R" + std::to_string(degree_bound_) + "_MD" +
           std::to_string(dimension_) + ".index";
  }

  void update_qg_out_of_memory(PID,
                               const std::vector<Candidate<float>> &,
                               LinuxAlignedFileReader &,
                               ThreadData);

  float scan_neighbors(const QGQuery &q_obj,
                       const float *cur_data,
                       float *appro_dist,
                       buffer::SearchBuffer &search_pool,
                       uint32_t cur_degree,
                       const HashBasedBooleanSet &visited) const;

 public:
  explicit QuantizedGraph(size_t num,
                          size_t max_deg,
                          size_t main_dim,
                          size_t dim,
                          uint64_t rotator_seed = 0,
                          std::string rotator_dump_path = "");

  ~QuantizedGraph();

  [[nodiscard]] auto num_vertices() const { return this->num_points_; }

  [[nodiscard]] auto dimension() const { return this->dimension_; }

  [[nodiscard]] auto residual_dimension() const { return this->residual_dimension_; }

  [[nodiscard]] auto degree_bound() const { return this->degree_bound_; }

  [[nodiscard]] auto entry_point() const { return this->entry_point_; }

  void set_ep(PID entry) { this->entry_point_ = entry; }

  void load_disk_index(const char *, float);

  void set_params(size_t ef_search, size_t num_threads, int beam_width);

  void load_medoids(const char *);

  void load_cache(std::string &cache_ids_file,
                  std::string &cache_nodes_file,
                  size_t online_cache_num);

  void load_cluster_stats(const char *filename);

  /* search and copy results to KNN */
  void search(const float *__restrict__ query, uint32_t knn, uint32_t *__restrict__ results);

  void batch_search(const float *__restrict__ query,
                    uint32_t knn,
                    uint32_t *__restrict__ results,
                    size_t num_queries);

  void destroy_thread_data() {
    while (thread_data_.size() > 0) {
      ThreadData data = thread_data_.pop();
      while (data.sector_scratch_ == nullptr) {
        thread_data_.wait_for_push_notify();
        data = this->thread_data_.pop();
      }
      if (data.sector_scratch_ != nullptr) {
        free(data.sector_scratch_);
      }
      if (data.pca_query_scratch_ != nullptr) {
        data.pca_query_scratch_storage_.reset();
        data.pca_query_scratch_ = nullptr;
      }
    }
    aligned_file_reader_.deregister_all_threads();
    aligned_file_reader_.close();
  }
};

inline QuantizedGraph::QuantizedGraph(size_t num,
                                      size_t max_deg,
                                      size_t main_dim,
                                      size_t dim,
                                      uint64_t rotator_seed,
                                      std::string rotator_dump_path)
    : num_points_(num),
      degree_bound_(max_deg),
      dimension_(main_dim),
      residual_dimension_(dim - dimension_)  // Residual dimension for extended vector
                                             // representation (e.g., for GIST dataset)
      ,
      padded_dim_(1 << ceil_log2(dimension_)),
      scanner_(padded_dim_, degree_bound_),
      rotator_(dimension_, rotator_seed),
      node_len_((32 * dimension_ + 32 * residual_dimension_ + 128 * degree_bound_ +
                 degree_bound_ * padded_dim_) /
                8) {
  // Dump the rotator's sign-scaled mat_ vector BEFORE any consumer
  // (RaBitQ training, search path) reads from it. When `rotator_dump_path`
  // is empty, no file is written and the on-disk dsqg.index is unchanged.
  if (!rotator_dump_path.empty()) {
    rotator_.dump_signs(rotator_dump_path);
  }

  node_per_page_ = std::max(1, static_cast<int>(kSectorLen / node_len_));
  page_size_ = (node_per_page_ * node_len_ + kSectorLen - 1) / kSectorLen * kSectorLen;

  if (dimension_ != padded_dim_) {  // Currently only supports dimension that is a power of 2
                                    // (dimension_ must equal padded_dim_)
    throw std::invalid_argument("QuantizedGraph: dimension must be a power of two");
  }
  // See docs/LASER.md (Known Issues): the on-disk write path in qg_builder
  // uses `i * page_size_ + kSectorLen` while the read path here uses
  // `page_size_ * (id/npp) + (id%npp) * node_len_`. They only agree when
  // node_per_page_ == 1, otherwise recall collapses (SIFT-1M → ~0.08%).
  // Fail at construction until the follow-up fix (proposal D8) lands.
  if (node_per_page_ > 1) {
    throw std::invalid_argument(
        "QuantizedGraph: node_per_page_ > 1 is not supported by the current port "
        "(write/read page layout mismatch; see docs/LASER.md Known Issues). "
        "Use a configuration with node_len_ >= kSectorLen (typically main_dim=256 "
        "and raw dim >= 768) until the follow-up fix lands.");
  }
  std::cout << "main_dim: " << main_dim << ", dim: " << dim << ", dimension_: " << dimension_
            << ", residual_dimension_: " << residual_dimension_ << std::endl;
  initialize();
}

inline QuantizedGraph::~QuantizedGraph() { destroy_thread_data(); }

inline void QuantizedGraph::set_params(size_t ef_search, size_t num_threads, int beam_width) {
  this->nthreads_ = num_threads;
  this->max_beam_width_ = beam_width;
  this->ef_search_ = ef_search;
  destroy_thread_data();
  if (index_file_name_ == "") {
    throw std::logic_error("QuantizedGraph::set_params: call load_disk_index() first");
  }
  aligned_file_reader_.open(index_file_name_);

#pragma omp parallel for num_threads(static_cast<int>(nthreads_))
  for (size_t thread = 0; thread < nthreads_; thread++) {
#pragma omp critical
    {
      this->aligned_file_reader_.register_thread();
      ThreadData data;
      data.ctx_ = aligned_file_reader_.get_ctx();
      data.search_pool_.resize(ef_search_);
      data.visited_ =
          HashBasedBooleanSet(std::min(this->num_points_ / 10, ef_search_ * ef_search_));
      data.sector_scratch_ = reinterpret_cast<char *>(
          memory::align_allocate<kSectorLen>(2 * max_beam_width_ * page_size_));
      data.pca_query_scratch_storage_ =
          std::make_shared<std::vector<float>>(dimension_ + residual_dimension_);
      data.pca_query_scratch_ =
          data.pca_query_scratch_storage_->data();  // Allocate PCA query buffer
      this->thread_data_.push(data);
    }
  }
}

/*
 * search single query
 */
inline void QuantizedGraph::search(const float *__restrict__ query,
                                   uint32_t knn,
                                   uint32_t *__restrict__ results) {
  disk_search_qg(query, knn, results);
}

inline void QuantizedGraph::batch_search(const float *__restrict__ query,
                                         uint32_t knn,
                                         uint32_t *__restrict__ results,
                                         size_t num_queries) {
#pragma omp parallel for schedule(dynamic) num_threads(static_cast<int>(nthreads_))
  for (size_t i = 0; i < num_queries; ++i) {
    disk_search_qg(query + i * (dimension_ + residual_dimension_), knn, results + i * knn);
  }
}

/**
 * @brief Performs k-nearest neighbor search on a disk-based quantized graph using asynchronous I/O.
 *
 * This function implements a beam search algorithm optimized for disk-resident graph indices.
 * It uses Linux AIO (Asynchronous I/O) to overlap disk reads with computation, achieving
 * high throughput by processing multiple nodes in parallel.
 *
 * Algorithm Overview:
 * 1. Initialize search with entry point(s) - either medoids (cluster centers) or global entry point
 * 2. Iteratively expand the search frontier using beam search with adaptive beam width
 * 3. For each candidate node:
 *    - If cached in memory: process immediately without disk I/O
 *    - If on disk: submit async read request and process when data arrives
 * 4. Use RaBitQ (Randomized Bit Quantization) for fast approximate distance computation
 * 5. Maintain a result pool to track the k nearest neighbors found so far
 *
 * Key Optimizations:
 * - Asynchronous I/O: Overlaps disk reads with CPU computation
 * - Adaptive beam width: Starts small and grows exponentially up to max_beam_width_
 * - In-memory caching: Frequently accessed nodes are cached to avoid repeated disk reads
 * - Pipelined processing: Processes nodes from previous iteration while waiting for new I/O
 *
 * @param query     Pointer to the query vector (unrotated). The vector should have
 *                  (dimension_ + residual_dimension_) float elements. The first dimension_
 *                  elements are the main vector, and the remaining are residual components.
 * @param knn       Number of nearest neighbors to retrieve.
 * @param results   Output array to store the IDs of k nearest neighbors. Must have
 *                  space for at least knn uint32_t elements.
 *
 * @note This function is thread-safe. Each thread acquires its own ThreadData from a
 *       concurrent queue, which includes scratch buffers and AIO context.
 * @note The query vector is internally rotated using Fast Hadamard Transform for
 *       compatibility with the RaBitQ quantization scheme.
 */
inline void QuantizedGraph::disk_search_qg(const float *__restrict__ query,
                                           uint32_t knn,
                                           uint32_t *__restrict__ results) {
  // ==================== Thread-local Data Acquisition ====================
  // Acquire thread-local workspace from the concurrent queue.
  // This includes: visited set, search buffer, AIO context, and scratch memory.
  ThreadData data = thread_data_.pop();
  while (data.sector_scratch_ == nullptr) {
    this->thread_data_.wait_for_push_notify();
    data = thread_data_.pop();
  }
  data.visited_.clear();
  data.search_pool_.clear();

  // ==================== PCA Transform ====================
  // Transform the original query using PCA for dimension reordering.
  // After transformation, high-variance dimensions are placed first.
  const float *transformed_query = query;
  if (pca_transform_.is_loaded()) {
    pca_transform_.transform(query, data.pca_query_scratch_);
    transformed_query = data.pca_query_scratch_;
  }

  // Performance timing variables (for profiling purposes)
  auto query_start = std::chrono::high_resolution_clock::now();
  int64_t submit_time = 0;
  int64_t wait_time = 0;
  int64_t process_time = 0;
  float n_hops = 0;  // Number of I/O rounds (disk access iterations)

  // ==================== Query Preparation ====================
  // Create query object and apply Fast Hadamard Transform rotation.
  // This rotation aligns the query with the quantized representation used in RaBitQ.
  QGQuery q_obj(transformed_query, padded_dim_);
  q_obj.query_prepare(rotator_, scanner_);

  // Pointer to residual query components (used for datasets like GIST with extended dimensions)
  const float *residual_query = transformed_query + dimension_;

  // Compute ||q_r||^2 for residual dimensions to improve approximate distance precision
  float sqr_qr = 0;
  for (size_t i = 0; i < residual_dimension_; ++i) {
    sqr_qr += residual_query[i] * residual_query[i];
  }
  q_obj.set_sqr_qr(sqr_qr);

  // ==================== Search Pool Initialization ====================
  // Initialize the search frontier with starting points.
  // If medoids (cluster centers) are available, find the closest one to the query
  // and use it as an additional entry point for better search quality.
  if (!medoids_.empty()) {
    PID best_medoid = 0;
    float best_dist = FLT_MAX;
    size_t best_medoid_idx = 0;
    // Linear scan through medoids to find the closest one
    for (size_t cur_m = 0; cur_m < medoids_.size(); cur_m++) {
      float cur_expanded_dist =
          space::l2_sqr(transformed_query,
                        medoids_vector_.data() + (dimension_ + residual_dimension_) * cur_m,
                        dimension_);
      if (cur_expanded_dist < best_dist) {
        best_medoid = medoids_[cur_m];
        best_dist = cur_expanded_dist;
        best_medoid_idx = cur_m;
      }
    }
    // Insert best medoid with max distance (distance will be computed when visited)
    data.search_pool_.insert(best_medoid, FLT_MAX);
  }
  // Always include the global entry point as a starting position
  data.search_pool_.insert(entry_point_, FLT_MAX);

  // ==================== Result and Distance Buffers ====================
  // Result pool maintains the top-k nearest neighbors found during search
  buffer::ResultBuffer res_pool(knn);

  // Buffer for storing approximate distances computed via RaBitQ fast scan.
  // RaBitQ computes distances to all neighbors of a node in a single SIMD-optimized pass.
  std::vector<float> appro_dist(degree_bound_);

  // ==================== Asynchronous I/O Data Structures ====================
  // frontier_read_reqs: Batch of aligned read requests to submit to AIO
  std::vector<AlignedRead> frontier_read_reqs;
  frontier_read_reqs.reserve(2 * max_beam_width_);

  // prepared_nodes: Nodes whose data has been fetched and is ready for processing
  std::deque<std::pair<PID, char *>> prepared_nodes;

  // ongoing_nodes: Maps node IDs to their buffer locations for in-flight I/O requests
  std::unordered_map<PID, char *> ongoing_nodes;

  // free_slots: Pool of available memory buffers for disk reads (double-buffering scheme)
  std::deque<char *> free_slots;
  for (size_t i = 0; i < 2 * max_beam_width_; i++) {
    free_slots.push_back(data.sector_scratch_ + i * page_size_);
  }

  // AIO event buffer for collecting completed I/O operations
  std::vector<io_event> evts(max_beam_width_);

  // cache_nhoods: Nodes found in memory cache (no disk I/O needed)
  std::vector<PID> cache_nhoods;

  // Adaptive beam width: starts at 1 and grows exponentially
  size_t cur_beam_size = 1;

  // ==================== Node Processing Lambda ====================
  // This lambda processes a single node: computes exact distance to query,
  // scans neighbors using RaBitQ approximation, and updates search/result pools.
  auto process_node = [&](PID cur_node, float *cur_data) {
    // Scan neighbors and compute approximate distances using RaBitQ.
    // Also computes exact L2 distance from query to current node.
    float sqr_y = scan_neighbors(q_obj,
                                 cur_data,
                                 appro_dist.data(),
                                 data.search_pool_,
                                 this->degree_bound_,
                                 data.visited_);
    // Add residual dimension distance if applicable (e.g., for GIST dataset)
    if (residual_dimension_ > 0) {
      float *residual_data = cur_data + dimension_;
      sqr_y += space::l2_sqr(reinterpret_cast<const float *>(residual_data),
                             residual_query,
                             residual_dimension_);
    }
    // Insert current node with exact distance into result pool
    res_pool.insert(cur_node, sqr_y);
  };

  // ==================== I/O Completion Handler Lambda ====================
  // Collects completed AIO events and moves nodes from ongoing to prepared queue.
  // Uses non-blocking io_getevents to check for completed I/O without waiting.
  auto wait_for_nodes = [&]() {
    // Non-blocking check: minimum events = 0, maximum = cur_beam_size
    int ret = io_getevents(data.ctx_, 0, static_cast<int64_t>(cur_beam_size), evts.data(), nullptr);

    // Process each completed I/O event
    for (unsigned int i = 0; i < ret; i++) {
      int id = static_cast<int>(reinterpret_cast<uintptr_t>(evts[i].data));
      if (ongoing_nodes.find(id) == ongoing_nodes.end()) {
        throw std::runtime_error("QuantizedGraph::search: AIO completion id not in ongoing_nodes");
      }
      // Move from ongoing to prepared queue
      prepared_nodes.emplace_back(id, ongoing_nodes[id]);
      ongoing_nodes.erase(id);
    }
  };

  // Track remaining nodes from previous iteration for pipelined processing
  size_t previous_remain_num = 0;

  // Total I/O operations counter (for profiling)
  int64_t io_num = 0;

  // ==================== Main Search Loop ====================
  // Iteratively expand search frontier until no more candidates remain.
  // Uses beam search with adaptive width and pipelined I/O processing.
  while (data.search_pool_.has_next()) {
    frontier_read_reqs.clear();
    cache_nhoods.clear();
    size_t n_ops = 0;
    size_t need_process_num = 0;
    size_t remain_num = 0;

    // Adaptive beam width: double the beam size each iteration (up to max)
    // This helps balance between exploration breadth and I/O efficiency.
    cur_beam_size = std::min(max_beam_width_,
                             static_cast<size_t>(std::ceil(2 * static_cast<float>(cur_beam_size))));

    auto wait_start = std::chrono::high_resolution_clock::now();

    // -------------------- Build I/O Request Batch --------------------
    // Pop candidates from search pool and prepare I/O requests for non-cached nodes.
    while (data.search_pool_.has_next() && frontier_read_reqs.size() < cur_beam_size) {
      PID cur_node = data.search_pool_.pop();

      // Skip already visited nodes to avoid redundant processing
      if (data.visited_.get(cur_node)) {
        continue;
      }
      data.visited_.set(cur_node);

      // Check if node is in memory cache
      if (caches_.find(cur_node) != caches_.end()) {
        // Cache hit: add to cache_nhoods for immediate processing
        cache_nhoods.push_back(cur_node);
      } else {
        // Cache miss: need to read from disk
        if (free_slots.empty()) {
          throw std::runtime_error("QuantizedGraph::search: free_buffer pool exhausted");
        }
        // Allocate a buffer slot for this read
        char *slot = free_slots.front();
        assert(slot != nullptr);
        free_slots.pop_front();
        ongoing_nodes[cur_node] = slot;
        // Create aligned read request (page-aligned for direct I/O)
        frontier_read_reqs.emplace_back(get_page_offset(cur_node), page_size_, cur_node, slot);
      }
      total_read_num_++;
    }

    // -------------------- Submit Async I/O Requests --------------------
    // Submit batch of read requests to the AIO subsystem
    if (!frontier_read_reqs.empty()) {
      n_hops++;  // Count as one I/O round
      auto submit_start = std::chrono::high_resolution_clock::now();
      n_ops = aligned_file_reader_.submit_reqs(frontier_read_reqs, data.ctx_);
      io_num += n_ops;
      auto submit_end = std::chrono::high_resolution_clock::now();
      submit_time +=
          std::chrono::duration_cast<std::chrono::microseconds>(submit_end - submit_start).count();
    }

    // -------------------- Process Cached Nodes --------------------
    // Process cached nodes (these are stored in memory and don't require disk I/O)
    for (auto &cache_id : cache_nhoods) {
      auto *cur_data = caches_[cache_id];
      process_node(cache_id, reinterpret_cast<float *>(cur_data));
    }

    // -------------------- Pipelined Processing --------------------
    // Calculate how many nodes to process in this iteration.
    // We process half of new I/O ops plus leftovers from previous iteration,
    // allowing I/O and computation to overlap (pipelining).
    remain_num = 0.5 * n_ops;
    need_process_num = n_ops + previous_remain_num - remain_num;
    previous_remain_num = remain_num;

    // Process nodes that have completed I/O (from previous or current iteration)
    while (need_process_num > 0) {
      if (!prepared_nodes.empty()) {
        // Process a node from the prepared queue
        auto node = prepared_nodes.front();
        prepared_nodes.pop_front();

        // Calculate offset within page and process the node
        process_node(node.first,
                     reinterpret_cast<float *>(node.second + offset_to_node(node.first)));

        need_process_num--;
        // Return buffer to free pool for reuse
        free_slots.push_back(node.second);
      } else {
        // No prepared nodes available; wait for I/O completion
        wait_for_nodes();
      }
    }
  }

  // ==================== Process Remaining Nodes ====================
  // After the main loop exits, there may still be nodes in the pipeline
  // that haven't been processed yet. Drain the remaining nodes.
  while (previous_remain_num > 0) {
    if (!prepared_nodes.empty()) {
      auto node = prepared_nodes.front();
      prepared_nodes.pop_front();
      process_node(node.first, reinterpret_cast<float *>(node.second + offset_to_node(node.first)));
      previous_remain_num--;
      free_slots.push_back(node.second);
    } else {
      // Wait for any remaining in-flight I/O operations to complete
      wait_for_nodes();
    }
  }

  // Record query end time for latency measurement
  auto query_end = std::chrono::high_resolution_clock::now();

  auto latency =
      std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start).count();

  // ==================== Copy Results and Cleanup ====================
  // Copy the k nearest neighbor IDs from result pool to output array
  res_pool.copy_results(results);

  // Return thread-local data to the concurrent queue for reuse by other threads
  thread_data_.push(data);
  thread_data_.push_notify_all();
}

// scan a data row (including data vec and quantization codes for its neighbors)
// return exact distance for current vertex
inline float QuantizedGraph::scan_neighbors(const QGQuery &q_obj,
                                            const float *cur_data,
                                            float *appro_dist,
                                            buffer::SearchBuffer &search_pool,
                                            uint32_t cur_degree,
                                            const HashBasedBooleanSet &visited) const {
  float sqr_y = space::l2_sqr(q_obj.query_data(), cur_data, dimension_);

  /* Compute approximate distance by Fast Scan */
  const auto *packed_code = reinterpret_cast<const uint8_t *>(&cur_data[code_offset_]);
  const auto *factor = &cur_data[factor_offset_];
  this->scanner_.scan_neighbors(appro_dist,
                                q_obj.lut().data(),
                                sqr_y,
                                q_obj.lower_val(),
                                q_obj.width(),
                                q_obj.sqr_qr(),
                                q_obj.sumq(),
                                packed_code,
                                factor);

  const PID *ptr_nb = reinterpret_cast<const PID *>(&cur_data[neighbor_offset_]);
  for (uint32_t i = 0; i < cur_degree; ++i) {
    PID cur_neighbor = ptr_nb[i];
    float tmp_dist = appro_dist[i];
    if (search_pool.is_full(tmp_dist) || visited.get(cur_neighbor)) {
      continue;
    }
    search_pool.insert(cur_neighbor, tmp_dist);
  }

  return sqr_y;
}

inline void QuantizedGraph::initialize() {
  /* check size */
  assert(padded_dim_ % 64 == 0);
  assert(padded_dim_ >= dimension_);

  this->res_dim_offset_ = dimension_;
  this->code_offset_ = dimension_ + residual_dimension_;  // Pos of packed code (aligned)
  this->factor_offset_ = code_offset_ + padded_dim_ / 64 * 2 * degree_bound_;  // Pos of Factor
  this->neighbor_offset_ = factor_offset_ + sizeof(Factor) * degree_bound_ / sizeof(float);
  this->row_offset_ = neighbor_offset_ + degree_bound_;
}

inline void QuantizedGraph::init_workspace() {
  aligned_file_reader_.open(index_file_name_);

#pragma omp parallel for num_threads(static_cast<int>(nthreads_))
  for (size_t thread = 0; thread < nthreads_; thread++) {
#pragma omp critical
    {
      this->aligned_file_reader_.register_thread();
      ThreadData data;
      data.ctx_ = aligned_file_reader_.get_ctx();
      data.search_pool_.resize(ef_search_);
      data.visited_ =
          HashBasedBooleanSet(std::min(this->num_points_ / 10, ef_search_ * ef_search_));
      data.sector_scratch_ = reinterpret_cast<char *>(
          memory::align_allocate<kSectorLen>(2 * max_beam_width_ * page_size_));
      data.pca_query_scratch_storage_ =
          std::make_shared<std::vector<float>>(dimension_ + residual_dimension_);
      data.pca_query_scratch_ =
          data.pca_query_scratch_storage_->data();  // Allocate PCA query buffer
      this->thread_data_.push(data);
    }
  }
}

inline void QuantizedGraph::update_qg_out_of_memory(
    PID cur_id,
    const std::vector<Candidate<float>> &new_neighbors,
    LinuxAlignedFileReader &vector_reader,
    ThreadData thread_data) {
  size_t cur_degree = new_neighbors.size();
  std::memset(thread_data.cur_page_scratch_, 0, page_size_);
  if (cur_degree == 0) {
    return;
  }
  char *vector_buf = thread_data.neighbor_vector_scratch_;
  char *page_buf = thread_data.cur_page_scratch_;

  PID *neighbor_ptr = reinterpret_cast<PID *>(page_buf + neighbor_offset_ * 4);
  for (size_t i = 0; i < cur_degree; ++i) {
    neighbor_ptr[i] = new_neighbors[i].id;
    // std::cout << new_neighbors[i].id << " ";
  }
  size_t full_page_size = ((dimension_ + residual_dimension_) * sizeof(float) + kSectorLen - 1) /
                          kSectorLen * kSectorLen;
  size_t main_page_size = (dimension_ * sizeof(float) + kSectorLen - 1) / kSectorLen * kSectorLen;

  RowMatrix<float> x_pad(cur_degree, padded_dim_);  // padded neighbors mat
  RowMatrix<float> c_pad(1, padded_dim_);           // padded duplicate centroid mat
  x_pad.setZero();
  c_pad.setZero();

  std::vector<AlignedRead> frontier_read_reqs;
  frontier_read_reqs.reserve(cur_degree + 1);

  // Create read requests for each neighbor node's vector data from disk
  for (size_t i = 0; i < cur_degree; ++i) {
    auto neighbor_id = new_neighbors[i].id;
    uint64_t offset = neighbor_id * full_page_size;
    uint64_t len = full_page_size;
    void *buf = reinterpret_cast<void *>(reinterpret_cast<char *>(vector_buf) + i * full_page_size);
    frontier_read_reqs.emplace_back(offset, len, neighbor_id, buf);
  }

  // Create read request for the current node (centroid) vector data
  uint64_t cur_offset = cur_id * full_page_size;
  uint64_t cur_len = full_page_size;
  void *cur_buf = reinterpret_cast<void *>(reinterpret_cast<char *>(page_buf));
  frontier_read_reqs.emplace_back(cur_offset, cur_len, cur_id, cur_buf);

  // Execute batch read operation to fetch all neighbor vectors and centroid in one I/O call
  vector_reader.read(frontier_read_reqs, thread_data.ctx_);

  /* Copy data */
  for (size_t i = 0; i < cur_degree; ++i) {
    auto neighbor_id = new_neighbors[i].id;
    const auto *cur_data =
        reinterpret_cast<const float *>(reinterpret_cast<char *>(vector_buf) + i * full_page_size);
    std::copy(cur_data, cur_data + dimension_, &x_pad(static_cast<int64_t>(i), 0));
  }
  const auto *cur_cent = reinterpret_cast<const float *>(reinterpret_cast<char *>(page_buf));
  std::copy(cur_cent, cur_cent + dimension_, &c_pad(0, 0));

  /* rotate Matrix */
  RowMatrix<float> x_rotated(cur_degree, padded_dim_);
  RowMatrix<float> c_rotated(1, padded_dim_);
  for (int64_t i = 0; i < static_cast<int64_t>(cur_degree); ++i) {
    this->rotator_.rotate(&x_pad(i, 0), &x_rotated(i, 0));
  }
  this->rotator_.rotate(&c_pad(0, 0), &c_rotated(0, 0));

  // Get codes and factors for rabitq
  auto *fac_ptr = reinterpret_cast<float *>(page_buf + 4 * factor_offset_);
  auto *packed_code_ptr = reinterpret_cast<uint8_t *>(page_buf + 4 * code_offset_);
  float *triple_x = fac_ptr;
  float *factor_dq = triple_x + this->degree_bound_;
  float *factor_vq = factor_dq + this->degree_bound_;
  rabitq_codes(x_rotated, c_rotated, packed_code_ptr, triple_x, factor_dq, factor_vq);

  // Add ||x_r||^2 (residual dimensions) directly to triple_x for improved precision
  // This avoids storing a separate sqr_xr array and saves computation during search
  for (size_t i = 0; i < cur_degree; ++i) {
    const auto *neighbor_data =
        reinterpret_cast<const float *>(reinterpret_cast<char *>(vector_buf) + i * full_page_size);
    const float *residual_data = neighbor_data + dimension_;
    float sqr_xr_val = 0;
    for (size_t j = 0; j < residual_dimension_; ++j) {
      sqr_xr_val += residual_data[j] * residual_data[j];
    }
    triple_x[i] += sqr_xr_val;
  }
}

inline void QuantizedGraph::load_disk_index(const char *filename, float search_DRAM_budget) {
  index_file_name_ = gen_index_path(filename);
  if (!std::filesystem::exists(index_file_name_)) {
    throw std::runtime_error("QuantizedGraph::load_disk_index: file not found: " +
                             index_file_name_);
  }
  std::ifstream input(index_file_name_, std::ios::binary);
  assert(input.is_open());

  std::vector<uint64_t> metas(kSectorLen / sizeof(uint64_t), 0);
  input.read(reinterpret_cast<char *>(metas.data()), kSectorLen);

  assert(metas[0] == num_points_);
  assert(metas[1] == dimension_);
  assert(metas[3] == node_len_);
  assert(metas[4] == node_per_page_);
  assert(metas[8] == std::filesystem::file_size(index_file_name_));

  entry_point_ = metas[2];

  std::string rotator_path = std::string(index_file_name_) + "_rotator";
  std::ifstream rotator_input(rotator_path, std::ios::binary);
  assert(rotator_input.is_open());
  rotator_.load(rotator_input);

  init_workspace();

  load_medoids(filename);

  // Load PCA parameters for online query transformation
  std::string pca_path = std::string(filename) + "_pca.bin";
  if (std::filesystem::exists(pca_path)) {
    pca_transform_.load(pca_path);
  } else {
    std::cerr << "Warning: PCA file not found: " << pca_path << std::endl;
  }

  // Calculate available cache space: use only 80% of the DRAM budget for caching.
  // The remaining 20% is reserved for other runtime memory allocations (e.g., search buffers,
  // AIO scratch space, and temporary data structures during query processing).
  auto cache_space = static_cast<size_t>(search_DRAM_budget * 1000 * 1000 * 1000 * 0.8);

  // Determine the number of nodes to cache: take the minimum of:
  // 1. Maximum nodes that fit in the available cache space (cache_space / node_len_)
  // 2. Maximum allowed cache ratio (kCacheRatio * num_points_) to limit memory usage
  size_t online_cache_num =
      std::min(cache_space / node_len_, static_cast<size_t>(kCacheRatio * num_points_));

  std::string cache_ids_file = std::string(index_file_name_) + "_cache_ids";
  std::string cache_nodes_file = std::string(index_file_name_) + "_cache_nodes";
  load_cache(cache_ids_file, cache_nodes_file, online_cache_num);
}

inline void QuantizedGraph::load_medoids(const char *filename) {
  // std::cout << "loading medoids..." << std::endl;
  std::string medoids_indices_file = std::string(filename) + "_medoids_indices";
  std::string medoids_file = std::string(filename) + "_medoids";

  if (!(std::filesystem::exists(medoids_file) && std::filesystem::exists(medoids_indices_file))) {
    return;
  }

  std::ifstream medoid_input(medoids_indices_file, std::ios::binary);
  assert(medoid_input.is_open());
  int medoid_num;
  int tmp;
  medoid_input.read(reinterpret_cast<char *>(&medoid_num), sizeof(int));
  medoid_input.read(reinterpret_cast<char *>(&tmp), sizeof(int));
  medoids_.resize(static_cast<uint64_t>(medoid_num) * static_cast<uint64_t>(tmp));
  medoid_input.read(reinterpret_cast<char *>(medoids_.data()),
                    static_cast<std::streamsize>(sizeof(int) * medoid_num * tmp));
  medoid_input.close();

  std::ifstream mediod_vector_input(medoids_file, std::ios::binary);
  assert(mediod_vector_input.is_open());
  int dim = 0;
  mediod_vector_input.read(reinterpret_cast<char *>(&medoid_num), sizeof(int));
  mediod_vector_input.read(reinterpret_cast<char *>(&dim), sizeof(int));
  if (medoid_num != static_cast<int>(medoids_.size())) {
    throw std::runtime_error(
        "QuantizedGraph::load_medoids: medoid count mismatch between indices and vectors file");
  }
  if (dim != static_cast<int>(dimension_ + residual_dimension_)) {
    throw std::runtime_error(
        "QuantizedGraph::load_medoids: medoid dimension mismatch vs. index dimension");
  }
  medoids_vector_.resize(static_cast<uint64_t>(medoid_num * (dimension_ + residual_dimension_)));
  mediod_vector_input.read(reinterpret_cast<char *>(medoids_vector_.data()),
                           static_cast<std::streamsize>(sizeof(float) * medoid_num *
                                                        (dimension_ + residual_dimension_)));
  mediod_vector_input.close();
}

inline void QuantizedGraph::load_cache(std::string &cache_ids_file,
                                       std::string &cache_nodes_file,
                                       size_t online_cache_num) {
  std::ifstream cache_ids_input(cache_ids_file, std::ios::binary);
  std::ifstream cache_vectors_input(cache_nodes_file, std::ios::binary);
  assert(cache_ids_input.is_open());
  assert(cache_vectors_input.is_open());
  size_t cache_ids_num;
  size_t cache_nodes_num;
  size_t tmp_node_len;
  cache_ids_input.read(reinterpret_cast<char *>(&cache_ids_num), sizeof(size_t));
  cache_vectors_input.read(reinterpret_cast<char *>(&cache_nodes_num), sizeof(size_t));
  cache_vectors_input.read(reinterpret_cast<char *>(&tmp_node_len), sizeof(size_t));
  online_cache_num = std::min(online_cache_num, std::min(cache_ids_num, cache_nodes_num));
  std::cout << "online_cache_num: " << online_cache_num << std::endl;
  cache_ids_.resize(online_cache_num);
  cache_ids_input.read(reinterpret_cast<char *>(cache_ids_.data()),
                       static_cast<std::streamsize>(sizeof(PID) * online_cache_num));
  assert(tmp_node_len == node_len_);
  cache_nodes_.resize(online_cache_num * node_len_);
  cache_vectors_input.read(reinterpret_cast<char *>(cache_nodes_.data()),
                           static_cast<std::streamsize>(sizeof(char) * online_cache_num *
                                                        node_len_));
  for (unsigned i = 0; i < cache_ids_.size(); i++) {
    PID cur_id = cache_ids_[i];
    caches_[cur_id] = cache_nodes_.data() + i * node_len_;
  }
}

}  // namespace alaya::laser
