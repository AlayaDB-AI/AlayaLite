// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file diskann_index.hpp
 * @brief Self-contained DiskANN disk index: build / load / search.
 *
 * `DiskANNIndex` is an autonomous index class (peer of LASER's QuantizedGraph,
 * design D1): it builds a Vamana graph in memory, packs it into a sector-aligned
 * disk layout, optionally trains PQ, and serves cached beam search over the
 * resulting on-disk index. It does not participate in the segment / disk-
 * collection subsystem.
 *
 * On-disk directory:
 *   meta.bin           index metadata (this file's MetaHeader)
 *   diskann.index      sector-aligned graph + vectors (disk_layout.hpp)
 *   ids.bin            internal-id -> external uint64 label map
 *   cache_ids.bin      BFS cache node ids        (node_cache.hpp)
 *   cache_nodes.bin    BFS cache node records
 *   pq_pivots.bin      PQ global centroid + codebook  (PQ builds only)
 *   pq_compressed.bin  PQ codes                       (PQ builds only)
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "index/graph/diskann/beam_search.hpp"
#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/diskann/disk_page_io.hpp"
#include "index/graph/diskann/disk_update_context.hpp"
#include "index/graph/diskann/node_cache.hpp"
#include "index/graph/diskann/pq_table.hpp"
#include "index/graph/diskann/search_scratch.hpp"
#include "index/graph/diskann/slot_allocator.hpp"
#include "index/graph/diskann/tombstone_bitmap.hpp"
#include "index/graph/laser/utils/aligned_file_reader_factory.hpp"
#include "index/graph/laser/utils/concurrent_queue.hpp"
#include "index/graph/vamana/robust_prune.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "simd/distance_l2.hpp"

namespace alaya::diskann {

/// Build-time configuration.
struct DiskANNBuildParams {
  uint32_t R = 64;            ///< graph degree bound
  uint32_t L = 100;           ///< Vamana build beam width
  float alpha = 1.2f;         ///< Vamana α-RNG pruning
  uint32_t pq_n_chunks = 0;   ///< 0 => no PQ
  double cache_ratio = 0.05;  ///< BFS cache fraction
  uint32_t num_threads = 0;   ///< 0 => all cores (Vamana build + PQ train/encode)
  uint32_t pq_train_iters = 15;
  uint64_t seed = 1234;
  bool verbose = false;  ///< print per-phase build wall-times to stderr
};

/// Load-time configuration (sizes the thread-scratch pool).
struct DiskANNLoadParams {
  uint32_t num_threads = 4;    ///< max concurrent searches (ThreadData pool size)
  uint32_t beam_width = 4;     ///< PQ I/O beam width (PQ caps in-flight reads at this)
  uint32_t nopq_io_depth = 0;  ///< No-PQ async pipeline depth (max reads in flight); 0 => 32,
                               ///< the benchmark-tuned default (SIFT1M/NVMe: ~2x single-query
                               ///< latency cut and +50% 8-thread throughput vs 2*beam_width,
                               ///< recall unchanged). No-PQ issues thousands of reads/query, so a
                               ///< deeper pipeline overlaps more I/O. Explicit values are floored
                               ///< at 2*beam_width and capped at the libaio context size (1024).
                               ///< Sizes the sector scratch to that many pages per thread.

  // --- In-place update mode (No-PQ only; see disk-update specs) ---
  bool updatable = false;          ///< open O_RDWR + enable insert/remove/update_node/flush
  uint32_t update_search_l = 100;  ///< L for the insert NN-search (candidate pool before prune)
  float update_alpha = 1.2f;       ///< alpha-RNG pruning for insert/reconnect (Vamana default)
  double safety_net_ratio = 0.05;  ///< tombstone ratio that arms the safety-net reconnect
  uint64_t safety_net_ops = 16;    ///< deletes without an insert before the safety net may fire
};

/// Per-query search configuration.
struct DiskANNSearchParams {
  uint32_t search_list_size = 100;  ///< L (retset capacity)
  bool use_pq = true;               ///< use PQ approx distances (ignored if no PQ)
  bool rerank = true;               ///< PQ only: re-score top candidates with exact L2
  uint32_t rerank_count = 0;        ///< PQ rerank pool size; 0 => top_k*3 (spec default)
  bool deterministic = false;       ///< Reproducible batch==sequential via a per-expansion
                                    ///< barrier (PQ: per-beam; ~10-15% slower). Default off =
                                    ///< async-pipelined I/O. Applies to both PQ and No-PQ.
};

class DiskANNIndex {
 public:
  static constexpr uint64_t kMetaMagic = 0x414C594144534B4EULL;  // "ALYADSKN"
  /// v2 adds max_slot_id + live_count for in-place updates; v1 is read back-compat.
  static constexpr uint32_t kMetaVersion = 2;
  /// Default No-PQ async pipeline depth when DiskANNLoadParams::nopq_io_depth == 0.
  /// Benchmark-tuned on SIFT1M/NVMe (knee at ~32; deeper gives no gain).
  static constexpr uint32_t kDefaultNoPQIoDepth = 32;

  DiskANNIndex() = default;
  ~DiskANNIndex() { teardown(); }

  DiskANNIndex(const DiskANNIndex &) = delete;
  DiskANNIndex &operator=(const DiskANNIndex &) = delete;
  DiskANNIndex(DiskANNIndex &&) = delete;
  DiskANNIndex &operator=(DiskANNIndex &&) = delete;

  // ------------------------------------------------------------------ build
  /**
   * @brief Build a complete index directory from vectors + external labels.
   * @throws std::invalid_argument for dim==0 / n==0 / null inputs.
   * @throws std::runtime_error if @p index_dir already exists.
   */
  static void build(const std::string &index_dir,
                    const float *vectors,
                    const uint64_t *labels,
                    uint64_t n,
                    uint64_t dim,
                    const DiskANNBuildParams &params) {
    if (dim == 0) {
      throw std::invalid_argument("DiskANNIndex::build: dim must be > 0");
    }
    if (n == 0) {
      throw std::invalid_argument("DiskANNIndex::build: n must be > 0");
    }
    if (vectors == nullptr || labels == nullptr) {
      throw std::invalid_argument("DiskANNIndex::build: null vectors/labels");
    }
    if (params.pq_n_chunks > 0 && dim % params.pq_n_chunks != 0) {
      throw std::invalid_argument("DiskANNIndex::build: dim not divisible by pq_n_chunks");
    }
    namespace fs = std::filesystem;
    if (fs::exists(index_dir)) {
      throw std::runtime_error("DiskANNIndex::build: index_dir already exists: " + index_dir);
    }
    if (!fs::create_directories(index_dir)) {
      throw std::runtime_error("DiskANNIndex::build: cannot create " + index_dir);
    }
    // Remove the partial directory if any build phase throws, so the failed build
    // does not block a retry with "index_dir already exists".
    struct DirGuard {
      const std::string &dir;
      bool committed = false;
      ~DirGuard() {
        if (!committed) {
          std::error_code ec;
          std::filesystem::remove_all(dir, ec);
        }
      }
    } dir_guard{index_dir};

    // Per-phase wall-clock timing (opt-in; mirrors official build_disk_index logging).
    using clk = std::chrono::steady_clock;
    auto stamp = [&params](const char *name, clk::time_point a, clk::time_point b) {
      if (params.verbose) {
        std::cerr << "[build] " << name << ": " << std::chrono::duration<double>(b - a).count()
                  << " s\n";
      }
    };

    // 1. Vamana graph.
    auto t_vamana0 = clk::now();
    alaya::vamana::VamanaBuildParams vparams;
    vparams.R = params.R;
    vparams.L = params.L;
    vparams.alpha = params.alpha;
    vparams.num_threads = params.num_threads;
    vparams.seed = params.seed;
    alaya::vamana::VamanaBuilder builder(vectors, n, static_cast<uint32_t>(dim), vparams);
    builder.build();
    const auto &graph = builder.graph();
    const uint32_t medoid = builder.medoid();
    auto t_vamana1 = clk::now();
    stamp("vamana", t_vamana0, t_vamana1);

    const DiskLayoutGeometry geom = DiskLayoutGeometry::compute(dim, params.R);

    // 2. Sector-aligned disk layout.  3. External labels.
    write_disk_layout(path(index_dir, "diskann.index"), vectors, graph, {n, dim, params.R, medoid});
    write_ids(path(index_dir, "ids.bin"), labels, n);
    auto t_layout = clk::now();
    stamp("layout+ids", t_vamana1, t_layout);

    // 4. Optional PQ.
    const bool has_pq = params.pq_n_chunks > 0;
    if (has_pq) {
      PQTable pq;
      pq.train(vectors,
               n,
               dim,
               params.pq_n_chunks,
               params.pq_train_iters,
               params.seed,
               params.num_threads);
      pq.encode(vectors, n, params.num_threads);
      pq.save(path(index_dir, "pq_pivots.bin"), path(index_dir, "pq_compressed.bin"));
    }
    auto t_pq = clk::now();
    stamp("pq(train+encode+save)", t_layout, t_pq);

    // 5. BFS cache.
    NodeCache cache;
    cache.generate(graph, vectors, medoid, n, dim, params.R, params.cache_ratio);
    cache.save(path(index_dir, "cache_ids.bin"), path(index_dir, "cache_nodes.bin"));
    stamp("cache", t_pq, clk::now());

    // 6. Metadata.
    MetaHeader meta;
    meta.num_points = n;
    meta.dim = dim;
    meta.max_degree = params.R;
    meta.medoid = medoid;
    meta.has_pq = has_pq ? 1 : 0;
    meta.pq_n_chunks = params.pq_n_chunks;
    meta.node_len = geom.node_len;
    meta.nodes_per_sector = geom.nodes_per_sector;
    meta.max_slot_id = n;  // fresh build: file capacity == num_points
    meta.live_count = n;   // fresh build: every slot is live
    write_meta(path(index_dir, "meta.bin"), meta);

    dir_guard.committed = true;  // build complete — keep the directory
  }

  // ------------------------------------------------------------------- load
  void load(const std::string &index_dir, const DiskANNLoadParams &params = {}) {
    teardown();
    namespace fs = std::filesystem;
    if (!fs::exists(index_dir) || !fs::is_directory(index_dir)) {
      throw std::runtime_error("DiskANNIndex::load: not a directory: " + index_dir);
    }

    const MetaHeader meta = read_meta(path(index_dir, "meta.bin"));
    const uint64_t num_points = meta.num_points;
    dim_ = meta.dim;
    max_degree_ = meta.max_degree;
    medoid_ = meta.medoid;
    has_pq_ = meta.has_pq != 0;
    pq_n_chunks_ = meta.pq_n_chunks;
    max_slot_id_ = meta.max_slot_id;  // file capacity (slots); == num_points for static/v1
    live_count_ = meta.live_count;    // live nodes; == num_points for static/v1
    geom_ = DiskLayoutGeometry::compute(dim_, max_degree_);
    if (geom_.node_len != meta.node_len || geom_.nodes_per_sector != meta.nodes_per_sector) {
      throw std::runtime_error("DiskANNIndex::load: meta geometry inconsistent");
    }

    read_ids(path(index_dir, "ids.bin"), max_slot_id_);
    cache_.load(path(index_dir, "cache_ids.bin"), path(index_dir, "cache_nodes.bin"));
    if (has_pq_) {
      pq_.load(path(index_dir, "pq_pivots.bin"),
               path(index_dir, "pq_compressed.bin"),
               num_points,
               dim_,
               pq_n_chunks_);
    }

    // Open the disk index and pre-register one libaio I/O context + scratch
    // buffer per pool slot. Each context is created on a short-lived
    // registration thread, then borrowed by whichever search thread pops the
    // slot. A libaio context is a process-wide handle, and the slot pool
    // guarantees only one thread uses a given context at a time, so this is safe
    // (verified by the concurrency stress test). The deterministic beam loop
    // (beam_search.hpp) makes results independent of I/O completion timing.
    reader_ = make_aligned_file_reader();
    reader_->open(path(index_dir, "diskann.index"));
    beam_width_ = std::max<uint32_t>(1, params.beam_width);
    const uint32_t pool = std::max<uint32_t>(1, params.num_threads);
    const uint32_t pq_table_entries = has_pq_ ? pq_n_chunks_ * kPQNumCentroids : 0;
    // One page slot per concurrent read. nopq_io_depth = 0 resolves to the
    // benchmark-tuned default (kDefaultNoPQIoDepth = 32); No-PQ issues far more
    // reads than PQ, so this deeper pipeline overlaps more I/O. Floored at
    // 2*beam_width (PQ's needs) and capped at MAX_EVENTS (libaio context size).
    const uint64_t nopq_depth =
        params.nopq_io_depth == 0 ? kDefaultNoPQIoDepth : params.nopq_io_depth;
    const uint64_t scratch_slots =
        std::min<uint64_t>(1024, std::max<uint64_t>(2ull * beam_width_, nopq_depth));
    thread_data_storage_.resize(pool);
    {
      std::vector<std::thread> regs;
      regs.reserve(pool);
      for (uint32_t t = 0; t < pool; ++t) {
        regs.emplace_back([this, t, pq_table_entries, scratch_slots]() {
          reader_->register_thread();
          auto td = std::make_unique<ThreadData>();
          td->ctx_ = reader_->get_ctx();
          td->alloc_scratch(scratch_slots, geom_.page_size, pq_table_entries);
          thread_data_storage_[t] = std::move(td);
        });
      }
      for (auto &th : regs) {
        th.join();
      }
    }
    for (auto &td : thread_data_storage_) {
      ThreadData *p = td.get();
      thread_data_pool_.push(p);
    }
    num_pool_ = pool;

    if (params.updatable) {
      init_updatable(index_dir, params);
    }
    loaded_ = true;
  }

  // ----------------------------------------------------------------- search
  /**
   * @brief Single-query search. Writes up to @p top_k external labels +
   *        distances (ascending L2) into the caller's buffers.
   * @return number of results written (<= top_k).
   */
  uint32_t search(const float *query,
                  uint32_t top_k,
                  uint64_t *out_labels,
                  float *out_distances,
                  const DiskANNSearchParams &params = {},
                  SearchStats *stats = nullptr) const {
    if (!loaded_) {
      throw std::runtime_error("DiskANNIndex::search: index not loaded");
    }
    if (query == nullptr) {
      throw std::invalid_argument("DiskANNIndex::search: null query");
    }
    if (top_k == 0) {
      throw std::invalid_argument("DiskANNIndex::search: top_k must be > 0");
    }
    if (out_labels == nullptr || out_distances == nullptr) {
      throw std::invalid_argument("DiskANNIndex::search: null output buffers");
    }

    std::unique_lock<std::mutex> serial_lock;
    if (updatable_) {
      serial_lock = std::unique_lock<std::mutex>(update_mutex_);
    }

    ThreadData *td = acquire();
    uint32_t count = 0;
    try {
      SearchContext ctx;
      ctx.reader = reader_.get();
      ctx.geom = &geom_;
      ctx.cache = &cache_;
      ctx.pq = has_pq_ ? &pq_ : nullptr;
      ctx.medoid = medoid_;
      ctx.num_points = max_slot_id_;
      if (updatable_) {
        ctx.tombstone = &slot_alloc_.tombstone();
      }

      SearchParams sp;
      sp.search_list_size = params.search_list_size;
      sp.beam_width = beam_width_;
      sp.use_pq = params.use_pq && has_pq_;
      sp.rerank = params.rerank;
      sp.rerank_count = params.rerank_count;
      sp.deterministic = params.deterministic;

      const auto results = cached_beam_search(ctx, query, top_k, sp, *td, stats);
      count = static_cast<uint32_t>(results.size());
      for (uint32_t i = 0; i < count; ++i) {
        out_labels[i] = labels_[results[i].first];
        out_distances[i] = results[i].second;
      }
    } catch (...) {
      release(td);
      throw;
    }
    release(td);

    // Pad unused slots with sentinels so callers can detect short result rows.
    for (uint32_t i = count; i < top_k; ++i) {
      out_labels[i] = kNoLabel;
      out_distances[i] = std::numeric_limits<float>::max();
    }
    return count;
  }

  // ------------------------------------------------------------ batch search
  /**
   * @brief Run @p n_queries searches across @p num_threads workers; results are
   *        written row-major into @p out_labels / @p out_distances
   *        (n_queries * top_k).
   */
  void batch_search(const float *queries,
                    uint32_t n_queries,
                    uint32_t top_k,
                    uint64_t *out_labels,
                    float *out_distances,
                    uint32_t num_threads,
                    const DiskANNSearchParams &params = {}) const {
    if (!loaded_) {
      throw std::runtime_error("DiskANNIndex::batch_search: index not loaded");
    }
    if (queries == nullptr) {
      throw std::invalid_argument("DiskANNIndex::batch_search: null queries");
    }
    if (top_k == 0) {
      throw std::invalid_argument("DiskANNIndex::batch_search: top_k must be > 0");
    }

    auto run_one = [&](uint32_t qi) {
      search(queries + static_cast<uint64_t>(qi) * dim_,
             top_k,
             out_labels + static_cast<uint64_t>(qi) * top_k,
             out_distances + static_cast<uint64_t>(qi) * top_k,
             params);
    };

    const uint32_t workers = std::min(std::max<uint32_t>(1, num_threads), num_pool_);
    if (workers <= 1) {
      for (uint32_t qi = 0; qi < n_queries; ++qi) {
        run_one(qi);
      }
      return;
    }

    std::atomic<uint32_t> next{0};
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (uint32_t w = 0; w < workers; ++w) {
      pool.emplace_back([&]() {
        for (;;) {
          const uint32_t qi = next.fetch_add(1);
          if (qi >= n_queries) {
            break;
          }
          run_one(qi);
        }
      });
    }
    for (auto &th : pool) {
      th.join();
    }
  }

  // --------------------------------------------------------------- accessors
  [[nodiscard]] uint64_t size() const { return live_count_; }  // live (non-tombstoned) vectors
  [[nodiscard]] uint64_t dim() const { return dim_; }
  [[nodiscard]] bool has_pq() const { return has_pq_; }
  [[nodiscard]] uint32_t medoid() const { return medoid_; }
  [[nodiscard]] bool updatable() const { return updatable_; }
  [[nodiscard]] uint64_t live_count() const { return live_count_; }
  [[nodiscard]] uint64_t max_slot_id() const { return max_slot_id_; }
  [[nodiscard]] uint64_t tombstone_count() const { return slot_alloc_.tombstone_count(); }
  [[nodiscard]] uint64_t free_slot_count() const { return slot_alloc_.free_count(); }
  [[nodiscard]] uint64_t safety_net_fire_count() const { return safety_net_fires_; }
  [[nodiscard]] bool is_deleted(uint32_t id) const { return slot_alloc_.is_deleted(id); }

  /// Sentinel label for padded (missing) result slots.
  static constexpr uint64_t kNoLabel = std::numeric_limits<uint64_t>::max();

  // ------------------------------------------------------- in-place updates
  /// Insert a vector: NN search -> alpha-RNG prune -> alloc slot -> write ->
  /// reconnect pruned neighbors. Returns the allocated internal slot id.
  uint32_t insert(const float *query, uint64_t label) {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::insert: index not loaded in updatable mode");
    }
    if (query == nullptr) {
      throw std::invalid_argument("DiskANNIndex::insert: null query");
    }
    std::lock_guard<std::mutex> lock(update_mutex_);
    page_io_->clear_cache();

    const auto cand = run_update_search(query, update_search_l_);

    std::vector<alaya::vamana::Neighbor> pool;
    pool.reserve(cand.size());
    for (const auto &c : cand) {
      pool.emplace_back(c.first, c.second);
    }
    std::vector<uint32_t> pruned;
    std::vector<float> occlude_scratch;
    auto dist_fn = [this](uint32_t a, uint32_t b) -> float {
      return cached_l2(a, b);
    };
    const uint32_t maxc = std::max<uint32_t>(max_degree_, static_cast<uint32_t>(pool.size()));
    alaya::vamana::prune_neighbors(kNoSelf,
                                   pool,
                                   update_alpha_,
                                   max_degree_,
                                   maxc,
                                   pruned,
                                   occlude_scratch,
                                   dist_fn);

    const uint32_t slot = slot_alloc_.alloc();
    update_ctx_.forget_slot(slot);
    max_slot_id_ = std::max<uint64_t>(max_slot_id_, slot_alloc_.next_fresh_id());

    page_io_->write_node(slot, query, static_cast<uint32_t>(pruned.size()), pruned.data());

    set_label(slot, label);
    ++live_count_;
    ops_since_last_insert_ = 0;

    for (const uint32_t n : pruned) {
      update_ctx_.inserted_edges_[n].push_back(slot);
    }
    for (const uint32_t n : pruned) {
      update_node_impl(n);
    }
    return slot;
  }

  /// Lazy-delete: cache old neighbors for two-hop, tombstone + free the slot.
  /// Reconnect is deferred to the next insert or the safety net.
  void remove(uint32_t internal_id) {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::remove: index not loaded in updatable mode");
    }
    std::lock_guard<std::mutex> lock(update_mutex_);
    if (internal_id >= max_slot_id_) {
      throw std::invalid_argument("DiskANNIndex::remove: internal_id out of range");
    }
    if (slot_alloc_.is_deleted(internal_id)) {
      throw std::invalid_argument("DiskANNIndex::remove: node already deleted");
    }
    page_io_->clear_cache();

    // 1. Read coords + old neighbors before tombstoning.
    const DiskPageIO::NodeData nd = page_io_->read_node(internal_id);
    update_ctx_.removed_node_nbrs_[internal_id] = nd.nbrs;

    // 2. Tombstone + free slot.
    slot_alloc_.free(internal_id);
    --live_count_;
    ++ops_since_last_insert_;

    // 3. IP-DiskANN: search around deleted point for repair candidates.
    const auto cand = run_update_search(nd.coords.data(), update_search_l_);
    std::vector<uint32_t> candidates;
    candidates.reserve(cand.size());
    for (const auto &c : cand) {
      candidates.push_back(c.first);
    }

    // 4. Nodes to reconnect = old neighbors ∪ search results (live only).
    std::unordered_set<uint32_t> nodes_to_upd;
    for (const uint32_t nbr : nd.nbrs) {
      if (!slot_alloc_.is_deleted(nbr)) {
        nodes_to_upd.insert(nbr);
      }
    }
    for (const uint32_t c : candidates) {
      if (!slot_alloc_.is_deleted(c)) {
        nodes_to_upd.insert(c);
      }
    }

    // 5. IP-DiskANN reconnect: each affected node gets c replacement edges.
    page_io_->clear_cache();
    for (const uint32_t nid : nodes_to_upd) {
      update_node_ipdiskann(nid, candidates);
    }
    maybe_safety_net_reconnect();
  }

  /**
   * @brief Reconnect @p node_id's neighbor list in place (Yi's connect-task).
   * @throws std::invalid_argument if @p node_id is out of range.
   */
  void update_node(uint32_t node_id) {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::update_node: index not loaded in updatable mode");
    }
    std::lock_guard<std::mutex> lock(update_mutex_);
    if (node_id >= max_slot_id_) {
      throw std::invalid_argument("DiskANNIndex::update_node: node_id out of range");
    }
    if (slot_alloc_.is_deleted(node_id)) {
      throw std::invalid_argument("DiskANNIndex::update_node: node_id is deleted");
    }
    page_io_->clear_cache();
    update_node_impl(node_id);
  }

  /// Persist meta + ids + slot allocator state to disk.
  void flush() {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::flush: index not loaded in updatable mode");
    }
    std::lock_guard<std::mutex> lock(update_mutex_);
    MetaHeader m;
    m.num_points = max_slot_id_;
    m.dim = dim_;
    m.max_degree = max_degree_;
    m.medoid = medoid_;
    m.has_pq = 0;
    m.pq_n_chunks = 0;
    m.node_len = geom_.node_len;
    m.nodes_per_sector = geom_.nodes_per_sector;
    m.max_slot_id = max_slot_id_;
    m.live_count = live_count_;
    write_meta(path(index_dir_, "meta.bin"), m);
    write_ids(path(index_dir_, "ids.bin"), labels_.data(), labels_.size());
    slot_alloc_.save(path(index_dir_, "slots.bin"));
  }

 private:
  static constexpr uint32_t kNoSelf = std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t kIPDiskANNCopies = 3;

  struct MetaHeader {
    uint64_t num_points = 0;
    uint64_t dim = 0;
    uint32_t max_degree = 0;
    uint32_t medoid = 0;
    uint8_t has_pq = 0;
    uint32_t pq_n_chunks = 0;
    uint64_t node_len = 0;
    uint64_t nodes_per_sector = 0;
    uint64_t max_slot_id = 0;  // v2: file capacity in slots (only grows)
    uint64_t live_count = 0;   // v2: num_points minus tombstones
  };

  static std::string path(const std::string &dir, const char *name) {
    return (std::filesystem::path(dir) / name).string();
  }

  // ---- in-place update internals (all called under update_mutex_) ----

  /// Wire up the update subsystem (page IO, slot allocator, context, config).
  void init_updatable(const std::string &index_dir, const DiskANNLoadParams &params) {
    if (has_pq_) {
      throw std::runtime_error(
          "DiskANNIndex::load: updatable mode requires a No-PQ index (PQ updates unsupported)");
    }
    index_dir_ = index_dir;
    update_alpha_ = params.update_alpha;
    update_search_l_ = std::max<uint32_t>(1, params.update_search_l);
    safety_net_ratio_ = params.safety_net_ratio;
    safety_net_ops_ = params.safety_net_ops;
    safety_net_fires_ = 0;
    ops_since_last_insert_ = 0;
    update_ctx_.clear();

    page_io_ = std::make_unique<DiskPageIO>(path(index_dir, "diskann.index"), geom_);

    const std::string slots_path = path(index_dir, "slots.bin");
    if (std::filesystem::exists(slots_path)) {
      slot_alloc_.load(slots_path);  // restore free list + next id + tombstones
      max_slot_id_ = std::max<uint64_t>(max_slot_id_, slot_alloc_.next_fresh_id());
    } else {
      slot_alloc_.reset(static_cast<uint32_t>(max_slot_id_));
    }
    updatable_ = true;
  }

  /// Tombstone-aware exact-L2 NN search returning up to @p l (id, dist) candidates.
  std::vector<std::pair<uint32_t, float>> run_update_search(const float *query, uint32_t l) {
    ThreadData *td = acquire();
    std::vector<std::pair<uint32_t, float>> results;
    try {
      SearchContext ctx;
      ctx.reader = reader_.get();
      ctx.geom = &geom_;
      ctx.cache = &cache_;
      ctx.pq = nullptr;
      ctx.medoid = medoid_;
      ctx.num_points = max_slot_id_;
      ctx.tombstone = &slot_alloc_.tombstone();
      SearchParams sp;
      sp.search_list_size = l;
      sp.beam_width = beam_width_;
      sp.use_pq = false;
      sp.rerank = false;
      sp.deterministic = true;  // reproducible graph construction
      results = disk_greedy_search(ctx, query, l, sp, *td, nullptr);
    } catch (...) {
      release(td);
      throw;
    }
    release(td);
    return results;
  }

  /// Cached L2 distance between two nodes via the page IO coords cache.
  float cached_l2(uint32_t a, uint32_t b) {
    const auto l2 = alaya::simd::get_l2_sqr_func();
    const std::vector<float> &ca = page_io_->read_coords_cached(a);
    const std::vector<float> &cb = page_io_->read_coords_cached(b);
    return l2(ca.data(), cb.data(), dim_);
  }

  /// Shared reconnect backbone: score candidates, prune, write if changed,
  /// consume inserted_edges for node_id.
  void prune_and_write(uint32_t node_id,
                       const std::vector<uint32_t> &old_nbrs,
                       std::unordered_set<uint32_t> &cand) {
    const auto ins_it = update_ctx_.inserted_edges_.find(node_id);
    if (ins_it != update_ctx_.inserted_edges_.end()) {
      for (const uint32_t v : ins_it->second) {
        if (v != node_id && !slot_alloc_.is_deleted(v)) {
          cand.insert(v);
        }
      }
    }

    if (!cand.empty()) {
      const std::vector<float> &self_coords = page_io_->read_coords_cached(node_id);
      const auto l2 = alaya::simd::get_l2_sqr_func();
      std::vector<alaya::vamana::Neighbor> pool;
      pool.reserve(cand.size());
      for (const uint32_t c : cand) {
        const std::vector<float> &cc = page_io_->read_coords_cached(c);
        pool.emplace_back(c, l2(self_coords.data(), cc.data(), dim_));
      }

      std::vector<uint32_t> new_nbrs;
      if (pool.size() <= max_degree_) {
        std::sort(pool.begin(), pool.end());
        for (const auto &nb : pool) {
          new_nbrs.push_back(nb.id);
        }
      } else {
        std::vector<float> occlude_scratch;
        auto dist_fn = [this](uint32_t a, uint32_t b) -> float {
          return cached_l2(a, b);
        };
        alaya::vamana::prune_neighbors(node_id,
                                       pool,
                                       update_alpha_,
                                       max_degree_,
                                       static_cast<uint32_t>(pool.size()),
                                       new_nbrs,
                                       occlude_scratch,
                                       dist_fn);
      }

      if (!same_neighbor_set(old_nbrs, new_nbrs)) {
        page_io_->write_node_neighbors(node_id,
                                       static_cast<uint32_t>(new_nbrs.size()),
                                       new_nbrs.data());
      }
    }

    if (ins_it != update_ctx_.inserted_edges_.end()) {
      update_ctx_.inserted_edges_.erase(ins_it);
    }
  }

  /// Reconnect node_id's neighbor list: candidates = live old + two-hop through
  /// deleted + inserted reverse edges, exact-L2 ranked, alpha-RNG pruned.
  void update_node_impl(uint32_t node_id) {
    const DiskPageIO::NodeData nd = page_io_->read_node(node_id);

    std::unordered_set<uint32_t> cand;
    for (const uint32_t nbr : nd.nbrs) {
      if (nbr == node_id) {
        continue;
      }
      if (slot_alloc_.is_deleted(nbr)) {
        const auto it = update_ctx_.removed_node_nbrs_.find(nbr);
        if (it != update_ctx_.removed_node_nbrs_.end()) {
          for (const uint32_t o : it->second) {
            if (o != node_id && !slot_alloc_.is_deleted(o)) {
              cand.insert(o);
            }
          }
        }
      } else {
        cand.insert(nbr);
      }
    }
    prune_and_write(node_id, nd.nbrs, cand);
  }

  /// IP-DiskANN reconnect: keep live old neighbors + c closest candidates, prune.
  void update_node_ipdiskann(uint32_t node_id, const std::vector<uint32_t> &candidates) {
    const DiskPageIO::NodeData nd = page_io_->read_node(node_id);

    std::unordered_set<uint32_t> cand;
    for (const uint32_t nbr : nd.nbrs) {
      if (nbr != node_id && !slot_alloc_.is_deleted(nbr)) {
        cand.insert(nbr);
      }
    }

    const std::vector<float> &self_coords = page_io_->read_coords_cached(node_id);
    const auto l2 = alaya::simd::get_l2_sqr_func();
    std::vector<std::pair<float, uint32_t>> scored;
    for (const uint32_t c : candidates) {
      if (c != node_id && !slot_alloc_.is_deleted(c) && cand.find(c) == cand.end()) {
        const std::vector<float> &cc = page_io_->read_coords_cached(c);
        scored.emplace_back(l2(self_coords.data(), cc.data(), dim_), c);
      }
    }
    std::partial_sort(scored.begin(),
                      scored.begin() +
                          static_cast<ptrdiff_t>(std::min<size_t>(kIPDiskANNCopies, scored.size())),
                      scored.end());
    for (size_t i = 0; i < std::min<size_t>(kIPDiskANNCopies, scored.size()); ++i) {
      cand.insert(scored[i].second);
    }

    prune_and_write(node_id, nd.nbrs, cand);
  }

  /// Lightweight consolidation: just strip dangling edges to tombstoned nodes.
  void maybe_safety_net_reconnect() {
    if (!update_ctx_.needs_safety_net_reconnect(safety_net_ratio_,
                                                slot_alloc_.tombstone_count(),
                                                max_slot_id_,
                                                ops_since_last_insert_,
                                                safety_net_ops_)) {
      return;
    }
    std::unordered_set<uint32_t> affected;
    for (const auto &entry : update_ctx_.removed_node_nbrs_) {
      for (const uint32_t nb : entry.second) {
        if (!slot_alloc_.is_deleted(nb)) {
          affected.insert(nb);
        }
      }
    }
    page_io_->clear_cache();
    for (const uint32_t nid : affected) {
      const DiskPageIO::NodeData nd = page_io_->read_node(nid);
      std::vector<uint32_t> live;
      for (const uint32_t nbr : nd.nbrs) {
        if (!slot_alloc_.is_deleted(nbr)) {
          live.push_back(nbr);
        }
      }
      if (live.size() != nd.nbrs.size()) {
        page_io_->write_node_neighbors(nid, static_cast<uint32_t>(live.size()), live.data());
      }
    }
    ops_since_last_insert_ = 0;
    ++safety_net_fires_;
  }

  /// Grow labels_ on append; assign in place on slot reuse.
  void set_label(uint32_t slot, uint64_t label) {
    if (slot >= labels_.size()) {
      labels_.resize(static_cast<size_t>(slot) + 1, kNoLabel);
    }
    labels_[slot] = label;
  }

  /// Order-independent equality of two neighbor id lists.
  static bool same_neighbor_set(std::vector<uint32_t> a, std::vector<uint32_t> b) {
    if (a.size() != b.size()) {
      return false;
    }
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a == b;
  }

  ThreadData *acquire() const {
    ThreadData *td = thread_data_pool_.pop();
    while (td == nullptr) {
      thread_data_pool_.wait_for_push_notify();
      td = thread_data_pool_.pop();
    }
    return td;
  }
  void release(ThreadData *td) const {
    thread_data_pool_.push(td);
    thread_data_pool_.push_notify_all();
  }

  void teardown() {
    if (reader_) {
      reader_->close();
      reader_->deregister_all_threads();
    }
    for (auto &td : thread_data_storage_) {
      if (td) {
        td->free_scratch();
      }
    }
    thread_data_storage_.clear();
    while (thread_data_pool_.pop() != nullptr) {
    }
    reader_.reset();
    page_io_.reset();
    update_ctx_.clear();
    updatable_ = false;
    loaded_ = false;
  }

  // --- meta.bin / ids.bin serialization ---
  static void write_meta(const std::string &p, const MetaHeader &m) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("DiskANNIndex: cannot write " + p);
    }
    const uint64_t magic = kMetaMagic;
    const uint32_t version = kMetaVersion;
    auto w = [&](const void *d, size_t n) {
      out.write(reinterpret_cast<const char *>(d), n);
    };
    w(&magic, sizeof(magic));
    w(&version, sizeof(version));
    w(&m.num_points, sizeof(m.num_points));
    w(&m.dim, sizeof(m.dim));
    w(&m.max_degree, sizeof(m.max_degree));
    w(&m.medoid, sizeof(m.medoid));
    w(&m.has_pq, sizeof(m.has_pq));
    w(&m.pq_n_chunks, sizeof(m.pq_n_chunks));
    w(&m.node_len, sizeof(m.node_len));
    w(&m.nodes_per_sector, sizeof(m.nodes_per_sector));
    w(&m.max_slot_id, sizeof(m.max_slot_id));  // v2
    w(&m.live_count, sizeof(m.live_count));    // v2
    if (!out) {
      throw std::runtime_error("DiskANNIndex: meta write failed " + p);
    }
  }

  static MetaHeader read_meta(const std::string &p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
      throw std::runtime_error("DiskANNIndex::load: cannot open meta " + p);
    }
    uint64_t magic = 0;
    uint32_t version = 0;
    MetaHeader m;
    auto r = [&](void *d, size_t n) {
      in.read(reinterpret_cast<char *>(d), n);
    };
    r(&magic, sizeof(magic));
    r(&version, sizeof(version));
    r(&m.num_points, sizeof(m.num_points));
    r(&m.dim, sizeof(m.dim));
    r(&m.max_degree, sizeof(m.max_degree));
    r(&m.medoid, sizeof(m.medoid));
    r(&m.has_pq, sizeof(m.has_pq));
    r(&m.pq_n_chunks, sizeof(m.pq_n_chunks));
    r(&m.node_len, sizeof(m.node_len));
    r(&m.nodes_per_sector, sizeof(m.nodes_per_sector));
    if (!in) {
      throw std::runtime_error("DiskANNIndex::load: meta.bin truncated/corrupt " + p);
    }
    if (magic != kMetaMagic) {
      throw std::runtime_error("DiskANNIndex::load: bad meta magic " + p);
    }
    if (version == 1) {
      // v1 predates in-place updates: every slot is live, capacity == num_points.
      m.max_slot_id = m.num_points;
      m.live_count = m.num_points;
    } else if (version == 2) {
      r(&m.max_slot_id, sizeof(m.max_slot_id));
      r(&m.live_count, sizeof(m.live_count));
      if (!in) {
        throw std::runtime_error("DiskANNIndex::load: meta.bin truncated/corrupt " + p);
      }
    } else {
      throw std::runtime_error("DiskANNIndex::load: unsupported meta version " + p);
    }
    if (m.num_points == 0 || m.dim == 0) {
      throw std::runtime_error("DiskANNIndex::load: zero num_points/dim in meta " + p);
    }
    return m;
  }

  static void write_ids(const std::string &p, const uint64_t *labels, uint64_t n) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("DiskANNIndex: cannot write " + p);
    }
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    out.write(reinterpret_cast<const char *>(labels),
              static_cast<std::streamsize>(n * sizeof(uint64_t)));
    if (!out) {
      throw std::runtime_error("DiskANNIndex: ids write failed " + p);
    }
  }

  void read_ids(const std::string &p, uint64_t expected_n) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
      throw std::runtime_error("DiskANNIndex::load: cannot open ids " + p);
    }
    uint64_t count = 0;
    in.read(reinterpret_cast<char *>(&count), sizeof(count));
    if (!in || count != expected_n) {
      throw std::runtime_error("DiskANNIndex::load: ids.bin count mismatch " + p);
    }
    labels_.assign(count, 0);
    in.read(reinterpret_cast<char *>(labels_.data()),
            static_cast<std::streamsize>(count * sizeof(uint64_t)));
    if (!in) {
      throw std::runtime_error("DiskANNIndex::load: ids.bin truncated " + p);
    }
  }

  // metadata
  uint64_t dim_ = 0;
  uint32_t max_degree_ = 0;
  uint32_t medoid_ = 0;
  bool has_pq_ = false;
  uint32_t pq_n_chunks_ = 0;
  uint32_t beam_width_ = 4;
  uint32_t num_pool_ = 0;
  bool loaded_ = false;
  DiskLayoutGeometry geom_;

  // in-memory artifacts
  std::vector<uint64_t> labels_;
  NodeCache cache_;
  PQTable pq_;
  std::unique_ptr<AlignedFileReader> reader_;

  // thread-scratch pool
  std::vector<std::unique_ptr<ThreadData>> thread_data_storage_;
  mutable ::ConcurrentQueue<ThreadData *> thread_data_pool_{nullptr};

  // in-place update state (active only when updatable_)
  bool updatable_ = false;
  uint64_t max_slot_id_ = 0;  ///< file capacity in slots (valid-id bound; only grows)
  uint64_t live_count_ = 0;   ///< live (non-tombstoned) vector count
  std::string index_dir_;     ///< saved at load for flush() output paths
  std::unique_ptr<DiskPageIO> page_io_;
  SlotAllocator slot_alloc_;
  DiskUpdateContext update_ctx_;
  mutable std::mutex update_mutex_;  ///< serialises search + mutation (v1)
  uint64_t ops_since_last_insert_ = 0;
  uint64_t safety_net_fires_ = 0;
  float update_alpha_ = 1.2f;
  uint32_t update_search_l_ = 100;
  double safety_net_ratio_ = 0.05;
  uint64_t safety_net_ops_ = 16;
};

}  // namespace alaya::diskann
