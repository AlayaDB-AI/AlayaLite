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
#include <array>
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
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "coro/sync_wait.hpp"
#include "coro/task.hpp"
#include "coro/thread_pool.hpp"
#include "coro/when_all.hpp"
#include "index/graph/diskann/beam_search.hpp"
#include "index/graph/diskann/beam_search_async.hpp"
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
#include "storage/io/uring_reactor.hpp"
#include "utils/coro_gate.hpp"

namespace alaya::diskann {

inline constexpr uint32_t kDefaultDiskANNScratchSearchListSize = 150;
inline constexpr uint32_t kDefaultDiskANNUpdateReconnectThreads = 4;
inline constexpr uint32_t kDefaultDiskANNUpdateInsertThreads = 32;

/// Build-time configuration.
struct DiskANNBuildParams {
  uint32_t R = 64;     ///< graph degree bound (Vamana build)
  uint32_t L = 100;    ///< Vamana build beam width
  float alpha = 1.2f;  ///< Vamana α-RNG pruning
  uint32_t record_capacity = 0;
  ///< On-disk neighbor slots per node; 0 => R. Setting it above R leaves
  ///< update headroom the way Yi does (records hold MAX_NEIGHBOURS=96 while
  ///< the built graph uses ~64): reconnects keep candidate pools verbatim
  ///< until a node exceeds the CAPACITY, instead of alpha-pruning on every
  ///< touch of an already-full node. The update-time degree bound
  ///< (max_degree) becomes this capacity.
  uint32_t pq_n_chunks = 0;   ///< 0 => no PQ
  double cache_ratio = 0.05;  ///< BFS cache fraction
  uint32_t num_threads = 0;   ///< 0 => all cores (Vamana build + PQ train/encode)
  uint32_t pq_train_iters = 15;
  uint64_t seed = 1234;
  bool verbose = false;  ///< print per-phase build wall-times to stderr
};

/// Load-time configuration (sizes the thread-scratch pool).
/// Update-path page-read backend.
enum class DiskANNUpdateIO {
  kAuto,      ///< io_uring reactor when the kernel supports it, else blocking preads
  kUring,     ///< force the reactor; load() throws if io_uring is unavailable
  kBlocking,  ///< pool-thread-blocking preads (the pre-reactor behavior)
};

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
  uint32_t scratch_search_list_size = kDefaultDiskANNScratchSearchListSize;
  ///< No-PQ neighbor scratch capacity, in search-list entries. Set this >= the
  ///< largest DiskANNSearchParams::search_list_size used after load.

  // --- In-place update mode ---
  bool updatable = false;             ///< open O_RDWR + enable insert/remove/update_node/flush
  uint32_t update_search_l = 0;       ///< L for the insert NN-search; 0 => max_degree + 32
                                      ///< (Yi's build_k = degree + 32 rule)
  bool update_rerank = true;          ///< re-rank insert search candidates by exact L2 (via the
                                      ///< coords cache) before taking the top max_degree. Matches
                                      ///< Yi's trace benchmark (_rerank_flag defaults true); Yi's
                                      ///< sequential UpdateRunner sets it false.
  bool update_insert_prune = false;   ///< alpha-RNG prune the insert pool instead of linking the
                                      ///< top max_degree candidates. Yi never prunes at insert
                                      ///< (its top_k == degree makes the prune branch dead code);
                                      ///< reconnect re-prunes on overflow either way.
  float update_alpha = 1.2f;          ///< alpha-RNG pruning for insert/reconnect (Vamana default)
  double safety_net_ratio = 0.05;     ///< tombstone ratio that arms the safety-net reconnect
  uint64_t safety_net_ops = 16;       ///< deletes without an insert before the safety net may fire
  size_t page_cache_capacity = 4096;  ///< update-path page LRU cache capacity; 0 disables it
  uint32_t update_insert_threads = kDefaultDiskANNUpdateInsertThreads;
  ///< Coroutine worker count for batch_insert's outer insert concurrency.
  uint32_t update_reconnect_threads = kDefaultDiskANNUpdateReconnectThreads;
  ///< Coroutine worker count for reconnecting a single insert's pruned neighbors.
  DiskANNUpdateIO update_io = DiskANNUpdateIO::kAuto;
  ///< Page-read backend for the update path: with the io_uring reactor a page
  ///< miss SUSPENDS the coroutine (the pool thread runs other inserts) instead
  ///< of blocking in pread — Yi's tasklet behavior at plain pool sizes.
  uint32_t update_search_concurrency = 0;
  ///< Reactor mode only: concurrent suspended update searches (AsyncGate
  ///< ThreadData count). Little's law sets insert throughput to this divided
  ///< by search latency — worker threads bound CPU, this bounds in-flight
  ///< searches (Yi: workers vs tasklets). 0 = 4x update_insert_threads.
  bool search_page_cache = true;
  ///< Updatable indices: let EVERY search (query-path and update-path) peek
  ///< the shard page cache before each device read and fill it with the
  ///< pages it reads — the cache then behaves like Yi's unified buffer pool
  ///< (dynamic LRU shared by searches and updates) instead of a write-only
  ///< view. Off = pre-async behavior: static BFS cache + raw device reads.
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
    validate_unique_external_labels(labels, n, "build");
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

    // Records are sized for `capacity` neighbor slots (>= the built degree);
    // the update-time degree bound follows the capacity.
    const uint32_t capacity = params.record_capacity != 0 ? params.record_capacity : params.R;
    if (capacity < params.R) {
      throw std::invalid_argument("DiskANNIndex::build: record_capacity must be >= R");
    }
    const DiskLayoutGeometry geom = DiskLayoutGeometry::compute(dim, capacity);

    // 2. Sector-aligned disk layout.  3. External labels.
    write_disk_layout(path(index_dir, "diskann.index"), vectors, graph, {n, dim, capacity, medoid});
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
    cache.generate(graph, vectors, medoid, n, dim, capacity, params.cache_ratio);
    cache.save(path(index_dir, "cache_ids.bin"), path(index_dir, "cache_nodes.bin"));
    stamp("cache", t_pq, clk::now());

    // 6. Metadata.
    MetaHeader meta;
    meta.num_points = n;
    meta.dim = dim;
    meta.max_degree = capacity;
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
    cache_.configure_geometry(dim_, max_degree_);
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
    uint32_t pool = std::max<uint32_t>(1, params.num_threads);
    if (params.updatable) {
      if (params.update_insert_threads == 0) {
        throw std::invalid_argument("DiskANNIndex::load: update_insert_threads must be > 0");
      }
      if (params.update_reconnect_threads == 0) {
        throw std::invalid_argument("DiskANNIndex::load: update_reconnect_threads must be > 0");
      }
      pool = std::max({pool, params.update_insert_threads, params.update_reconnect_threads});
    }
    const uint32_t pq_table_entries = has_pq_ ? pq_n_chunks_ * kPQNumCentroids : 0;
    // One page slot per concurrent read. nopq_io_depth = 0 resolves to the
    // benchmark-tuned default (kDefaultNoPQIoDepth = 32); No-PQ issues far more
    // reads than PQ, so this deeper pipeline overlaps more I/O. Floored at
    // 2*beam_width (PQ's needs) and capped at MAX_EVENTS (libaio context size).
    const uint64_t nopq_depth =
        params.nopq_io_depth == 0 ? kDefaultNoPQIoDepth : params.nopq_io_depth;
    const uint64_t scratch_slots =
        std::min<uint64_t>(1024, std::max<uint64_t>(2ull * beam_width_, nopq_depth));
    const uint32_t scratch_list_size = std::max({DiskANNSearchParams{}.search_list_size,
                                                 params.update_search_l,
                                                 params.scratch_search_list_size});
    ThreadDataScratchConfig scratch_config;
    scratch_config.n_page_slots = scratch_slots;
    scratch_config.page_size = geom_.page_size;
    scratch_config.pq_table_entries = pq_table_entries;
    scratch_config.max_slot_id = max_slot_id_;
    scratch_config.max_degree = max_degree_;
    scratch_config.search_list_size = scratch_list_size;
    scratch_config.query_dim = dim_;
    scratch_config_ = scratch_config;  // reused by init_updatable's gate ThreadData set
    search_page_cache_ = params.search_page_cache;
    thread_data_storage_.resize(pool);
    {
      std::vector<std::thread> regs;
      regs.reserve(pool);
      for (uint32_t t = 0; t < pool; ++t) {
        regs.emplace_back([this, t, scratch_config]() {
          reader_->register_thread();
          auto td = std::make_unique<ThreadData>();
          td->ctx_ = reader_->get_ctx();
          td->alloc_scratch(scratch_config);
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

    const auto ts0 = std::chrono::steady_clock::now();
    const SearchSnapshot snapshot = make_search_snapshot();
    const bool use_pq = params.use_pq && has_pq_;
    std::shared_lock<std::shared_mutex> pq_lock;
    if (use_pq) {
      pq_lock = std::shared_lock<std::shared_mutex>(pq_mutex_);
    }

    ThreadData *td = acquire();
    uint32_t count = 0;
    std::vector<std::pair<uint32_t, float>> results;
    try {
      td->resize_slot_capacity(snapshot.max_slot_id);
      if (stats != nullptr) {
        stats->setup_us +=
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - ts0)
                                      .count());
      }
      SearchContext ctx;
      ctx.reader = reader_.get();
      ctx.geom = &geom_;
      ctx.cache = &cache_;
      ctx.pq = use_pq ? &pq_ : nullptr;
      ctx.medoid = medoid_;
      ctx.num_points = snapshot.max_slot_id;
      if (snapshot.has_tombstone) {
        ctx.tombstone = &snapshot.tombstone;
      }
      ctx.page_io = search_page_io();

      SearchParams sp;
      sp.search_list_size = params.search_list_size;
      sp.beam_width = beam_width_;
      sp.use_pq = use_pq;
      sp.rerank = params.rerank;
      sp.rerank_count = params.rerank_count;
      sp.deterministic = params.deterministic;

      results = cached_beam_search(ctx, query, top_k, sp, *td, stats);
      count = static_cast<uint32_t>(results.size());
    } catch (...) {
      release(td);
      throw;
    }
    release(td);

    {
      std::shared_lock<std::shared_mutex> state_lock(update_mutex_);
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t id = results[i].first;
        out_labels[i] = id < labels_.size() ? labels_[id] : kNoLabel;
        out_distances[i] = results[i].second;
      }
    }

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

#if defined(__linux__)
  // --------------------------------------------------- pipelined batch search
  /**
   * @brief Pre-allocates and first-touches the per-slot scratch for up to
   *        @p pipeline in-flight slots, using @p num_threads workers for the
   *        touch. search_pipelined self-heals without it, but pays the
   *        allocation inside its own wall clock — at depth 1024 on a 10M-slot
   *        index that is ~130 GB of first-touch, which would dominate a timed
   *        benchmark call. Idempotent.
   */
  void prewarm_pipeline(uint32_t num_threads, uint32_t pipeline) const {
    const uint32_t inflight = std::max<uint32_t>(1, pipeline);
    const uint32_t workers = std::max<uint32_t>(1, num_threads);
    std::lock_guard<std::mutex> pipeline_guard(pipeline_tds_mutex_);
    coro::thread_pool pool{{.thread_count = workers,
                            .on_thread_start_functor = nullptr,
                            .on_thread_stop_functor = nullptr}};
    ensure_pipeline_tds_locked(inflight, pool);
  }

  /**
   * @brief Query-level pipelined batch search: @p num_threads pool threads
   *        drive up to @p pipeline concurrent query coroutines; a query
   *        suspends on its beam-wave reads (io_uring reactor) instead of
   *        parking its thread, which then resumes another in-flight query.
   *        This is Yi's worker/tasklet split for the read-only search path:
   *        threads bound CPU, @p pipeline bounds in-flight queries, and
   *        throughput follows Little's law instead of 8/latency.
   *
   * Per-query semantics match search() with rerank off (the async beams sit in
   * the sync schedulers' recall-equivalence class). Rerank is rejected, as in
   * pq_beam_search_async — in the trace-bench protocol it is a no-op anyway
   * (every retset entry is expanded during traversal, so rerank re-reads 0
   * nodes). Requires an updatable load with update_io=uring for the shared
   * reactor. Each in-flight slot owns a private ThreadData (visited bits +
   * scratch), so @p pipeline trades memory for I/O overlap; the search()/
   * update ThreadData pools are not touched.
   *
   * @p per_query_stats / @p per_query_us: optional arrays of @p n_queries.
   */
  void search_pipelined(const float *queries,
                        uint32_t n_queries,
                        uint32_t top_k,
                        uint64_t *out_labels,
                        float *out_distances,
                        uint32_t num_threads,
                        uint32_t pipeline,
                        const DiskANNSearchParams &params = {},
                        SearchStats *per_query_stats = nullptr,
                        double *per_query_us = nullptr) const {
    if (!loaded_) {
      throw std::runtime_error("DiskANNIndex::search_pipelined: index not loaded");
    }
    if (queries == nullptr) {
      throw std::invalid_argument("DiskANNIndex::search_pipelined: null queries");
    }
    if (top_k == 0) {
      throw std::invalid_argument("DiskANNIndex::search_pipelined: top_k must be > 0");
    }
    if (out_labels == nullptr || out_distances == nullptr) {
      throw std::invalid_argument("DiskANNIndex::search_pipelined: null output buffers");
    }
    if (params.rerank) {
      throw std::invalid_argument("DiskANNIndex::search_pipelined: rerank is not supported");
    }
    if (!update_reactor_ || !reader_ || reader_->get_fd() < 0) {
      throw std::runtime_error(
          "DiskANNIndex::search_pipelined: requires an updatable load with "
          "update_io=uring (shared reactor)");
    }
    if (n_queries == 0) {
      return;
    }

    const bool use_pq = params.use_pq && has_pq_;
    const uint32_t inflight = std::max<uint32_t>(1, std::min(pipeline, n_queries));
    const uint32_t workers = std::max<uint32_t>(1, num_threads);
    const int fd = reader_->get_fd();
    // Rerank-pool semantics without rerank reads: ask the traversal for the
    // full search list (every entry carries its exact distance by the time it
    // is expanded), then cut the exact-sorted list to top_k below. Requesting
    // only top_k would pre-cut by PQ order — measurably worse recall.
    const uint32_t ask = std::max<uint32_t>(top_k, params.search_list_size);

    // One private ThreadData per in-flight coroutine, cached across calls: a
    // slot holds its td for the whole run, so no gate is needed, and reuse
    // matters because per-slot scratch is ~12 B/slot/td (visited bits +
    // exact-dist + neighbor offsets — ~1 GiB per td at 90M slots). Concurrent
    // search_pipelined calls serialize on the cache mutex by design. These
    // tds never issue libaio reads (pure reactor path), so they cost no
    // fs.aio-max-nr quota.
    std::lock_guard<std::mutex> pipeline_guard(pipeline_tds_mutex_);

    std::atomic<uint32_t> next{0};
    std::mutex error_mutex;
    std::exception_ptr first_error;

    coro::thread_pool pool{{.thread_count = workers,
                            .on_thread_start_functor = nullptr,
                            .on_thread_stop_functor = nullptr}};
    ensure_pipeline_tds_locked(inflight, pool);

    auto drive = [&](ThreadData *td) -> coro::task<void> {
      co_await pool.schedule();
      for (;;) {
        const uint32_t qi = next.fetch_add(1, std::memory_order_relaxed);
        if (qi >= n_queries) {
          break;
        }
        try {
          const auto t0 = std::chrono::steady_clock::now();
          const SearchSnapshot snapshot = make_search_snapshot();
          td->resize_slot_capacity(snapshot.max_slot_id);
          SearchContext ctx;
          ctx.reader = reader_.get();
          ctx.geom = &geom_;
          ctx.cache = &cache_;
          ctx.pq = use_pq ? &pq_ : nullptr;
          ctx.medoid = medoid_;
          ctx.num_points = snapshot.max_slot_id;
          if (snapshot.has_tombstone) {
            ctx.tombstone = &snapshot.tombstone;
          }
          ctx.page_io = search_page_io();

          SearchParams sp;
          sp.search_list_size = params.search_list_size;
          sp.beam_width = beam_width_;
          sp.use_pq = use_pq;
          sp.rerank = false;
          sp.deterministic = params.deterministic;

          SearchStats *stats = per_query_stats != nullptr ? &per_query_stats[qi] : nullptr;
          const float *query = queries + static_cast<uint64_t>(qi) * dim_;
          // No pq_mutex_ across the coroutine (a shared_mutex may not be
          // released on another thread) — safe by the dark-slot protocol, see
          // run_update_search_async.
          std::vector<std::pair<uint32_t, float>> results;
          if (use_pq) {
            results = co_await pq_beam_search_async(ctx,
                                                    query,
                                                    ask,
                                                    sp,
                                                    *td,
                                                    stats,
                                                    *update_reactor_,
                                                    pool,
                                                    fd);
          } else {
            results = co_await disk_greedy_search_async(ctx,
                                                        query,
                                                        ask,
                                                        sp,
                                                        *td,
                                                        stats,
                                                        *update_reactor_,
                                                        pool,
                                                        fd);
          }

          uint64_t *labels_row = out_labels + static_cast<uint64_t>(qi) * top_k;
          float *dist_row = out_distances + static_cast<uint64_t>(qi) * top_k;
          const uint32_t count = std::min<uint32_t>(static_cast<uint32_t>(results.size()), top_k);
          {
            std::shared_lock<std::shared_mutex> state_lock(update_mutex_);
            for (uint32_t i = 0; i < count; ++i) {
              const uint32_t id = results[i].first;
              labels_row[i] = id < labels_.size() ? labels_[id] : kNoLabel;
              dist_row[i] = results[i].second;
            }
          }
          for (uint32_t i = count; i < top_k; ++i) {
            labels_row[i] = kNoLabel;
            dist_row[i] = std::numeric_limits<float>::max();
          }
          if (per_query_us != nullptr) {
            per_query_us[qi] =
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
                    .count();
          }
        } catch (...) {
          {
            std::lock_guard<std::mutex> guard(error_mutex);
            if (!first_error) {
              first_error = std::current_exception();
            }
          }
          next.store(n_queries, std::memory_order_relaxed);  // drain remaining work
          break;
        }
      }
      co_return;
    };

    std::vector<coro::task<void>> tasks;
    tasks.reserve(inflight);
    for (uint32_t t = 0; t < inflight; ++t) {
      tasks.emplace_back(drive(pipeline_tds_[t].get()));
    }
    try {
      coro::sync_wait(coro::when_all(std::move(tasks)));
    } catch (...) {
      pool.shutdown();
      throw;
    }
    pool.shutdown();
    if (first_error) {
      std::rethrow_exception(first_error);
    }
  }
#endif  // __linux__

  // --------------------------------------------------------------- accessors
  [[nodiscard]] uint64_t size() const { return live_count(); }  // live (non-tombstoned) vectors
  [[nodiscard]] uint64_t dim() const { return dim_; }
  [[nodiscard]] bool has_pq() const { return has_pq_; }
  [[nodiscard]] uint32_t medoid() const { return medoid_; }
  [[nodiscard]] bool updatable() const { return updatable_; }
  [[nodiscard]] uint64_t live_count() const {
    std::shared_lock<std::shared_mutex> lock(update_mutex_);
    return live_count_;
  }
  [[nodiscard]] uint64_t max_slot_id() const {
    std::shared_lock<std::shared_mutex> lock(update_mutex_);
    return max_slot_id_;
  }
  [[nodiscard]] uint64_t tombstone_count() const {
    std::shared_lock<std::shared_mutex> lock(update_mutex_);
    return slot_alloc_.tombstone_count();
  }
  [[nodiscard]] uint64_t free_slot_count() const {
    std::shared_lock<std::shared_mutex> lock(update_mutex_);
    return slot_alloc_.free_count();
  }
  [[nodiscard]] uint64_t safety_net_fire_count() const {
    std::shared_lock<std::shared_mutex> lock(update_mutex_);
    return safety_net_fires_;
  }
  [[nodiscard]] bool is_deleted(uint32_t id) const {
    std::shared_lock<std::shared_mutex> lock(update_mutex_);
    return slot_alloc_.is_deleted(id);
  }
  [[nodiscard]] bool contains_label(uint64_t label) const {
    std::shared_lock<std::shared_mutex> lock(update_mutex_);
    return label_to_slot_.contains(label);
  }

  /// Sentinel label for padded (missing) result slots.
  static constexpr uint64_t kNoLabel = std::numeric_limits<uint64_t>::max();

  // ------------------------------------------------------- in-place updates
  /// Insert a vector: NN search -> select top-degree neighbors (Yi rule) ->
  /// alloc slot -> write -> publish -> reconnect selected neighbors. Returns
  /// the allocated internal slot id. Dirty pages are NOT flushed here (Yi
  /// defers write-back out of the update path); call flush() to persist.
  uint32_t insert(const float *query, uint64_t label) {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::insert: index not loaded in updatable mode");
    }
    if (query == nullptr) {
      throw std::invalid_argument("DiskANNIndex::insert: null query");
    }
    std::lock_guard<std::mutex> update_guard(update_serial_mutex_);
    {
      std::shared_lock<std::shared_mutex> state_lock(update_mutex_);
      validate_insert_label_unlocked(label);
    }
    page_io_->clear_cache();

    const std::vector<uint32_t> pruned = select_insert_neighbors(query);
    uint32_t slot = 0;
    {
      std::unique_lock<std::shared_mutex> state_lock(update_mutex_);
      std::lock_guard<std::mutex> slot_lock(slot_mutex_);
      slot = allocate_update_slot_unlocked(label);
    }
    encode_pq_slot(query, slot);
    write_inserted_node(slot, query, pruned);
    publish_slots(&slot, &slot + 1);
    reconnect_inserted_neighbors(pruned, slot);
    return slot;
  }

  /// Insert a batch of row-major vectors. The synchronous API is preserved, but
  /// the work is scheduled as coroutine tasks in chunks of @p batch_size.
  std::vector<uint32_t> batch_insert(const float *vectors,
                                     const uint64_t *labels,
                                     uint32_t count,
                                     uint32_t batch_size = 32) {
    validate_batch_insert_args(vectors, labels, count, batch_size, "batch_insert");
    if (count == 0) {
      return {};
    }

    std::lock_guard<std::mutex> update_guard(update_serial_mutex_);
    {
      std::shared_lock<std::shared_mutex> state_lock(update_mutex_);
      validate_insert_labels_unlocked(labels, count);
    }
    page_io_->clear_cache();

    const uint32_t workers = std::min({batch_size, count, update_insert_threads_});
    coro::thread_pool pool{{.thread_count = workers,
                            .on_thread_start_functor = nullptr,
                            .on_thread_stop_functor = nullptr}};
    try {
      std::vector<uint32_t> ids =
          batch_insert_locked_with_pool(vectors, labels, count, batch_size, pool);
      pool.shutdown();
      return ids;
    } catch (...) {
      pool.shutdown();
      throw;
    }
  }

  /// Insert a batch using a caller-owned coroutine pool. This keeps benchmarked
  /// mixed workloads on one shared worker queue while preserving the blocking
  /// public API.
  std::vector<uint32_t> batch_insert_with_pool(const float *vectors,
                                               const uint64_t *labels,
                                               uint32_t count,
                                               uint32_t batch_size,
                                               coro::thread_pool &pool) {
    validate_batch_insert_args(vectors, labels, count, batch_size, "batch_insert_with_pool");
    if (count == 0) {
      return {};
    }

    std::lock_guard<std::mutex> update_guard(update_serial_mutex_);
    {
      std::shared_lock<std::shared_mutex> state_lock(update_mutex_);
      validate_insert_labels_unlocked(labels, count);
    }
    page_io_->clear_cache();
    return batch_insert_locked_with_pool(vectors, labels, count, batch_size, pool);
  }

  /// Lazy-delete: cache old neighbors for two-hop, tombstone + free the slot.
  /// Reconnect is deferred to the next insert or the safety net.
  void remove(uint32_t internal_id) {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::remove: index not loaded in updatable mode");
    }
    std::lock_guard<std::mutex> update_guard(update_serial_mutex_);
    remove_internal_locked(internal_id);
  }

  /// Lazy-delete by external label.
  void remove_by_label(uint64_t label) {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::remove_by_label: index not loaded in updatable mode");
    }
    std::lock_guard<std::mutex> update_guard(update_serial_mutex_);
    uint32_t internal_id = 0;
    {
      std::shared_lock<std::shared_mutex> state_lock(update_mutex_);
      internal_id = lookup_label_unlocked(label, "remove_by_label");
    }
    remove_internal_locked(internal_id);
  }

  /// Lazy-delete a batch using a caller-owned coroutine pool. This mirrors
  /// batch_remove() but avoids creating an update-private worker pool.
  void batch_remove_with_pool(const uint32_t *internal_ids,
                              uint32_t count,
                              coro::thread_pool &pool) {
    validate_batch_remove_args(internal_ids, count, "batch_remove_with_pool");
    if (count == 0) {
      return;
    }
    std::lock_guard<std::mutex> update_guard(update_serial_mutex_);
    batch_remove_internal_locked(internal_ids, count, &pool);
  }

  /// Lazy-delete a batch of internal ids using the same semantics as remove().
  /// The batch form amortizes update locking and cache setup over Yi-style
  /// delete rounds.
  void batch_remove(const uint32_t *internal_ids, uint32_t count) {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::batch_remove: index not loaded in updatable mode");
    }
    if (count == 0) {
      return;
    }
    if (internal_ids == nullptr) {
      throw std::invalid_argument("DiskANNIndex::batch_remove: null ids");
    }
    std::lock_guard<std::mutex> update_guard(update_serial_mutex_);
    batch_remove_internal_locked(internal_ids, count, nullptr);
  }

  /// Lazy-delete a batch of external labels using the same semantics as remove_by_label().
  void batch_remove_by_labels(const uint64_t *labels, uint32_t count) {
    if (!updatable_) {
      throw std::runtime_error(
          "DiskANNIndex::batch_remove_by_labels: index not loaded in updatable mode");
    }
    if (count == 0) {
      return;
    }
    if (labels == nullptr) {
      throw std::invalid_argument("DiskANNIndex::batch_remove_by_labels: null labels");
    }
    std::lock_guard<std::mutex> update_guard(update_serial_mutex_);
    std::vector<uint32_t> internal_ids;
    internal_ids.reserve(count);
    {
      std::shared_lock<std::shared_mutex> state_lock(update_mutex_);
      std::unordered_set<uint64_t> seen;
      seen.reserve(count);
      for (uint32_t i = 0; i < count; ++i) {
        if (!seen.insert(labels[i]).second) {
          throw std::invalid_argument(
              "DiskANNIndex::batch_remove_by_labels: duplicate external label");
        }
        internal_ids.push_back(lookup_label_unlocked(labels[i], "batch_remove_by_labels"));
      }
    }
    batch_remove_internal_locked(internal_ids.data(), count, nullptr);
  }

  /**
   * @brief Reconnect @p node_id's neighbor list in place (Yi's connect-task).
   * @throws std::invalid_argument if @p node_id is out of range.
   */
  void update_node(uint32_t node_id) {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::update_node: index not loaded in updatable mode");
    }
    std::lock_guard<std::mutex> update_guard(update_serial_mutex_);
    {
      std::shared_lock<std::shared_mutex> state_lock(update_mutex_);
      if (node_id >= max_slot_id_) {
        throw std::invalid_argument("DiskANNIndex::update_node: node_id out of range");
      }
      if (slot_alloc_.is_deleted(node_id)) {
        throw std::invalid_argument("DiskANNIndex::update_node: node_id is deleted");
      }
    }
    page_io_->clear_cache();
    update_node_impl(node_id);
    page_io_->flush_dirty_pages();
  }

  /// Write dirty pages and drop disk-backed overrides — a lightweight
  /// durability point (the analog of Yi's round-boundary writeback_remaining).
  /// flush() remains the full checkpoint (meta + PQ + cache + slots).
  void flush_pages() {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::flush_pages: index not loaded in updatable mode");
    }
    std::lock_guard<std::mutex> update_guard(update_serial_mutex_);
    std::shared_lock<std::shared_mutex> state_lock(update_mutex_);
    page_io_->flush_dirty_pages();
    cache_.drop_disk_backed_overrides();
  }

  /// Persist meta + ids + slot allocator state to disk.
  void flush() {
    if (!updatable_) {
      throw std::runtime_error("DiskANNIndex::flush: index not loaded in updatable mode");
    }
    std::lock_guard<std::mutex> update_guard(update_serial_mutex_);
    std::shared_lock<std::shared_mutex> state_lock(update_mutex_);
    page_io_->flush_dirty_pages();
    // Disk is current now; overrides for nodes outside the BFS hot cache can
    // fall back to the (fresh) disk read path.
    cache_.drop_disk_backed_overrides();
    MetaHeader m;
    m.num_points = max_slot_id_;
    m.dim = dim_;
    m.max_degree = max_degree_;
    m.medoid = medoid_;
    m.has_pq = has_pq_ ? 1 : 0;
    m.pq_n_chunks = pq_n_chunks_;
    m.node_len = geom_.node_len;
    m.nodes_per_sector = geom_.nodes_per_sector;
    m.max_slot_id = max_slot_id_;
    m.live_count = live_count_;
    if (has_pq_) {
      if (pq_.num_points() != max_slot_id_) {
        throw std::runtime_error("DiskANNIndex::flush: PQ code count does not match max_slot_id");
      }
      pq_.save(path(index_dir_, "pq_pivots.bin"), path(index_dir_, "pq_compressed.bin"));
    }
    write_meta(path(index_dir_, "meta.bin"), m);
    write_ids(path(index_dir_, "ids.bin"), labels_.data(), labels_.size());
    cache_.save(path(index_dir_, "cache_ids.bin"), path(index_dir_, "cache_nodes.bin"));
    slot_alloc_.save(path(index_dir_, "slots.bin"));
  }

 private:
  static constexpr uint32_t kNoSelf = std::numeric_limits<uint32_t>::max();
  static constexpr size_t kUpdateNodeLockStripes = 4096;
  /// Yi's co_update pulls at most 5 live two-hop candidates per deleted
  /// neighbor; without the cap the pool inflates to the deleted node's whole
  /// ex-neighborhood and reconnect cost explodes.
  static constexpr uint32_t kTwoHopBypassPerDeleted = 5;
  /// Yi heap-caps the reconnect pool at build_k = degree + 32 before pruning.
  static constexpr uint32_t kReconnectPoolSlack = 32;

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

  struct SearchSnapshot {
    uint64_t max_slot_id = 0;
    TombstoneSnapshot tombstone;
    bool has_tombstone = false;
  };

  static std::string path(const std::string &dir, const char *name) {
    return (std::filesystem::path(dir) / name).string();
  }

  SearchSnapshot make_search_snapshot() const {
    SearchSnapshot snapshot;
    std::shared_lock<std::shared_mutex> lock(update_mutex_);
    snapshot.max_slot_id = max_slot_id_;
    // All tombstone mutators run under the exclusive side of update_mutex_
    // (verified in the A3 audit), so count() is stable under our shared lock:
    // zero set bits means the snapshot would be all-live — skip the copy. The
    // bitmap's capacity never shrinks, so after the first update round every
    // query would otherwise pay a full-capacity copy even with no tombstones.
    if (updatable_ && slot_alloc_.tombstone().count() > 0) {
      slot_alloc_.tombstone().snapshot_into(snapshot.tombstone);
      snapshot.has_tombstone = true;
    }
    return snapshot;
  }

  // ---- in-place update internals (update APIs are serialized; metadata writes
  //      take update_mutex_ only for the narrow mutation/snapshot windows) ----

  /// Wire up the update subsystem (page IO, slot allocator, context, config).
  void init_updatable(const std::string &index_dir, const DiskANNLoadParams &params) {
    index_dir_ = index_dir;
    update_alpha_ = params.update_alpha;
    update_search_l_ = params.update_search_l != 0 ? params.update_search_l : max_degree_ + 32;
    update_rerank_ = params.update_rerank;
    update_insert_prune_ = params.update_insert_prune;
    safety_net_ratio_ = params.safety_net_ratio;
    safety_net_ops_ = params.safety_net_ops;
    if (params.update_reconnect_threads == 0) {
      throw std::invalid_argument("DiskANNIndex::load: update_reconnect_threads must be > 0");
    }
    if (params.update_insert_threads == 0) {
      throw std::invalid_argument("DiskANNIndex::load: update_insert_threads must be > 0");
    }
    update_insert_threads_ = params.update_insert_threads;
    update_reconnect_threads_ = params.update_reconnect_threads;
    safety_net_fires_ = 0;
    ops_since_last_insert_ = 0;
    update_ctx_.clear();

    page_io_ = std::make_unique<DiskPageIO>(path(index_dir, "diskann.index"),
                                            geom_,
                                            params.page_cache_capacity);
    update_reactor_.reset();
    if (params.update_io != DiskANNUpdateIO::kBlocking) {
      if (alaya::UringReactor::is_available()) {
        update_reactor_ = std::make_unique<alaya::UringReactor>();
        page_io_->set_reactor(update_reactor_.get());
      } else if (params.update_io == DiskANNUpdateIO::kUring) {
        throw std::runtime_error(
            "DiskANNIndex::load: update_io=kUring but io_uring is unavailable");
      }
    }
    page_io_->set_fallback_threads(params.update_insert_threads);

#if defined(__linux__)
    // Reactor-mode No-PQ update searches suspend on I/O waves while holding a
    // ThreadData, so they draw from a dedicated suspending gate (AsyncGate)
    // instead of the thread-blocking pool: with more queued insert coroutines
    // than ThreadData objects, blocking acquisition would park every pool thread and
    // leave no thread to run the resume that releases one (deadlock). The gate
    // size is the in-flight-search bound — suspended searches burn memory, not
    // CPU, so it deliberately exceeds the thread count (workers vs tasklets).
    // These tds never issue libaio reads — no reader thread registration, so
    // they cost no fs.aio-max-nr quota.
    update_search_td_gate_.clear();
    update_search_async_ = false;
    if (update_reactor_ && reader_ && reader_->get_fd() >= 0) {
      const uint32_t gate_tds = params.update_search_concurrency != 0
                                    ? params.update_search_concurrency
                                    : 4 * params.update_insert_threads;
      const size_t base = thread_data_storage_.size();
      thread_data_storage_.resize(base + gate_tds);
      for (uint32_t t = 0; t < gate_tds; ++t) {
        auto td = std::make_unique<ThreadData>();
        td->alloc_scratch(scratch_config_);
        td->ensure_wave_scratch(static_cast<uint64_t>(std::max<uint32_t>(1, max_degree_)) *
                                geom_.page_size);
        update_search_td_gate_.add(td.get());
        thread_data_storage_[base + t] = std::move(td);
      }
      update_search_async_ = true;
    }
#endif

    const std::string slots_path = path(index_dir, "slots.bin");
    if (std::filesystem::exists(slots_path)) {
      slot_alloc_.load(slots_path);  // restore free list + next id + tombstones
      max_slot_id_ = std::max<uint64_t>(max_slot_id_, slot_alloc_.next_fresh_id());
      resize_thread_data_slot_capacity();
    } else {
      slot_alloc_.reset(static_cast<uint32_t>(max_slot_id_));
    }
    rebuild_label_lookup_unlocked();
    updatable_ = true;
  }

  /// The unified-pool handle searches should use, or nullptr. Non-null only
  /// for updatable indices with a nonzero page cache and the load param on —
  /// then every search peeks the shard cache and fills it with read pages,
  /// which is exactly the view Yi's unified buffer pool gives its tasklets.
  DiskPageIO *search_page_io() const {
    return (search_page_cache_ && page_io_ != nullptr && page_io_->page_cache_enabled())
               ? page_io_.get()
               : nullptr;
  }

  /// Tombstone-aware update search returning up to @p l (id, dist) candidates.
  /// PQ indexes use PQ beam distances; No-PQ indexes use exact-L2 disk greedy.
  std::vector<std::pair<uint32_t, float>> run_update_search(const float *query, uint32_t l) {
    const SearchSnapshot snapshot = make_search_snapshot();
    std::shared_lock<std::shared_mutex> pq_lock;
    if (has_pq_) {
      pq_lock = std::shared_lock<std::shared_mutex>(pq_mutex_);
    }
    ThreadData *td = acquire();
    std::vector<std::pair<uint32_t, float>> results;
    try {
      td->resize_slot_capacity(snapshot.max_slot_id);
      SearchContext ctx;
      ctx.reader = reader_.get();
      ctx.geom = &geom_;
      ctx.cache = &cache_;
      ctx.pq = has_pq_ ? &pq_ : nullptr;
      ctx.medoid = medoid_;
      ctx.num_points = snapshot.max_slot_id;
      ctx.tombstone = &snapshot.tombstone;
      ctx.page_io = search_page_io();
      SearchParams sp;
      sp.search_list_size = l;
      sp.beam_width = beam_width_;
      sp.use_pq = has_pq_;
      sp.rerank = false;
      sp.deterministic = false;
      results = cached_beam_search(ctx, query, l, sp, *td, nullptr);
    } catch (...) {
      release(td);
      throw;
    }
    release(td);
    return results;
  }

  /// run_update_search with the disk reads awaitable (No-PQ + reactor only):
  /// the greedy search suspends on reactor waves instead of parking the pool
  /// thread in io_getevents. Falls back to the sync search otherwise.
  coro::task<std::vector<std::pair<uint32_t, float>>>
  run_update_search_async(const float *query, uint32_t l, coro::thread_pool &pool) {
#if defined(__linux__)
    if (update_search_async_) {
      const SearchSnapshot snapshot = make_search_snapshot();
      auto mark = std::chrono::steady_clock::now();
      // Suspending acquisition — see the gate comment in init_updatable().
      ThreadData *td = co_await update_search_td_gate_.acquire(pool);
      st_gate_us_.fetch_add(stage_us_since(mark), std::memory_order_relaxed);
      std::vector<std::pair<uint32_t, float>> results;
      try {
        td->resize_slot_capacity(snapshot.max_slot_id);
        SearchContext ctx;
        ctx.reader = reader_.get();
        ctx.geom = &geom_;
        ctx.cache = &cache_;
        ctx.pq = has_pq_ ? &pq_ : nullptr;
        ctx.medoid = medoid_;
        ctx.num_points = snapshot.max_slot_id;
        ctx.tombstone = &snapshot.tombstone;
        ctx.page_io = search_page_io();
        SearchParams sp;
        sp.search_list_size = l;
        sp.beam_width = beam_width_;
        sp.use_pq = has_pq_;
        sp.rerank = false;
        // No pq_mutex_ here (a shared_mutex cannot be released on a different
        // thread than locked it): safe because concurrent encode_pq_slot only
        // writes codes of still-dark slots, all masked by snapshot.tombstone —
        // see the note on pq_beam_search_async.
        if (has_pq_) {
          results = co_await pq_beam_search_async(ctx,
                                                  query,
                                                  l,
                                                  sp,
                                                  *td,
                                                  nullptr,
                                                  *update_reactor_,
                                                  pool,
                                                  reader_->get_fd());
        } else {
          results = co_await disk_greedy_search_async(ctx,
                                                      query,
                                                      l,
                                                      sp,
                                                      *td,
                                                      nullptr,
                                                      *update_reactor_,
                                                      pool,
                                                      reader_->get_fd());
        }
        st_greedy_us_.fetch_add(stage_us_since(mark), std::memory_order_relaxed);
      } catch (...) {
        update_search_td_gate_.release(td);
        throw;
      }
      update_search_td_gate_.release(td);
      co_return results;
    }
#endif
    co_return run_update_search(query, l);
  }

 public:
  /// Wall-microseconds by update stage, aggregated across coroutines since the
  /// last take (wall, not CPU: a stage that yields through the pool queue books
  /// its scheduling latency here — that is the point of measuring it).
  struct UpdateStageStats {
    uint64_t gate_us = 0;         ///< waiting for a search ThreadData
    uint64_t greedy_us = 0;       ///< async greedy search proper
    uint64_t search_us = 0;       ///< whole selection stage (incl. gate+greedy)
    uint64_t alloc_us = 0;        ///< slot alloc + PQ encode
    uint64_t prefetch_us = 0;     ///< slot page warm wave
    uint64_t write_us = 0;        ///< write_inserted_node + publish + staging
    uint64_t reconnect_us = 0;    ///< when_all over the insert's reconnects
    uint64_t rc_prefetch_us = 0;  ///< reconnect input warm waves
    uint64_t rc_lock_us = 0;      ///< node mutex acquisition wait
    uint64_t rc_impl_us = 0;      ///< sync update_node_impl body
    uint64_t inserts = 0;
    uint64_t reconnects = 0;
  };

  /// Snapshot-and-reset the stage stats (benchmark instrumentation).
  UpdateStageStats take_update_stage_stats() {
    UpdateStageStats out;
    out.gate_us = st_gate_us_.exchange(0, std::memory_order_acq_rel);
    out.greedy_us = st_greedy_us_.exchange(0, std::memory_order_acq_rel);
    out.search_us = st_search_us_.exchange(0, std::memory_order_acq_rel);
    out.alloc_us = st_alloc_us_.exchange(0, std::memory_order_acq_rel);
    out.prefetch_us = st_prefetch_us_.exchange(0, std::memory_order_acq_rel);
    out.write_us = st_write_us_.exchange(0, std::memory_order_acq_rel);
    out.reconnect_us = st_reconnect_us_.exchange(0, std::memory_order_acq_rel);
    out.rc_prefetch_us = st_rc_prefetch_us_.exchange(0, std::memory_order_acq_rel);
    out.rc_lock_us = st_rc_lock_us_.exchange(0, std::memory_order_acq_rel);
    out.rc_impl_us = st_rc_impl_us_.exchange(0, std::memory_order_acq_rel);
    out.inserts = st_inserts_.exchange(0, std::memory_order_acq_rel);
    out.reconnects = st_reconnects_.exchange(0, std::memory_order_acq_rel);
    return out;
  }

 private:
  /// Cached L2 distance between two nodes via the page IO coords cache.
  float cached_l2(uint32_t a, uint32_t b) {
    const auto l2 = alaya::simd::get_l2_sqr_func();
    const std::vector<float> ca = page_io_->read_coords_cached(a);
    const std::vector<float> cb = page_io_->read_coords_cached(b);
    return l2(ca.data(), cb.data(), dim_);
  }

  template <typename DistFn>
  std::vector<uint32_t> prune_insert_pool(std::vector<alaya::vamana::Neighbor> &pool,
                                          DistFn dist_fn) {
    std::vector<uint32_t> pruned;
    std::vector<float> occlude_scratch;
    const uint32_t maxc = std::max<uint32_t>(max_degree_, static_cast<uint32_t>(pool.size()));
    alaya::vamana::prune_neighbors(kNoSelf,
                                   pool,
                                   update_alpha_,
                                   max_degree_,
                                   maxc,
                                   pruned,
                                   occlude_scratch,
                                   dist_fn);
    return pruned;
  }

  /// Exact L2 between @p query and a stored node's coords. Served from the
  /// NodeCache record when available (shared-lock memory read — the analog of
  /// Yi's buffer-pool get_raw_dist); falls back to the update page cache.
  float exact_query_distance(const float *query, uint32_t id) {
    const auto l2 = alaya::simd::get_l2_sqr_func();
    const NodeCache::Lookup rec = cache_.lookup_record(id);
    if (rec) {
      return l2(query, reinterpret_cast<const float *>(rec.get()), dim_);
    }
    const std::vector<float> coords = page_io_->read_coords_cached(id);
    return l2(query, coords.data(), dim_);
  }

  std::vector<uint32_t> select_insert_neighbors_exact(
      const float *query,
      std::vector<std::pair<uint32_t, float>> cand) {
    if (update_insert_prune_) {
      std::vector<alaya::vamana::Neighbor> pool;
      pool.reserve(cand.size());
      for (const auto &c : cand) {
        pool.emplace_back(c.first, c.second);
      }
      auto dist_fn = [this](uint32_t a, uint32_t b) -> float {
        return cached_l2(a, b);
      };
      return prune_insert_pool(pool, dist_fn);
    }
    // Yi rule: link the top max_degree_ nearest, no alpha-prune. The No-PQ
    // search already returns exact distances in ascending order, so the
    // rerank option is a no-op here.
    if (cand.size() > max_degree_) {
      cand.resize(max_degree_);
    }
    std::vector<uint32_t> out;
    out.reserve(cand.size());
    for (const auto &c : cand) {
      out.push_back(c.first);
    }
    return out;
  }

  std::vector<uint32_t> select_insert_neighbors_pq(const float *query,
                                                   std::vector<std::pair<uint32_t, float>> cand) {
    std::vector<uint8_t> query_code(pq_n_chunks_);
    pq_.encode_to_code(query, query_code.data());
    if (update_insert_prune_) {
      std::vector<alaya::vamana::Neighbor> pool;
      pool.reserve(cand.size());
      for (const auto &c : cand) {
        pool.emplace_back(c.first, pq_.pq_symmetric_distance(query_code.data(), c.first));
      }
      auto dist_fn = [this](uint32_t a, uint32_t b) -> float {
        return pq_.pq_symmetric_distance(a, b);
      };
      return prune_insert_pool(pool, dist_fn);
    }
    // Yi rule (co_insert:756-775): optionally re-rank the search pool by exact
    // L2 (Yi's _rerank_flag inside the insert search), keep the top
    // max_degree_, re-score the survivors with symmetric PQ and link in that
    // order. The alpha-prune branch is dead code in Yi (top_k == degree).
    if (update_rerank_) {
      for (auto &c : cand) {
        c.second = exact_query_distance(query, c.first);
      }
      std::sort(cand.begin(), cand.end(), [](const auto &a, const auto &b) {
        return a.second < b.second || (a.second == b.second && a.first < b.first);
      });
    }
    if (cand.size() > max_degree_) {
      cand.resize(max_degree_);
    }
    std::vector<alaya::vamana::Neighbor> pool;
    pool.reserve(cand.size());
    for (const auto &c : cand) {
      pool.emplace_back(c.first, pq_.pq_symmetric_distance(query_code.data(), c.first));
    }
    std::sort(pool.begin(), pool.end());
    std::vector<uint32_t> out;
    out.reserve(pool.size());
    for (const auto &nb : pool) {
      out.push_back(nb.id);
    }
    return out;
  }

  std::vector<uint32_t> select_insert_neighbors_from(const float *query,
                                                     std::vector<std::pair<uint32_t, float>> cand) {
    if (has_pq_) {
      return select_insert_neighbors_pq(query, std::move(cand));
    }
    return select_insert_neighbors_exact(query, std::move(cand));
  }

  /// Which insert-selection variants do exact-coords distance work (and thus
  /// profit from a coords prefetch wave)?
  [[nodiscard]] bool insert_selection_needs_coords() const {
    if (has_pq_) {
      // The rerank loop calls exact_query_distance per candidate.
      return !update_insert_prune_ && update_rerank_;
    }
    // No-PQ prune scores with cached_l2; the no-prune path reuses the exact
    // search distances and reads nothing.
    return update_insert_prune_;
  }

  /// select_insert_neighbors with the disk work made awaitable: the candidate
  /// coords the selection will score are wave-prefetched through the reactor
  /// (one suspension, all misses in flight together) before the sync selection
  /// logic runs against warm caches.
  coro::task<std::vector<uint32_t>> select_insert_neighbors_async(const float *query,
                                                                  coro::thread_pool &pool) {
    auto cand = co_await run_update_search_async(query, update_search_l_, pool);
    if (page_io_->reactor_enabled() && insert_selection_needs_coords() && !cand.empty()) {
      std::vector<uint32_t> want;
      want.reserve(cand.size());
      for (const auto &c : cand) {
        // exact_query_distance (PQ rerank) is served by the NodeCache first;
        // cached_l2 (No-PQ prune) always goes through the coords cache.
        if (!has_pq_ || !cache_.lookup_record(c.first)) {
          want.push_back(c.first);
        }
      }
      if (!want.empty()) {
        co_await page_io_->prefetch_coords(want.data(), want.size(), pool);
      }
    }
    co_return select_insert_neighbors_from(query, std::move(cand));
  }

  std::vector<uint32_t> select_insert_neighbors(const float *query) {
    return select_insert_neighbors_from(query, run_update_search(query, update_search_l_));
  }

  uint32_t allocate_update_slot_unlocked(uint64_t label) {
    const uint32_t slot = slot_alloc_.alloc();
    update_ctx_.forget_slot(slot);
    max_slot_id_ = std::max<uint64_t>(max_slot_id_, slot_alloc_.next_fresh_id());
    set_label(slot, label);
    ++live_count_;
    ops_since_last_insert_ = 0;
    return slot;
  }

  static void validate_unique_external_labels(const uint64_t *labels,
                                              uint64_t count,
                                              const char *method) {
    std::unordered_set<uint64_t> seen;
    seen.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
      if (labels[i] == kNoLabel) {
        throw std::invalid_argument(std::string("DiskANNIndex::") + method +
                                    ": external label is reserved");
      }
      if (!seen.insert(labels[i]).second) {
        throw std::invalid_argument(std::string("DiskANNIndex::") + method +
                                    ": duplicate external label");
      }
    }
  }

  void validate_insert_label_unlocked(uint64_t label) const {
    if (label == kNoLabel) {
      throw std::invalid_argument("DiskANNIndex::insert: external label is reserved");
    }
    if (label_to_slot_.contains(label)) {
      throw std::invalid_argument("DiskANNIndex::insert: duplicate external label");
    }
  }

  void validate_insert_labels_unlocked(const uint64_t *labels, uint32_t count) const {
    validate_unique_external_labels(labels, count, "batch_insert");
    for (uint32_t i = 0; i < count; ++i) {
      if (label_to_slot_.contains(labels[i])) {
        throw std::invalid_argument("DiskANNIndex::batch_insert: duplicate external label");
      }
    }
  }

  uint32_t lookup_label_unlocked(uint64_t label, const char *method) const {
    const auto it = label_to_slot_.find(label);
    if (it == label_to_slot_.end()) {
      throw std::out_of_range(std::string("DiskANNIndex::") + method +
                              ": external label not found: " + std::to_string(label));
    }
    return it->second;
  }

  void rebuild_label_lookup_unlocked() {
    label_to_slot_.clear();
    label_to_slot_.reserve(static_cast<size_t>(live_count_));
    for (uint32_t slot = 0; slot < labels_.size(); ++slot) {
      if (slot_alloc_.is_deleted(slot)) {
        continue;
      }
      const uint64_t label = labels_[slot];
      if (label == kNoLabel) {
        throw std::runtime_error("DiskANNIndex::load: live slot has reserved external label");
      }
      if (!label_to_slot_.emplace(label, slot).second) {
        throw std::runtime_error("DiskANNIndex::load: duplicate live external label: " +
                                 std::to_string(label));
      }
    }
  }

  void encode_pq_slot(const float *query, uint32_t slot) {
    if (!has_pq_) {
      return;
    }
    std::unique_lock<std::shared_mutex> pq_lock(pq_mutex_);
    pq_.encode_one(query, slot);
  }

  void validate_batch_insert_args(const float *vectors,
                                  const uint64_t *labels,
                                  uint32_t count,
                                  uint32_t batch_size,
                                  const char *method) const {
    if (!updatable_) {
      throw std::runtime_error(std::string("DiskANNIndex::") + method +
                               ": index not loaded in updatable mode");
    }
    if (count == 0) {
      return;
    }
    if (vectors == nullptr || labels == nullptr) {
      throw std::invalid_argument(std::string("DiskANNIndex::") + method + ": null vectors/labels");
    }
    if (batch_size == 0) {
      throw std::invalid_argument(std::string("DiskANNIndex::") + method +
                                  ": batch_size must be > 0");
    }
  }

  void validate_batch_remove_args(const uint32_t *internal_ids,
                                  uint32_t count,
                                  const char *method) const {
    if (!updatable_) {
      throw std::runtime_error(std::string("DiskANNIndex::") + method +
                               ": index not loaded in updatable mode");
    }
    if (count != 0 && internal_ids == nullptr) {
      throw std::invalid_argument(std::string("DiskANNIndex::") + method + ": null ids");
    }
  }

  void validate_removable_slot(uint32_t internal_id) const {
    if (internal_id >= max_slot_id_) {
      throw std::invalid_argument("DiskANNIndex::remove: internal_id out of range");
    }
    if (slot_alloc_.is_deleted(internal_id)) {
      throw std::invalid_argument("DiskANNIndex::remove: node already deleted");
    }
  }

  void validate_remove_batch(const uint32_t *internal_ids, uint32_t count) const {
    std::unordered_set<uint32_t> seen;
    seen.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      validate_removable_slot(internal_ids[i]);
      if (!seen.insert(internal_ids[i]).second) {
        throw std::invalid_argument("DiskANNIndex::batch_remove: duplicate internal_id");
      }
    }
  }

  void remove_unlocked(uint32_t internal_id) {
    const DiskPageIO::NodeData nd = page_io_->read_node(internal_id);
    std::unique_lock<std::shared_mutex> state_lock(update_mutex_);
    remove_unlocked_with_neighbors(internal_id, nd.nbrs);
  }

  void remove_internal_locked(uint32_t internal_id) {
    {
      std::shared_lock<std::shared_mutex> state_lock(update_mutex_);
      validate_removable_slot(internal_id);
    }
    page_io_->clear_cache();
    remove_unlocked(internal_id);
    if (maybe_safety_net_reconnect()) {
      page_io_->flush_dirty_pages();
    }
  }

  void batch_remove_internal_locked(const uint32_t *internal_ids,
                                    uint32_t count,
                                    coro::thread_pool *pool) {
    {
      std::shared_lock<std::shared_mutex> state_lock(update_mutex_);
      validate_remove_batch(internal_ids, count);
    }
    page_io_->clear_cache();
    std::vector<std::vector<uint32_t>> old_neighbors =
        read_delete_neighbors(internal_ids, count, pool);
    {
      std::unique_lock<std::shared_mutex> state_lock(update_mutex_);
      for (uint32_t i = 0; i < count; ++i) {
        remove_unlocked_with_neighbors(internal_ids[i], std::move(old_neighbors[i]));
      }
    }
    if (maybe_safety_net_reconnect()) {
      page_io_->flush_dirty_pages();
    }
  }

  /// Neighbor lists of a delete batch. With the reactor this is one wave of
  /// concurrent io_uring reads over the batch's unique pages (a single
  /// suspension); without it, the blocking std::thread reader.
  std::vector<std::vector<uint32_t>> read_delete_neighbors(const uint32_t *internal_ids,
                                                           uint32_t count,
                                                           coro::thread_pool *pool) {
    if (page_io_->reactor_enabled() && count > 1) {
      if (pool != nullptr) {
        return coro::sync_wait(page_io_->read_neighbors_batch_async(internal_ids, count, *pool));
      }
      coro::thread_pool wave_pool{{.thread_count = 1,
                                   .on_thread_start_functor = nullptr,
                                   .on_thread_stop_functor = nullptr}};
      auto out =
          coro::sync_wait(page_io_->read_neighbors_batch_async(internal_ids, count, wave_pool));
      wave_pool.shutdown();
      return out;
    }
    return page_io_->read_neighbors_batch_parallel(internal_ids, count, update_insert_threads_);
  }

  void remove_unlocked_with_neighbors(uint32_t internal_id, std::vector<uint32_t> old_neighbors) {
    update_ctx_.removed_node_nbrs_[internal_id] = std::move(old_neighbors);
    const uint64_t label = labels_[internal_id];
    const auto label_it = label_to_slot_.find(label);
    if (label_it != label_to_slot_.end() && label_it->second == internal_id) {
      label_to_slot_.erase(label_it);
    }
    slot_alloc_.free(internal_id);
    --live_count_;
    ++ops_since_last_insert_;
  }

  void write_inserted_node(uint32_t slot,
                           const float *query,
                           const std::vector<uint32_t> &neighbors) {
    page_io_->write_node(slot, query, static_cast<uint32_t>(neighbors.size()), neighbors.data());
    cache_.upsert_node(slot, query, static_cast<uint32_t>(neighbors.size()), neighbors.data());
  }

  /// Staged reverse edges for a batch — Yi's `_inserted_edges` analog. Each
  /// insert stages `selected neighbor -> new slot` edges, and whichever
  /// reconnect task reaches a neighbor first drains ALL edges staged for it
  /// (natural dedup across concurrent inserts, exactly like Yi's co_update).
  struct StagedEdges {
    static constexpr size_t kStripes = 64;
    std::array<std::mutex, kStripes> mutexes;
    std::array<std::unordered_map<uint32_t, std::vector<uint32_t>>, kStripes> maps;

    void add(uint32_t node_id, uint32_t slot) {
      const size_t s = node_id % kStripes;
      std::lock_guard<std::mutex> lock(mutexes[s]);
      maps[s][node_id].push_back(slot);
    }

    std::vector<uint32_t> drain(uint32_t node_id) {
      const size_t s = node_id % kStripes;
      std::lock_guard<std::mutex> lock(mutexes[s]);
      const auto it = maps[s].find(node_id);
      if (it == maps[s].end()) {
        return {};
      }
      std::vector<uint32_t> out = std::move(it->second);
      maps[s].erase(it);
      return out;
    }
  };

  /// Yi-style batch: end-to-end insert coroutines (search -> alloc -> write ->
  /// publish -> stage edges -> reconnect), one when_all barrier per chunk of
  /// batch_size. The earlier plan/reserve/write/reconnect phase pipeline put
  /// four sync_wait barriers on every 32-insert chunk, which inverted thread
  /// scaling (32 workers ran slower than 8); Yi has no intra-batch barriers.
  std::vector<uint32_t> batch_insert_locked_with_pool(const float *vectors,
                                                      const uint64_t *labels,
                                                      uint32_t count,
                                                      uint32_t batch_size,
                                                      coro::thread_pool &pool) {
    std::vector<uint32_t> ids(count, 0);
    StagedEdges staged;

    auto reconnect_one = [this, &pool, &staged](uint32_t node_id) -> coro::task<> {
      co_await pool.schedule();
      auto mark = std::chrono::steady_clock::now();
      const std::vector<uint32_t> extra = staged.drain(node_id);
      // The insert already warmed every selected node's page in ONE wave (see
      // insert_one); per-reconnect work here is No-PQ only — deriving the
      // candidate set (from the now-cached page) and waving candidate COORDS
      // in before the prune. PQ scores in memory: nothing left to warm, and
      // skipping the call avoids 1 page copy + a dead derivation per edge.
      if (!has_pq_) {
        co_await prefetch_reconnect_inputs(node_id, extra, pool);
      }
      st_rc_prefetch_us_.fetch_add(stage_us_since(mark), std::memory_order_relaxed);
      std::lock_guard<std::mutex> node_lock(update_node_mutex(node_id));
      st_rc_lock_us_.fetch_add(stage_us_since(mark), std::memory_order_relaxed);
      update_node_impl(node_id, extra);
      st_rc_impl_us_.fetch_add(stage_us_since(mark), std::memory_order_relaxed);
      st_reconnects_.fetch_add(1, std::memory_order_relaxed);
    };

    auto insert_one =
        [this, &pool, &staged, &reconnect_one, vectors, labels, &ids](uint32_t i) -> coro::task<> {
      co_await pool.schedule();
      auto mark = std::chrono::steady_clock::now();
      const float *vec = vectors + static_cast<uint64_t>(i) * dim_;
      const std::vector<uint32_t> selected = co_await select_insert_neighbors_async(vec, pool);
      st_search_us_.fetch_add(stage_us_since(mark), std::memory_order_relaxed);
      {
        std::unique_lock<std::shared_mutex> state_lock(update_mutex_);
        std::lock_guard<std::mutex> slot_lock(slot_mutex_);
        ids[i] = allocate_update_slot_unlocked(labels[i]);
      }
      encode_pq_slot(vec, ids[i]);
      st_alloc_us_.fetch_add(stage_us_since(mark), std::memory_order_relaxed);
      if (page_io_->reactor_enabled()) {
        // ONE wave warms every page this insert will RMW: the slot's own page
        // (written below) plus all selected neighbors' pages (their reconnects
        // read-modify-write them right after). One suspension instead of a
        // wave per reconnect — the per-edge waves were the dominant queue load.
        std::vector<uint32_t> warm;
        warm.reserve(selected.size() + 1);
        warm.push_back(ids[i]);
        warm.insert(warm.end(), selected.begin(), selected.end());
        co_await page_io_->prefetch_pages(warm.data(), warm.size(), pool);
      }
      st_prefetch_us_.fetch_add(stage_us_since(mark), std::memory_order_relaxed);
      write_inserted_node(ids[i], vec, selected);
      publish_slots(&ids[i], &ids[i] + 1);
      for (const uint32_t n : selected) {
        staged.add(n, ids[i]);
      }
      st_write_us_.fetch_add(stage_us_since(mark), std::memory_order_relaxed);
      std::vector<coro::task<>> reconnects;
      reconnects.reserve(selected.size());
      for (const uint32_t n : selected) {
        reconnects.emplace_back(reconnect_one(n));
      }
      co_await coro::when_all(std::move(reconnects));
      st_reconnect_us_.fetch_add(stage_us_since(mark), std::memory_order_relaxed);
      st_inserts_.fetch_add(1, std::memory_order_relaxed);
    };

    for (uint32_t off = 0; off < count; off += batch_size) {
      const uint32_t end = std::min<uint32_t>(count, off + batch_size);
      auto run = [&]() -> coro::task<> {
        std::vector<coro::task<>> tasks;
        tasks.reserve(end - off);
        for (uint32_t i = off; i < end; ++i) {
          tasks.emplace_back(insert_one(i));
        }
        co_await coro::when_all(std::move(tasks));
      };
      coro::sync_wait(run());
    }
    return ids;
  }

  /// Make written slots search-visible. Slots stay tombstoned ("dark") from
  /// allocation until their node record and PQ code exist, so a concurrent
  /// search can never surface a reused slot's stale bytes under a new label.
  void publish_slots(const uint32_t *begin, const uint32_t *end) {
    std::unique_lock<std::shared_mutex> state_lock(update_mutex_);
    for (const uint32_t *it = begin; it != end; ++it) {
      slot_alloc_.publish(*it);
      label_to_slot_[labels_[*it]] = *it;
    }
  }

  std::vector<alaya::vamana::Neighbor> score_candidates(uint32_t node_id,
                                                        const std::vector<uint32_t> &cand) {
    if (has_pq_) {
      return score_candidates_pq(node_id, cand);
    }
    const std::vector<float> self_coords = page_io_->read_coords_cached(node_id);
    const auto l2 = alaya::simd::get_l2_sqr_func();
    std::vector<alaya::vamana::Neighbor> pool;
    pool.reserve(cand.size());
    for (const uint32_t c : cand) {
      const std::vector<float> cc = page_io_->read_coords_cached(c);
      pool.emplace_back(c, l2(self_coords.data(), cc.data(), dim_));
    }
    return pool;
  }

  std::vector<alaya::vamana::Neighbor> score_candidates_pq(uint32_t node_id,
                                                           const std::vector<uint32_t> &cand) {
    std::vector<alaya::vamana::Neighbor> pool;
    pool.reserve(cand.size());
    for (const uint32_t c : cand) {
      pool.emplace_back(c, pq_.pq_symmetric_distance(node_id, c));
    }
    return pool;
  }

  template <typename DistFn>
  std::vector<uint32_t> prune_candidate_pool_with_dist(uint32_t node_id,
                                                       std::vector<alaya::vamana::Neighbor> &pool,
                                                       DistFn dist_fn) {
    std::vector<uint32_t> new_nbrs;
    std::vector<float> occlude_scratch;
    alaya::vamana::prune_neighbors(node_id,
                                   pool,
                                   update_alpha_,
                                   max_degree_,
                                   static_cast<uint32_t>(pool.size()),
                                   new_nbrs,
                                   occlude_scratch,
                                   dist_fn);
    return new_nbrs;
  }

  std::vector<uint32_t> prune_candidate_pool(uint32_t node_id,
                                             std::vector<alaya::vamana::Neighbor> &pool) {
    std::vector<uint32_t> new_nbrs;
    if (pool.size() <= max_degree_) {
      std::sort(pool.begin(), pool.end());
      for (const auto &nb : pool) {
        new_nbrs.push_back(nb.id);
      }
      return new_nbrs;
    }
    if (has_pq_) {
      auto dist_fn = [this](uint32_t a, uint32_t b) -> float {
        return pq_.pq_symmetric_distance(a, b);
      };
      return prune_candidate_pool_with_dist(node_id, pool, dist_fn);
    }
    auto dist_fn = [this](uint32_t a, uint32_t b) -> float {
      return cached_l2(a, b);
    };
    return prune_candidate_pool_with_dist(node_id, pool, dist_fn);
  }

  /// Shared reconnect backbone (Yi's co_update tail): pools at or under the
  /// degree bound are kept verbatim with no distance work; larger pools are
  /// scored, capped to degree+32 nearest (Yi's build_k heap), and alpha-pruned.
  void prune_and_write(uint32_t node_id,
                       const std::vector<uint32_t> &old_nbrs,
                       const std::vector<uint32_t> &cand) {
    std::vector<uint32_t> new_nbrs;
    if (cand.size() <= max_degree_) {
      new_nbrs = cand;
    } else {
      auto pool = score_candidates(node_id, cand);
      const uint32_t cap = max_degree_ + kReconnectPoolSlack;
      if (pool.size() > cap) {
        std::nth_element(pool.begin(), pool.begin() + cap, pool.end());
        pool.resize(cap);
      }
      new_nbrs = prune_candidate_pool(node_id, pool);
    }
    if (!same_neighbor_set(old_nbrs, new_nbrs)) {
      page_io_->write_node_neighbors(node_id,
                                     static_cast<uint32_t>(new_nbrs.size()),
                                     new_nbrs.data());
      mirror_neighbors_to_cache(node_id, new_nbrs);
    }
  }

  /// Publish a node's new neighbor list to the NodeCache override layer.
  /// Search reads uncached nodes from DISK via the O_RDONLY reader, which
  /// cannot see dirty pages in the update page cache — the override is the
  /// only thing that keeps searches coherent between flushes, so it must be
  /// installed unconditionally (NodeCache::update_neighbors silently no-ops
  /// for nodes it has no record of).
  void mirror_neighbors_to_cache(uint32_t node_id, const std::vector<uint32_t> &new_nbrs) {
    const NodeCache::Lookup rec = cache_.lookup_record(node_id);
    if (rec) {
      cache_.upsert_node(node_id,
                         reinterpret_cast<const float *>(rec.get()),
                         static_cast<uint32_t>(new_nbrs.size()),
                         new_nbrs.data());
      return;
    }
    const std::vector<float> coords = page_io_->read_coords_cached(node_id);
    cache_.upsert_node(node_id,
                       coords.data(),
                       static_cast<uint32_t>(new_nbrs.size()),
                       new_nbrs.data());
  }

  void update_node_impl(uint32_t node_id) {
    const std::vector<uint32_t> extra_edges;
    update_node_impl_locked(node_id, extra_edges);
  }

  /// Reconnect node_id's neighbor list: candidates = live old + two-hop through
  /// deleted nodes + explicit extra edges, exact-L2 ranked, alpha-RNG pruned.
  void update_node_impl(uint32_t node_id, const std::vector<uint32_t> &extra_edges) {
    const DiskPageIO::NodeData nd = page_io_->read_node(node_id);
    update_node_impl_from_snapshot(node_id, nd, extra_edges);
  }

  void update_node_impl_from_snapshot(uint32_t node_id,
                                      const DiskPageIO::NodeData &nd,
                                      const std::vector<uint32_t> &extra_edges) {
    // Yi's co_update candidate order: staged/extra edges first, then live old
    // neighbors, then at most kTwoHopBypassPerDeleted live two-hop candidates
    // per deleted neighbor. Order-preserving with dedup (Yi skips the dedup
    // but the fast path keeps the pool verbatim, so duplicates would waste
    // degree slots).
    std::vector<uint32_t> cand;
    cand.reserve(extra_edges.size() + nd.nbrs.size());
    std::unordered_set<uint32_t> seen;
    const auto push = [&](uint32_t v) {
      if (v != node_id && !slot_alloc_.is_deleted(v) && seen.insert(v).second) {
        cand.push_back(v);
      }
    };
    for (const uint32_t e : extra_edges) {
      push(e);
    }
    const auto &removed_node_nbrs = update_ctx_.removed_node_nbrs_;
    for (const uint32_t nbr : nd.nbrs) {
      if (nbr == node_id) {
        continue;
      }
      if (!slot_alloc_.is_deleted(nbr)) {
        push(nbr);
        continue;
      }
      const auto it = removed_node_nbrs.find(nbr);
      if (it == removed_node_nbrs.end()) {
        continue;
      }
      uint32_t pulled = 0;
      for (const uint32_t o : it->second) {
        if (o == node_id || slot_alloc_.is_deleted(o)) {
          continue;
        }
        push(o);
        if (++pulled == kTwoHopBypassPerDeleted) {
          break;
        }
      }
    }
    prune_and_write(node_id, nd.nbrs, cand);
  }

  void update_node_impl_locked(uint32_t node_id, const std::vector<uint32_t> &extra_edges) {
    std::lock_guard<std::mutex> node_lock(update_node_mutex(node_id));
    update_node_impl(node_id, extra_edges);
  }

  /// Warm every page a reconnect of @p node_id will touch, with suspending
  /// reactor reads, BEFORE the caller takes the per-node mutex: the node's own
  /// page (read + rewritten in place) and — when the candidate pool will
  /// overflow into distance work — the candidate coords. The sync reconnect
  /// body re-derives everything under the lock; a page evicted in between just
  /// falls back to the blocking pread it does today, so slight staleness here
  /// costs latency, never correctness.
  coro::task<> prefetch_reconnect_inputs(uint32_t node_id,
                                         const std::vector<uint32_t> &extra_edges,
                                         coro::thread_pool &pool) {
    if (!page_io_->reactor_enabled()) {
      co_return;
    }
    const DiskPageIO::NodeData nd = co_await page_io_->read_node_async(node_id, pool);
    // Mirror update_node_impl_from_snapshot's candidate derivation.
    std::vector<uint32_t> cand;
    cand.reserve(extra_edges.size() + nd.nbrs.size() + 1);
    std::unordered_set<uint32_t> seen;
    const auto push = [&](uint32_t v) {
      if (v != node_id && !slot_alloc_.is_deleted(v) && seen.insert(v).second) {
        cand.push_back(v);
      }
    };
    for (const uint32_t e : extra_edges) {
      push(e);
    }
    const auto &removed_node_nbrs = update_ctx_.removed_node_nbrs_;
    for (const uint32_t nbr : nd.nbrs) {
      if (nbr == node_id) {
        continue;
      }
      if (!slot_alloc_.is_deleted(nbr)) {
        push(nbr);
        continue;
      }
      const auto it = removed_node_nbrs.find(nbr);
      if (it == removed_node_nbrs.end()) {
        continue;
      }
      uint32_t pulled = 0;
      for (const uint32_t o : it->second) {
        if (o == node_id || slot_alloc_.is_deleted(o)) {
          continue;
        }
        push(o);
        if (++pulled == kTwoHopBypassPerDeleted) {
          break;
        }
      }
    }
    if (cand.size() <= max_degree_) {
      co_return;  // fast path keeps the pool verbatim — no distance reads at all
    }
    if (!has_pq_) {
      // score_candidates reads self + candidate coords through the coords cache.
      // (PQ mode scores in memory; its only coords read — mirror_neighbors_to_cache
      // on a NodeCache miss — is served by the node page warmed above.)
      cand.push_back(node_id);
      co_await page_io_->prefetch_coords(cand.data(), cand.size(), pool);
    }
    co_return;
  }

  void update_node_impl_for_insert(uint32_t node_id, uint32_t inserted_slot) {
    const std::vector<uint32_t> extra_edges{inserted_slot};
    update_node_impl_locked(node_id, extra_edges);
  }

  void reconnect_inserted_neighbors_serial(const std::vector<uint32_t> &neighbors, uint32_t slot) {
    for (const uint32_t n : neighbors) {
      update_node_impl_for_insert(n, slot);
    }
  }

  void reconnect_inserted_neighbors_parallel(const std::vector<uint32_t> &neighbors,
                                             uint32_t slot) {
    const uint32_t workers =
        std::min<uint32_t>(update_reconnect_threads_, static_cast<uint32_t>(neighbors.size()));
    coro::thread_pool pool{{.thread_count = workers,
                            .on_thread_start_functor = nullptr,
                            .on_thread_stop_functor = nullptr}};
    auto reconnect_one = [this, &pool, slot](uint32_t node_id) -> coro::task<> {
      co_await pool.schedule();
      const std::vector<uint32_t> extra_edges{slot};
      // Node pages were warmed by the batch wave in run() below; No-PQ still
      // derives + waves candidate coords per node (see batch reconnect_one).
      if (!has_pq_) {
        co_await prefetch_reconnect_inputs(node_id, extra_edges, pool);
      }
      std::lock_guard<std::mutex> node_lock(update_node_mutex(node_id));
      update_node_impl(node_id, extra_edges);
    };
    auto run = [&]() -> coro::task<> {
      if (page_io_->reactor_enabled() && !neighbors.empty()) {
        co_await page_io_->prefetch_pages(neighbors.data(), neighbors.size(), pool);
      }
      std::vector<coro::task<>> tasks;
      tasks.reserve(neighbors.size());
      for (const uint32_t node_id : neighbors) {
        tasks.emplace_back(reconnect_one(node_id));
      }
      co_await coro::when_all(std::move(tasks));
    };
    try {
      coro::sync_wait(run());
    } catch (...) {
      pool.shutdown();
      throw;
    }
    pool.shutdown();
  }

  void reconnect_inserted_neighbors(const std::vector<uint32_t> &neighbors, uint32_t slot) {
    if (neighbors.empty()) {
      return;
    }
    if (update_reconnect_threads_ == 1 || neighbors.size() == 1) {
      reconnect_inserted_neighbors_serial(neighbors, slot);
      return;
    }
    reconnect_inserted_neighbors_parallel(neighbors, slot);
  }

  /// Lightweight consolidation: just strip dangling edges to tombstoned nodes.
  bool maybe_safety_net_reconnect() {
    if (!update_ctx_.needs_safety_net_reconnect(safety_net_ratio_,
                                                slot_alloc_.tombstone_count(),
                                                max_slot_id_,
                                                ops_since_last_insert_,
                                                safety_net_ops_)) {
      return false;
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
        mirror_neighbors_to_cache(nid, live);
      }
    }
    {
      std::unique_lock<std::shared_mutex> state_lock(update_mutex_);
      ops_since_last_insert_ = 0;
      ++safety_net_fires_;
    }
    return true;
  }

  /// Grow labels_ on append; assign in place on slot reuse.
  void set_label(uint32_t slot, uint64_t label) {
    if (slot >= labels_.size()) {
      labels_.resize(static_cast<size_t>(slot) + 1, kNoLabel);
    }
    labels_[slot] = label;
  }

  void resize_thread_data_slot_capacity() {
    for (auto &td : thread_data_storage_) {
      td->resize_slot_capacity(max_slot_id_);
    }
  }

  std::mutex &update_node_mutex(uint32_t node_id) {
    return update_node_locks_[node_id % update_node_locks_.size()];
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
    // Gate holds raw pointers into thread_data_storage_: drop them before the
    // storage goes away. Updates are quiesced here, so no waiter is parked.
    update_search_td_gate_.clear();
    update_search_async_ = false;
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
    update_reactor_.reset();  // after page_io_: it holds a raw pointer to the reactor
    update_ctx_.clear();
    label_to_slot_.clear();
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
  std::unordered_map<uint64_t, uint32_t> label_to_slot_;  ///< Live external label to slot.
  NodeCache cache_;
  PQTable pq_;
  std::unique_ptr<AlignedFileReader> reader_;

  // thread-scratch pool
  std::vector<std::unique_ptr<ThreadData>> thread_data_storage_;
  mutable ::ConcurrentQueue<ThreadData *> thread_data_pool_{nullptr};
  ThreadDataScratchConfig scratch_config_;  ///< saved at load for the gate tds
  bool search_page_cache_ = true;  ///< searches peek+fill the shard page cache (Yi pool view)
  alaya::AsyncGate<ThreadData> update_search_td_gate_;  ///< reactor-mode update searches
  bool update_search_async_ = false;                    ///< No-PQ + reactor + O_DIRECT fd available
#if defined(__linux__)
  // search_pipelined slot contexts, cached across calls (per-slot scratch is
  // ~12 B/slot/td — rebuilding per call would dwarf the queries themselves).
  mutable std::vector<std::unique_ptr<ThreadData>> pipeline_tds_;
  mutable std::mutex pipeline_tds_mutex_;  ///< serializes concurrent pipelined calls

  /// Grows the pipeline td cache to @p inflight slots and first-touches every
  /// td's per-slot scratch in parallel on @p pool (the first touch of
  /// ~1 GiB/td at 90M slots costs ~0.5 s single-threaded and must not
  /// serialize into the first in-flight queries). Caller must hold
  /// pipeline_tds_mutex_.
  void ensure_pipeline_tds_locked(uint32_t inflight, coro::thread_pool &pool) const {
    while (pipeline_tds_.size() < inflight) {
      auto td = std::make_unique<ThreadData>();
      td->alloc_scratch(scratch_config_);
      td->ensure_wave_scratch(static_cast<uint64_t>(std::max<uint32_t>(1, beam_width_)) *
                              geom_.page_size);
      pipeline_tds_.push_back(std::move(td));
    }
    const uint64_t max_slot = max_slot_id();
    std::vector<coro::task<void>> warm;
    warm.reserve(inflight);
    auto presize = [&](ThreadData *td) -> coro::task<void> {
      co_await pool.schedule();
      td->resize_slot_capacity(max_slot);
    };
    for (uint32_t t = 0; t < inflight; ++t) {
      warm.emplace_back(presize(pipeline_tds_[t].get()));
    }
    coro::sync_wait(coro::when_all(std::move(warm)));
  }
#endif

  // in-place update state (active only when updatable_)
  bool updatable_ = false;
  uint64_t max_slot_id_ = 0;  ///< file capacity in slots (valid-id bound; only grows)
  uint64_t live_count_ = 0;   ///< live (non-tombstoned) vector count
  std::string index_dir_;     ///< saved at load for flush() output paths
  std::unique_ptr<alaya::UringReactor> update_reactor_;  ///< declared before page_io_ so the
                                                         ///< page IO (raw-pointer user) dies first
  std::unique_ptr<DiskPageIO> page_io_;
  SlotAllocator slot_alloc_;
  DiskUpdateContext update_ctx_;
  mutable std::shared_mutex update_mutex_;  ///< shared search, exclusive mutation
  std::mutex update_serial_mutex_;
  std::mutex slot_mutex_;
  mutable std::shared_mutex pq_mutex_;
  std::array<std::mutex, kUpdateNodeLockStripes> update_node_locks_;
  uint64_t ops_since_last_insert_ = 0;
  uint64_t safety_net_fires_ = 0;
  float update_alpha_ = 1.2f;
  uint32_t update_search_l_ = 100;
  bool update_rerank_ = true;
  bool update_insert_prune_ = false;
  double safety_net_ratio_ = 0.05;
  uint64_t safety_net_ops_ = 16;
  uint32_t update_insert_threads_ = kDefaultDiskANNUpdateInsertThreads;
  uint32_t update_reconnect_threads_ = kDefaultDiskANNUpdateReconnectThreads;

  // update-stage instrumentation (see UpdateStageStats)
  std::atomic<uint64_t> st_gate_us_{0};
  std::atomic<uint64_t> st_greedy_us_{0};
  std::atomic<uint64_t> st_search_us_{0};
  std::atomic<uint64_t> st_alloc_us_{0};
  std::atomic<uint64_t> st_prefetch_us_{0};
  std::atomic<uint64_t> st_write_us_{0};
  std::atomic<uint64_t> st_reconnect_us_{0};
  std::atomic<uint64_t> st_rc_prefetch_us_{0};
  std::atomic<uint64_t> st_rc_lock_us_{0};
  std::atomic<uint64_t> st_rc_impl_us_{0};
  std::atomic<uint64_t> st_inserts_{0};
  std::atomic<uint64_t> st_reconnects_{0};

  static uint64_t stage_us_since(std::chrono::steady_clock::time_point &mark) {
    const auto now = std::chrono::steady_clock::now();
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(now - mark).count();
    mark = now;
    return static_cast<uint64_t>(us);
  }
};

}  // namespace alaya::diskann
