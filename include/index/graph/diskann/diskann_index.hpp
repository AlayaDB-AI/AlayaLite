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
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "index/graph/diskann/beam_search.hpp"
#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/diskann/node_cache.hpp"
#include "index/graph/diskann/pq_table.hpp"
#include "index/graph/diskann/search_scratch.hpp"
#include "index/graph/laser/utils/aligned_file_reader_factory.hpp"
#include "index/graph/laser/utils/concurrent_queue.hpp"
#include "index/graph/vamana/vamana_builder.hpp"

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
  static constexpr uint32_t kMetaVersion = 1;
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
    num_points_ = meta.num_points;
    dim_ = meta.dim;
    max_degree_ = meta.max_degree;
    medoid_ = meta.medoid;
    has_pq_ = meta.has_pq != 0;
    pq_n_chunks_ = meta.pq_n_chunks;
    geom_ = DiskLayoutGeometry::compute(dim_, max_degree_);
    if (geom_.node_len != meta.node_len || geom_.nodes_per_sector != meta.nodes_per_sector) {
      throw std::runtime_error("DiskANNIndex::load: meta geometry inconsistent");
    }

    read_ids(path(index_dir, "ids.bin"), num_points_);
    cache_.load(path(index_dir, "cache_ids.bin"), path(index_dir, "cache_nodes.bin"));
    if (has_pq_) {
      pq_.load(path(index_dir, "pq_pivots.bin"),
               path(index_dir, "pq_compressed.bin"),
               num_points_,
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

    ThreadData *td = acquire();
    uint32_t count = 0;
    try {
      SearchContext ctx;
      ctx.reader = reader_.get();
      ctx.geom = &geom_;
      ctx.cache = &cache_;
      ctx.pq = has_pq_ ? &pq_ : nullptr;
      ctx.medoid = medoid_;
      ctx.num_points = num_points_;

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
  [[nodiscard]] uint64_t size() const { return num_points_; }
  [[nodiscard]] uint64_t dim() const { return dim_; }
  [[nodiscard]] bool has_pq() const { return has_pq_; }
  [[nodiscard]] uint32_t medoid() const { return medoid_; }

  /// Sentinel label for padded (missing) result slots.
  static constexpr uint64_t kNoLabel = std::numeric_limits<uint64_t>::max();

 private:
  struct MetaHeader {
    uint64_t num_points = 0;
    uint64_t dim = 0;
    uint32_t max_degree = 0;
    uint32_t medoid = 0;
    uint8_t has_pq = 0;
    uint32_t pq_n_chunks = 0;
    uint64_t node_len = 0;
    uint64_t nodes_per_sector = 0;
  };

  static std::string path(const std::string &dir, const char *name) {
    return (std::filesystem::path(dir) / name).string();
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
    reader_.reset();
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
    if (version != kMetaVersion) {
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
  uint64_t num_points_ = 0;
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
  mutable ::ConcurrentQueue<ThreadData *> thread_data_pool_;
};

}  // namespace alaya::diskann
