// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file disk_page_io.hpp
 * @brief Sector-aligned read-modify-write of node records in diskann.index.
 *
 * `DiskPageIO` (design D6) is the single place that writes the disk index during
 * in-place updates. It owns its own `O_DIRECT | O_RDWR` file descriptor (separate
 * from the search reader's read-only descriptor); in serial-update mode (a global
 * mutex serialises search and update) the two descriptors never race, and because
 * both use O_DIRECT a completed pwrite is visible to a subsequent pread.
 *
 * Each operation read-modify-writes one sector-aligned page so co-resident nodes
 * survive updates. Appends extend the file with ftruncate. Linux O_DIRECT is
 * required for the private syscall path; non-Linux update calls throw loudly.
 *
 * Concurrency: state is sharded by page offset (mutex + LRU page cache + RMW
 * scratch per shard), so parallel reconnect workers touching different pages
 * proceed independently — the analog of Yi's per-buffer locks; a single global
 * mutex here was measured to flatline update throughput regardless of worker
 * count. Pages map to exactly one shard, which preserves per-page RMW atomicity.
 * pread/pwrite are positional and thread-safe on one fd; ftruncate extension is
 * serialized by a dedicated file mutex (lock order: shard -> file, always).
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "coro/sync_wait.hpp"
#include "coro/task.hpp"
#include "coro/thread_pool.hpp"
#include "coro/when_all.hpp"
#if defined(__linux__)
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/diskann/disk_page_cache.hpp"
#include "index/graph/laser/utils/memory.hpp"
#include "storage/io/uring_reactor.hpp"

namespace alaya::diskann {

class DiskPageIO {
 public:
  /// A decoded node record: raw coordinates plus the live neighbor id list.
  struct NodeData {
    std::vector<float> coords;   ///< dim float32 entries
    std::vector<uint32_t> nbrs;  ///< n_nbrs neighbor ids
  };

  DiskPageIO(const std::string &index_path,
             const DiskLayoutGeometry &geom,
             size_t page_cache_capacity = 0)
      : geom_(geom) {
    try {
      // Tiny caches keep the single-cache eviction semantics (one shard per
      // cached page); real workloads get the full shard fan-out.
      num_shards_ =
          page_cache_capacity == 0
              ? kNumShards
              : static_cast<uint32_t>(
                    std::min<size_t>(std::max<size_t>(page_cache_capacity, 1), kNumShards));
      const size_t shard_capacity =
          page_cache_capacity == 0 ? 0 : std::max<size_t>(1, page_cache_capacity / num_shards_);
      cache_enabled_ = shard_capacity > 0;
      page_cache_capacity_ = cache_enabled_ ? shard_capacity * num_shards_ : 0;
      shards_.reserve(num_shards_);
      for (uint32_t s = 0; s < num_shards_; ++s) {
        shards_.push_back(std::make_unique<Shard>(shard_capacity, geom_.page_size));
      }
      open_rw(index_path);  // sets fd_ + file_size_
    } catch (...) {
      close_fd();
      throw;
    }
  }

  ~DiskPageIO() { close_fd(); }

  DiskPageIO(const DiskPageIO &) = delete;
  DiskPageIO &operator=(const DiskPageIO &) = delete;
  DiskPageIO(DiskPageIO &&) = delete;
  DiskPageIO &operator=(DiskPageIO &&) = delete;

  /// Read the full node record (coords + neighbors) of @p id.
  NodeData read_node(uint32_t id) {
    Shard &shard = shard_for(geom_.get_page_offset(id));
    std::lock_guard<std::mutex> lock(shard.mutex);
    read_page_locked(shard, id);
    return node_from_page(shard.page_buf, id);
  }

  /// Attach a UringReactor: page misses in the awaitable paths become
  /// suspending io_uring reads instead of pool-thread-blocking preads.
  /// nullptr (default) keeps every path on blocking pread.
  void set_reactor(UringReactor *reactor) {
#if defined(__linux__)
    reactor_ = reactor;
#else
    (void)reactor;
#endif
  }

  [[nodiscard]] bool reactor_enabled() const {
#if defined(__linux__)
    return reactor_ != nullptr;
#else
    return false;
#endif
  }

  /// Awaitable single-node read. With a reactor the miss suspends on io_uring;
  /// otherwise the blocking O_DIRECT pread executes on @p pool.
  coro::task<NodeData> read_node_async(uint32_t id, coro::thread_pool &pool) {
#if defined(__linux__)
    if (reactor_ != nullptr) {
      co_return co_await read_node_reactor(id, pool);
    }
#endif
    co_await pool.schedule();
    co_return read_node_with_private_page(id);
  }

  /// Wave-prefetch the coords of [ids, ids+count) into the transient coords
  /// cache: pages missing from the shard caches are read CONCURRENTLY through
  /// the reactor with a single suspension. Afterwards read_coords_cached() (and
  /// the sync helpers built on it) are memory hits for these ids. Ids already
  /// cached are skipped; without a reactor this is a no-op (callers fall back
  /// to the lazy blocking reads they do today).
  coro::task<> prefetch_coords(const uint32_t *ids, size_t count, coro::thread_pool &pool) {
#if defined(__linux__)
    if (reactor_ == nullptr || count == 0) {
      co_return;
    }
    std::vector<uint32_t> missing;
    missing.reserve(count);
    {
      std::unordered_set<uint32_t> seen;
      seen.reserve(count);
      std::lock_guard<std::mutex> vec_lock(vec_mutex_);
      for (size_t i = 0; i < count; ++i) {
        const uint32_t id = ids[i];
        if (seen.insert(id).second && vec_cache_.find(id) == vec_cache_.end()) {
          missing.push_back(id);
        }
      }
    }
    if (!missing.empty()) {
      co_await wave_load_pages(missing,
                               pool,
                               /*skip_beyond_eof=*/false,
                               [this](uint64_t /*off*/,
                                      const char *page,
                                      const std::vector<uint32_t> &page_ids) {
                                 std::vector<std::pair<uint32_t, std::vector<float>>> decoded;
                                 decoded.reserve(page_ids.size());
                                 for (const uint32_t id : page_ids) {
                                   const NodeRecordView view{page + geom_.offset_to_node(id),
                                                             geom_.dim};
                                   decoded.emplace_back(id,
                                                        std::vector<float>(view.coords(),
                                                                           view.coords() +
                                                                               geom_.dim));
                                 }
                                 std::lock_guard<std::mutex> vec_lock(vec_mutex_);
                                 for (auto &kv : decoded) {
                                   vec_cache_.emplace(kv.first, std::move(kv.second));
                                 }
                               });
    }
#else
    (void)ids;
    (void)count;
    (void)pool;
#endif
    co_return;
  }

  /// Wave-warm the shard page caches with the pages holding [ids, ids+count)
  /// so an upcoming read-modify-write hits memory instead of a blocking pread
  /// under the shard mutex. Pages beyond EOF (fresh append slots) are skipped —
  /// the write path zero-fills those without reading. No-op without a reactor.
  coro::task<> prefetch_pages(const uint32_t *ids, size_t count, coro::thread_pool &pool) {
#if defined(__linux__)
    if (reactor_ == nullptr || count == 0) {
      co_return;
    }
    std::vector<uint32_t> unique_ids;
    unique_ids.reserve(count);
    {
      std::unordered_set<uint64_t> seen_pages;
      seen_pages.reserve(count);
      for (size_t i = 0; i < count; ++i) {
        if (seen_pages.insert(geom_.get_page_offset(ids[i])).second) {
          unique_ids.push_back(ids[i]);
        }
      }
    }
    co_await wave_load_pages(unique_ids,
                             pool,
                             /*skip_beyond_eof=*/true,
                             [](uint64_t /*off*/,
                                const char * /*page*/,
                                const std::vector<uint32_t> & /*page_ids*/) {});
#else
    (void)ids;
    (void)count;
    (void)pool;
#endif
    co_return;
  }

  /// Read multiple node records using coroutine tasks over private page buffers.
  /// Shared cache state is protected briefly; O_DIRECT pread runs outside the mutex
  /// and is guarded by a page epoch before cached state is reused.
  std::vector<NodeData> read_nodes_async(const std::vector<uint32_t> &ids, uint32_t threads) {
    return read_nodes_async(ids.data(), static_cast<uint32_t>(ids.size()), threads);
  }

  std::vector<NodeData> read_nodes_async(const std::vector<uint32_t> &ids,
                                         coro::thread_pool &pool) {
    return read_nodes_async(ids.data(), static_cast<uint32_t>(ids.size()), pool);
  }

  std::vector<NodeData> read_nodes_async(const uint32_t *ids, uint32_t count, uint32_t threads) {
    if (count == 0) {
      return {};
    }
    if (ids == nullptr) {
      throw std::invalid_argument("DiskPageIO::read_nodes_async: null ids");
    }
    const uint32_t workers = std::min<uint32_t>({std::max<uint32_t>(1, threads), count});
    if (workers == 1) {
      std::vector<NodeData> out(count);
      for (uint32_t i = 0; i < count; ++i) {
        out[i] = read_node(ids[i]);
      }
      return out;
    }

    coro::thread_pool pool{{.thread_count = workers,
                            .on_thread_start_functor = nullptr,
                            .on_thread_stop_functor = nullptr}};
    try {
      std::vector<NodeData> out = read_nodes_async(ids, count, pool);
      pool.shutdown();
      return out;
    } catch (...) {
      pool.shutdown();
      throw;
    }
  }

  std::vector<NodeData> read_nodes_async(const uint32_t *ids,
                                         uint32_t count,
                                         coro::thread_pool &pool) {
    if (count == 0) {
      return {};
    }
    if (ids == nullptr) {
      throw std::invalid_argument("DiskPageIO::read_nodes_async: null ids");
    }
    std::vector<NodeData> out(count);
    auto read_one = [this, &pool, ids, &out](uint32_t i) -> coro::task<> {
      out[i] = co_await read_node_async(ids[i], pool);
    };
    auto run = [&]() -> coro::task<> {
      std::vector<coro::task<>> tasks;
      tasks.reserve(count);
      for (uint32_t i = 0; i < count; ++i) {
        tasks.emplace_back(read_one(i));
      }
      co_await coro::when_all(std::move(tasks));
    };
    coro::sync_wait(run());
    return out;
  }

  /// Read only neighbor lists for a delete batch. IDs sharing a disk page reuse
  /// the loaded page buffer, avoiding repeated O_DIRECT reads and coords copies.
  std::vector<std::vector<uint32_t>> read_neighbors_batch(const uint32_t *ids, uint32_t count) {
    std::vector<std::vector<uint32_t>> out(count);
    const std::vector<uint32_t> order = page_sorted_order(ids, count);
    for (uint32_t begin = 0; begin < count;) {
      const uint64_t page_off = geom_.get_page_offset(ids[order[begin]]);
      uint32_t end = begin + 1;
      while (end < count && geom_.get_page_offset(ids[order[end]]) == page_off) {
        ++end;
      }
      Shard &shard = shard_for(page_off);
      std::lock_guard<std::mutex> lock(shard.mutex);
      read_page_locked(shard, ids[order[begin]]);
      for (uint32_t pos = begin; pos < end; ++pos) {
        out[order[pos]] = neighbors_from_page(shard.page_buf, ids[order[pos]]);
      }
      begin = end;
    }
    return out;
  }

  /// Parallel variant for Yi-style delete batches. Pages are read via the shard
  /// cache when present (dirty pages stay coherent) and raw pread otherwise;
  /// callers must serialize it against writes.
  std::vector<std::vector<uint32_t>> read_neighbors_batch_parallel(const uint32_t *ids,
                                                                   uint32_t count,
                                                                   uint32_t threads) {
#if defined(__linux__)
    if (count == 0) {
      return {};
    }
    if (threads <= 1) {
      return read_neighbors_batch(ids, count);
    }
    std::vector<uint32_t> order = page_sorted_order(ids, count);
    std::vector<NeighborPageGroup> groups = neighbor_page_groups(ids, order);
    const uint32_t workers =
        std::min<uint32_t>({threads, static_cast<uint32_t>(groups.size()), count});
    if (workers <= 1) {
      return read_neighbors_batch(ids, count);
    }

    std::vector<std::vector<uint32_t>> out(count);
    std::exception_ptr error;
    std::mutex error_mutex;
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (uint32_t worker = 0; worker < workers; ++worker) {
      pool.emplace_back([&, worker]() {
        try {
          read_neighbor_group_stride(ids, order, groups, worker, workers, out);
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (error == nullptr) {
            error = std::current_exception();
          }
        }
      });
    }
    for (auto &thread : pool) {
      thread.join();
    }
    if (error != nullptr) {
      std::rethrow_exception(error);
    }
    return out;
#else
    (void)threads;
    return read_neighbors_batch(ids, count);
#endif
  }

  /// Reactor variant of read_neighbors_batch_parallel: one wave of concurrent
  /// io_uring reads over the unique pages (single suspension) instead of a
  /// std::thread pool of blocking preads. Callers must serialize against writes,
  /// exactly like the parallel variant.
  coro::task<std::vector<std::vector<uint32_t>>>
  read_neighbors_batch_async(const uint32_t *ids, uint32_t count, coro::thread_pool &pool) {
#if defined(__linux__)
    if (reactor_ != nullptr && count > 1) {
      std::vector<std::vector<uint32_t>> out(count);
      std::vector<uint32_t> unique_ids;
      std::unordered_map<uint64_t, std::vector<uint32_t>> page_positions;
      page_positions.reserve(count);
      for (uint32_t i = 0; i < count; ++i) {
        auto [it, fresh] = page_positions.try_emplace(geom_.get_page_offset(ids[i]));
        if (fresh) {
          unique_ids.push_back(ids[i]);
        }
        it->second.push_back(i);
      }
      co_await wave_load_pages(unique_ids,
                               pool,
                               /*skip_beyond_eof=*/false,
                               [&](uint64_t page_off,
                                   const char *page,
                                   const std::vector<uint32_t> & /*page_ids*/) {
                                 for (const uint32_t pos : page_positions.at(page_off)) {
                                   out[pos] = neighbors_from_page(page, ids[pos]);
                                 }
                               });
      co_return out;
    }
#endif
    co_await pool.schedule();
    co_return read_neighbors_batch_parallel(ids, count, update_fallback_threads_);
  }

  /// Thread count used by the blocking fallback of read_neighbors_batch_async.
  void set_fallback_threads(uint32_t threads) {
    update_fallback_threads_ = std::max<uint32_t>(1, threads);
  }

  /// Write a complete node record `[coords | n_nbrs | nbr_ids]`. Read-modify-write
  /// that preserves co-resident nodes; extends the file for a new-append slot.
  void write_node(uint32_t id, const float *coords, uint32_t n_nbrs, const uint32_t *nbr_ids) {
    if (n_nbrs > geom_.max_degree) {
      throw std::invalid_argument("DiskPageIO::write_node: n_nbrs exceeds max_degree");
    }
    {
      Shard &shard = shard_for(geom_.get_page_offset(id));
      std::lock_guard<std::mutex> lock(shard.mutex);
      load_page_for_write_locked(shard, id);
      char *rec = shard.page_buf + geom_.offset_to_node(id);
      std::memset(rec, 0, geom_.node_len);  // clear stale bytes (esp. a reused slot's tail)
      pack_node_record(rec, coords, nbr_ids, n_nbrs, geom_.dim);
      write_page_locked(shard, id);
    }
    std::lock_guard<std::mutex> vec_lock(vec_mutex_);
    vec_cache_[id].assign(coords, coords + geom_.dim);  // keep the coords cache coherent
  }

  /// Overwrite only the `n_nbrs` + `nbr_ids` fields, leaving coords untouched.
  void write_node_neighbors(uint32_t id, uint32_t n_nbrs, const uint32_t *nbr_ids) {
    if (n_nbrs > geom_.max_degree) {
      throw std::invalid_argument("DiskPageIO::write_node_neighbors: n_nbrs exceeds max_degree");
    }
    Shard &shard = shard_for(geom_.get_page_offset(id));
    std::lock_guard<std::mutex> lock(shard.mutex);
    read_page_locked(shard, id);  // must read first to preserve coords + co-resident nodes
    char *rec = shard.page_buf + geom_.offset_to_node(id);
    const uint64_t coords_bytes = geom_.dim * sizeof(float);
    // Zero the whole neighbor region first so no stale trailing ids survive.
    std::memset(rec + coords_bytes, 0, geom_.node_len - coords_bytes);
    std::memcpy(rec + coords_bytes, &n_nbrs, sizeof(n_nbrs));
    if (n_nbrs > 0) {
      std::memcpy(rec + coords_bytes + sizeof(uint32_t),
                  nbr_ids,
                  static_cast<size_t>(n_nbrs) * sizeof(uint32_t));
    }
    write_page_locked(shard, id);
  }

  /// Coords of @p id, served from the transient cache when present (design D8).
  std::vector<float> read_coords_cached(uint32_t id) {
    {
      std::lock_guard<std::mutex> vec_lock(vec_mutex_);
      auto it = vec_cache_.find(id);
      if (it != vec_cache_.end()) {
        return it->second;
      }
    }
    std::vector<float> coords;
    {
      Shard &shard = shard_for(geom_.get_page_offset(id));
      std::lock_guard<std::mutex> lock(shard.mutex);
      read_page_locked(shard, id);
      const NodeRecordView view{shard.page_buf + geom_.offset_to_node(id), geom_.dim};
      coords.assign(view.coords(), view.coords() + geom_.dim);
    }
    std::lock_guard<std::mutex> vec_lock(vec_mutex_);
    return vec_cache_.emplace(id, std::move(coords)).first->second;
  }

  /// Drop the transient coords cache (call between independent update operations).
  void clear_cache() {
    std::lock_guard<std::mutex> vec_lock(vec_mutex_);
    vec_cache_.clear();
  }

  /// Write all dirty cached pages to disk; clean pages stay cached.
  void flush_dirty_pages() {
    for (auto &shard_ptr : shards_) {
      Shard &shard = *shard_ptr;
      std::lock_guard<std::mutex> lock(shard.mutex);
      shard.cache.flush_dirty([this, &shard](uint64_t page_off, const char *page) {
        write_page_to_disk(shard, page_off, page);
      });
    }
  }

  [[nodiscard]] uint64_t file_size() const { return file_size_.load(std::memory_order_acquire); }
  [[nodiscard]] const DiskLayoutGeometry &geometry() const { return geom_; }

 private:
  /// Page-offset-sharded state: mutex + LRU cache + aligned RMW scratch. A page
  /// maps to exactly one shard, so per-page RMW atomicity is preserved while
  /// different pages proceed in parallel.
  static constexpr uint32_t kNumShards = 64;

  struct Shard {
    Shard(size_t cache_capacity, uint64_t page_size) : cache(cache_capacity) {
      page_buf = static_cast<char *>(alaya::laser::memory::align_allocate<kSectorLen>(page_size));
      try {
        flush_buf =
            static_cast<char *>(alaya::laser::memory::align_allocate<kSectorLen>(page_size));
      } catch (...) {
        alaya::laser::memory::align_free(page_buf);
        page_buf = nullptr;
        throw;
      }
    }
    ~Shard() {
      if (page_buf != nullptr) {
        alaya::laser::memory::align_free(page_buf);
      }
      if (flush_buf != nullptr) {
        alaya::laser::memory::align_free(flush_buf);
      }
    }
    Shard(const Shard &) = delete;
    Shard &operator=(const Shard &) = delete;

    std::mutex mutex;
    DiskPageCache cache;
    std::unordered_map<uint64_t, uint64_t> versions;
    char *page_buf = nullptr;   ///< RMW scratch, guarded by mutex
    char *flush_buf = nullptr;  ///< aligned bounce buffer for O_DIRECT cache flushes
  };

  Shard &shard_for(uint64_t page_off) {
    return *shards_[(page_off / geom_.page_size) % num_shards_];
  }

  std::vector<uint32_t> neighbors_from_page(const char *page, uint32_t id) const {
    const NodeRecordView view{page + geom_.offset_to_node(id), geom_.dim};
    const uint32_t n = view.n_nbrs();
    return std::vector<uint32_t>(view.nbrs(), view.nbrs() + n);
  }

  NodeData node_from_page(const char *page, uint32_t id) const {
    const NodeRecordView view{page + geom_.offset_to_node(id), geom_.dim};
    NodeData d;
    d.coords.assign(view.coords(), view.coords() + geom_.dim);
    const uint32_t n = view.n_nbrs();
    d.nbrs.assign(view.nbrs(), view.nbrs() + n);
    return d;
  }

  struct NeighborPageGroup {
    uint32_t begin = 0;
    uint32_t end = 0;
  };

  std::vector<uint32_t> page_sorted_order(const uint32_t *ids, uint32_t count) const {
    std::vector<uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return geom_.get_page_offset(ids[a]) < geom_.get_page_offset(ids[b]);
    });
    return order;
  }

  std::vector<NeighborPageGroup> neighbor_page_groups(const uint32_t *ids,
                                                      const std::vector<uint32_t> &order) const {
    std::vector<NeighborPageGroup> groups;
    for (uint32_t begin = 0; begin < order.size();) {
      uint32_t end = begin + 1;
      const uint64_t page_off = geom_.get_page_offset(ids[order[begin]]);
      while (end < order.size() && geom_.get_page_offset(ids[order[end]]) == page_off) {
        ++end;
      }
      groups.push_back(NeighborPageGroup{begin, end});
      begin = end;
    }
    return groups;
  }

#if defined(__linux__)
  /// Load the pages of the (unique-id) list @p ids with one reactor wave.
  /// Cached pages are consumed immediately without I/O; misses are read
  /// CONCURRENTLY with a single suspension, version-reconciled, installed into
  /// the shard caches, then consumed. @p consume runs OUTSIDE shard locks with
  /// (page_offset, page_bytes, ids_of_this_page).
  template <typename PageConsumer>
  coro::task<> wave_load_pages(const std::vector<uint32_t> &ids,
                               coro::thread_pool &pool,
                               bool skip_beyond_eof,
                               PageConsumer &&consume) {
    struct PageWork {
      uint64_t off = 0;
      std::vector<uint32_t> ids;
      uint64_t version = 0;
      char *buf = nullptr;
    };
    std::vector<PageWork> pages;
    {
      std::unordered_map<uint64_t, size_t> index_of;
      index_of.reserve(ids.size());
      for (const uint32_t id : ids) {
        const uint64_t off = geom_.get_page_offset(id);
        if (skip_beyond_eof && off + geom_.page_size > file_size_.load(std::memory_order_acquire)) {
          continue;  // fresh append page: the write path zero-fills it, nothing to read
        }
        auto [it, fresh] = index_of.try_emplace(off, pages.size());
        if (fresh) {
          pages.push_back(PageWork{off, {}, 0, nullptr});
        }
        pages[it->second].ids.push_back(id);
      }
    }
    if (pages.empty()) {
      co_return;
    }

    // One aligned block backs the whole wave (fd_ is O_DIRECT: buffer, offset
    // and length must all be sector-aligned — page_size slices of an aligned
    // base satisfy that).
    char *wave_buf = static_cast<char *>(
        alaya::laser::memory::align_allocate<kSectorLen>(pages.size() * geom_.page_size));
    try {
      // Phase 1: serve cache hits immediately; snapshot versions for misses.
      std::vector<size_t> misses;
      misses.reserve(pages.size());
      for (size_t p = 0; p < pages.size(); ++p) {
        PageWork &work = pages[p];
        work.buf = wave_buf + p * geom_.page_size;
        bool hit = false;
        {
          Shard &shard = shard_for(work.off);
          std::lock_guard<std::mutex> lock(shard.mutex);
          hit = shard.cache.read(work.off, work.buf, geom_.page_size);
          if (!hit) {
            work.version = page_version_locked(shard, work.off);
          }
        }
        if (hit) {
          consume(work.off, work.buf, work.ids);
        } else {
          misses.push_back(p);
        }
      }

      if (!misses.empty()) {
        // Phase 2: one suspension for every missing page of the wave.
        std::vector<IORequest> reqs(misses.size());
        for (size_t m = 0; m < misses.size(); ++m) {
          PageWork &work = pages[misses[m]];
          reqs[m] = IORequest{work.buf, geom_.page_size, work.off, nullptr};
        }
        const uint32_t failures = co_await reactor_->read_batch(pool,
                                                                fd_,
                                                                reqs.data(),
                                                                static_cast<uint32_t>(reqs.size()));
        if (failures != 0) {
          throw std::runtime_error("DiskPageIO::wave_load_pages: " + std::to_string(failures) +
                                   " short/failed io_uring reads");
        }
        // Phase 3: reconcile each read page with its shard cache, then consume.
        for (const size_t p : misses) {
          PageWork &work = pages[p];
          reconcile_wave_page(work.off, work.buf, work.version);
          consume(work.off, work.buf, work.ids);
        }
      }
    } catch (...) {
      alaya::laser::memory::align_free(wave_buf);
      throw;
    }
    alaya::laser::memory::align_free(wave_buf);
    co_return;
  }

  /// A reactor read of @p off landed in @p buf; fold it into the shard cache
  /// under the shard lock — read_page_snapshot's protocol, shared with the
  /// public search-side fill.
  void reconcile_wave_page(uint64_t off, char *buf, uint64_t observed_version) {
    search_fill_page(off, buf, observed_version);
  }

  /// Single-node read through the reactor: cache hit is immediate, a miss
  /// suspends on io_uring and reconciles like a wave page.
  coro::task<NodeData> read_node_reactor(uint32_t id, coro::thread_pool &pool) {
    const uint64_t off = geom_.get_page_offset(id);
    char *buf =
        static_cast<char *>(alaya::laser::memory::align_allocate<kSectorLen>(geom_.page_size));
    try {
      bool hit = false;
      uint64_t observed_version = 0;
      {
        Shard &shard = shard_for(off);
        std::lock_guard<std::mutex> lock(shard.mutex);
        hit = shard.cache.read(off, buf, geom_.page_size);
        if (!hit) {
          observed_version = page_version_locked(shard, off);
        }
      }
      if (!hit) {
        const int32_t got =
            co_await reactor_->read(pool, fd_, buf, static_cast<uint32_t>(geom_.page_size), off);
        if (got != static_cast<int32_t>(geom_.page_size)) {
          throw std::runtime_error("DiskPageIO::read_node_reactor: short/failed read at " +
                                   std::to_string(off) + " (got " + std::to_string(got) + ")");
        }
        reconcile_wave_page(off, buf, observed_version);
      }
      NodeData node = node_from_page(buf, id);
      alaya::laser::memory::align_free(buf);
      co_return node;
    } catch (...) {
      alaya::laser::memory::align_free(buf);
      throw;
    }
  }
#endif

  NodeData read_node_with_private_page(uint32_t id) {
    char *buf =
        static_cast<char *>(alaya::laser::memory::align_allocate<kSectorLen>(geom_.page_size));
    try {
      read_page_snapshot(id, buf);
      NodeData node = node_from_page(buf, id);
      alaya::laser::memory::align_free(buf);
      return node;
    } catch (...) {
      alaya::laser::memory::align_free(buf);
      throw;
    }
  }

 public:
  /// Non-blocking peek: copy the page containing @p off out of the update
  /// shard cache if present. The reactor-mode UPDATE search consults this
  /// before issuing a device read — recently written pages are exactly where
  /// an update workload's searches land (Yi's unified buffer pool gives its
  /// tasklets the same view). Eval/query search paths do not use this.
  bool try_read_cached_page(uint64_t off, char *out) {
    Shard &shard = shard_for(off);
    std::lock_guard<std::mutex> lock(shard.mutex);
    return shard.cache.read(off, out, geom_.page_size);
  }

  /// True when the shard page caches were built with a nonzero capacity —
  /// the search-side peek/fill pair below is pointless without them.
  [[nodiscard]] bool page_cache_enabled() const { return cache_enabled_; }

  /// Effective total page capacity across shards (0 when the cache is
  /// disabled). Prefetch-then-read waves must stay under this or the wave
  /// evicts its own pages before the reads land.
  [[nodiscard]] size_t page_cache_capacity() const { return page_cache_capacity_; }

  /// Peek + version snapshot, the read half of the search fill protocol.
  /// Hit: the page is copied into @p out and true is returned. Miss: false,
  /// and @p version_out receives the page's current write version so the
  /// caller can offer the device bytes back via search_fill_page().
  bool search_peek_page(uint64_t off, char *out, uint64_t *version_out) {
    Shard &shard = shard_for(off);
    std::lock_guard<std::mutex> lock(shard.mutex);
    if (shard.cache.read(off, out, geom_.page_size)) {
      return true;
    }
    *version_out = page_version_locked(shard, off);
    return false;
  }

  /// Fold a device-read page into the shard cache — the unified-pool fill
  /// that makes SEARCH misses populate the cache the way Yi's buffer pool
  /// does for its tasklets. If a writer bumped the page version while the
  /// read was in flight, @p buf is refreshed from the cache (or disk)
  /// instead of installing stale bytes, so callers must fill BEFORE they
  /// parse @p buf. Safe to call with a version from search_peek_page().
  void search_fill_page(uint64_t off, char *buf, uint64_t observed_version) {
    Shard &shard = shard_for(off);
    std::lock_guard<std::mutex> lock(shard.mutex);
    if (shard.cache.read(off, buf, geom_.page_size)) {
      return;  // someone installed meanwhile (possibly a fresher write) — cache wins
    }
    if (page_version_locked(shard, off) != observed_version) {
      read_page_locked_off(shard, off);  // rare: a write and an eviction raced our read
      std::memcpy(buf, shard.page_buf, geom_.page_size);
      return;
    }
    shard.cache.write(off,
                      buf,
                      geom_.page_size,
                      false,
                      [this, &shard](uint64_t page_off, const char *page) {
                        write_page_to_disk(shard, page_off, page);
                      });
  }

 private:
  void read_page_snapshot(uint32_t id, char *out) {
    const uint64_t off = geom_.get_page_offset(id);
    Shard &shard = shard_for(off);
    uint64_t observed_version = 0;
    {
      std::lock_guard<std::mutex> lock(shard.mutex);
      if (shard.cache.read(off, out, geom_.page_size)) {
        return;
      }
      observed_version = page_version_locked(shard, off);
    }

    read_page_from_disk(off, out);

    std::lock_guard<std::mutex> lock(shard.mutex);
    if (shard.cache.read(off, out, geom_.page_size)) {
      return;
    }
    if (page_version_locked(shard, off) != observed_version) {
      read_page_locked(shard, id);
      std::memcpy(out, shard.page_buf, geom_.page_size);
      return;
    }
    shard.cache.write(off,
                      out,
                      geom_.page_size,
                      false,
                      [this, &shard](uint64_t page_off, const char *page) {
                        write_page_to_disk(shard, page_off, page);
                      });
  }

  uint64_t page_version_locked(const Shard &shard, uint64_t page_off) const {
    const auto it = shard.versions.find(page_off);
    if (it == shard.versions.end()) {
      return 0;
    }
    return it->second;
  }

#if defined(__linux__)
  void read_neighbor_group_stride(const uint32_t *ids,
                                  const std::vector<uint32_t> &order,
                                  const std::vector<NeighborPageGroup> &groups,
                                  uint32_t worker,
                                  uint32_t workers,
                                  std::vector<std::vector<uint32_t>> &out) {
    char *buf =
        static_cast<char *>(alaya::laser::memory::align_allocate<kSectorLen>(geom_.page_size));
    try {
      for (uint32_t group_id = worker; group_id < groups.size(); group_id += workers) {
        const NeighborPageGroup group = groups[group_id];
        const uint32_t first_id = ids[order[group.begin]];
        const uint64_t page_off = geom_.get_page_offset(first_id);
        // Serve from the shard cache when present — a dirty cached page is
        // newer than the disk copy. Miss => the disk copy is current.
        bool cached = false;
        {
          Shard &shard = shard_for(page_off);
          std::lock_guard<std::mutex> lock(shard.mutex);
          cached = shard.cache.read(page_off, buf, geom_.page_size);
        }
        if (!cached) {
          const ssize_t r = ::pread(fd_, buf, geom_.page_size, static_cast<off_t>(page_off));
          if (r != static_cast<ssize_t>(geom_.page_size)) {
            throw std::runtime_error(
                "DiskPageIO::read_neighbors_batch_parallel: short/failed pread");
          }
        }
        for (uint32_t pos = group.begin; pos < group.end; ++pos) {
          const uint32_t out_pos = order[pos];
          out[out_pos] = neighbors_from_page(buf, ids[out_pos]);
        }
      }
      alaya::laser::memory::align_free(buf);
    } catch (...) {
      alaya::laser::memory::align_free(buf);
      throw;
    }
  }
#endif

  /// Ensure the page holding @p id exists, with the shard's page_buf holding its
  /// current content (existing page: RMW) or zeros (freshly extended page).
  /// Caller holds the shard mutex; extension takes file_mutex_ (shard -> file
  /// lock order, always).
  void load_page_for_write_locked(Shard &shard, uint32_t id) {
    const uint64_t page_off = geom_.get_page_offset(id);
    const uint64_t page_end = page_off + geom_.page_size;
    if (page_end <= file_size_.load(std::memory_order_acquire)) {
      read_page_locked(shard, id);
      return;
    }
    {
      std::lock_guard<std::mutex> file_lock(file_mutex_);
      if (page_end > file_size_.load(std::memory_order_acquire)) {
        extend_to(page_end);  // ftruncate; OS zero-fills the new region
        std::memset(shard.page_buf, 0, geom_.page_size);
        return;
      }
    }
    read_page_locked(shard, id);  // another thread extended past us meanwhile
  }

  // ---- platform-gated syscalls (Linux O_DIRECT) ----
  void open_rw(const std::string &path);
  void read_page_locked(Shard &shard, uint32_t id);
  void read_page_locked_off(Shard &shard, uint64_t page_off);
  void read_page_from_disk(uint64_t page_off, char *page) const;
  void write_page_locked(Shard &shard, uint32_t id);
  void write_page_to_disk(Shard &shard, uint64_t page_off, const char *page);
  void extend_to(uint64_t new_size);
  void close_fd();

  DiskLayoutGeometry geom_;
  int fd_ = -1;
  std::atomic<uint64_t> file_size_{0};
  uint32_t num_shards_ = kNumShards;
  bool cache_enabled_ = false;
  size_t page_cache_capacity_ = 0;
  std::vector<std::unique_ptr<Shard>> shards_;
  std::unordered_map<uint32_t, std::vector<float>> vec_cache_;
  mutable std::mutex vec_mutex_;
  std::mutex file_mutex_;  ///< serializes ftruncate extension (after shard lock)
#if defined(__linux__)
  UringReactor *reactor_ = nullptr;  ///< not owned; nullptr = blocking pread paths
#endif
  uint32_t update_fallback_threads_ = 8;  ///< blocking fallback of read_neighbors_batch_async
};

#if defined(__linux__)

inline void DiskPageIO::open_rw(const std::string &path) {
  fd_ = ::open(path.c_str(), O_DIRECT | O_RDWR);  // NOLINT(hicpp-vararg)
  if (fd_ < 0) {
    throw std::runtime_error("DiskPageIO::open_rw: cannot open (O_DIRECT|O_RDWR) " + path);
  }
  struct stat st{};
  if (::fstat(fd_, &st) != 0) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("DiskPageIO::open_rw: fstat failed " + path);
  }
  file_size_.store(static_cast<uint64_t>(st.st_size), std::memory_order_release);
}

inline void DiskPageIO::read_page_locked(Shard &shard, uint32_t id) {
  read_page_locked_off(shard, geom_.get_page_offset(id));
}

inline void DiskPageIO::read_page_locked_off(Shard &shard, uint64_t page_off) {
  if (shard.cache.read(page_off, shard.page_buf, geom_.page_size)) {
    return;
  }
  read_page_from_disk(page_off, shard.page_buf);
  shard.cache.write(page_off,
                    shard.page_buf,
                    geom_.page_size,
                    false,
                    [this, &shard](uint64_t off, const char *page) {
                      write_page_to_disk(shard, off, page);
                    });
}

inline void DiskPageIO::read_page_from_disk(uint64_t page_off, char *page) const {
  const ssize_t r = ::pread(fd_, page, geom_.page_size, static_cast<off_t>(page_off));
  if (r != static_cast<ssize_t>(geom_.page_size)) {
    throw std::runtime_error("DiskPageIO::read_page: short/failed pread at " +
                             std::to_string(page_off) + " (got " + std::to_string(r) + ")");
  }
}

inline void DiskPageIO::write_page_locked(Shard &shard, uint32_t id) {
  const uint64_t off = geom_.get_page_offset(id);
  if (shard.cache.enabled()) {
    shard.cache.write(off,
                      shard.page_buf,
                      geom_.page_size,
                      true,
                      [this, &shard](uint64_t page_off, const char *page) {
                        write_page_to_disk(shard, page_off, page);
                      });
    ++shard.versions[off];
    return;
  }
  write_page_to_disk(shard, off, shard.page_buf);
  ++shard.versions[off];
}

inline void DiskPageIO::write_page_to_disk(Shard &shard, uint64_t page_off, const char *page) {
  const char *write_buf = page;
  if (page != shard.page_buf) {
    std::memcpy(shard.flush_buf, page, geom_.page_size);
    write_buf = shard.flush_buf;
  }
  const ssize_t w = ::pwrite(fd_, write_buf, geom_.page_size, static_cast<off_t>(page_off));
  if (w != static_cast<ssize_t>(geom_.page_size)) {
    throw std::runtime_error("DiskPageIO::write_page: short/failed pwrite at " +
                             std::to_string(page_off) + " (got " + std::to_string(w) + ")");
  }
}

inline void DiskPageIO::extend_to(uint64_t new_size) {
  if (::ftruncate(fd_, static_cast<off_t>(new_size)) != 0) {
    throw std::runtime_error("DiskPageIO::extend_to: ftruncate failed to " +
                             std::to_string(new_size));
  }
  file_size_.store(new_size, std::memory_order_release);
}

inline void DiskPageIO::close_fd() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

#else  // !__linux__ : in-place updates require Linux O_DIRECT — fail loudly when used.

inline void DiskPageIO::open_rw(const std::string &) {
  throw std::runtime_error("DiskPageIO: in-place DiskANN updates require Linux (O_DIRECT)");
}
inline void DiskPageIO::read_page_locked(Shard &, uint32_t) {
  throw std::runtime_error("DiskPageIO: unsupported platform (needs Linux O_DIRECT)");
}
inline void DiskPageIO::read_page_locked_off(Shard &, uint64_t) {
  throw std::runtime_error("DiskPageIO: unsupported platform (needs Linux O_DIRECT)");
}
inline void DiskPageIO::read_page_from_disk(uint64_t, char *) const {
  throw std::runtime_error("DiskPageIO: unsupported platform (needs Linux O_DIRECT)");
}
inline void DiskPageIO::write_page_locked(Shard &, uint32_t) {
  throw std::runtime_error("DiskPageIO: unsupported platform (needs Linux O_DIRECT)");
}
inline void DiskPageIO::write_page_to_disk(Shard &, uint64_t, const char *) {
  throw std::runtime_error("DiskPageIO: unsupported platform (needs Linux O_DIRECT)");
}
inline void DiskPageIO::extend_to(uint64_t) {
  throw std::runtime_error("DiskPageIO: unsupported platform (needs Linux O_DIRECT)");
}
inline void DiskPageIO::close_fd() {}

#endif  // __linux__

}  // namespace alaya::diskann
