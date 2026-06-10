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
 * Every operation works on one sector-aligned page:
 *   1. compute the page offset via `DiskLayoutGeometry` (O(1));
 *   2. read the full page into a 4096-aligned buffer (preserving co-resident
 *      nodes when several share a sector);
 *   3. modify the target node's record in place;
 *   4. pwrite the full page back.
 *
 * Appends past the current file size extend it with `ftruncate` first (the kernel
 * zero-fills the new region). A transient coords cache (design D8) lets a single
 * insert's reconnect avoid re-reading candidate vectors it already touched.
 *
 * The public API is platform-neutral; the five private syscalls are gated to
 * Linux (O_DIRECT). On other platforms they throw — updates are Linux-only by
 * design, and the header still compiles so cross-platform search builds are
 * unaffected.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/laser/utils/memory.hpp"

namespace alaya::diskann {

class DiskPageIO {
 public:
  /// A decoded node record: raw coordinates plus the live neighbor id list.
  struct NodeData {
    std::vector<float> coords;   ///< dim float32 entries
    std::vector<uint32_t> nbrs;  ///< n_nbrs neighbor ids
  };

  DiskPageIO(const std::string &index_path, const DiskLayoutGeometry &geom) : geom_(geom) {
    page_buf_ =
        static_cast<char *>(alaya::laser::memory::align_allocate<kSectorLen>(geom_.page_size));
    try {
      open_rw(index_path);  // sets fd_ + file_size_
    } catch (...) {
      alaya::laser::memory::align_free(page_buf_);
      page_buf_ = nullptr;
      throw;
    }
  }

  ~DiskPageIO() {
    close_fd();
    if (page_buf_ != nullptr) {
      alaya::laser::memory::align_free(page_buf_);
      page_buf_ = nullptr;
    }
  }

  DiskPageIO(const DiskPageIO &) = delete;
  DiskPageIO &operator=(const DiskPageIO &) = delete;
  DiskPageIO(DiskPageIO &&) = delete;
  DiskPageIO &operator=(DiskPageIO &&) = delete;

  /// Read the full node record (coords + neighbors) of @p id.
  NodeData read_node(uint32_t id) {
    read_page(id);
    const NodeRecordView view{page_buf_ + geom_.offset_to_node(id), geom_.dim};
    NodeData d;
    d.coords.assign(view.coords(), view.coords() + geom_.dim);
    const uint32_t n = view.n_nbrs();
    d.nbrs.assign(view.nbrs(), view.nbrs() + n);
    return d;
  }

  /// Write a complete node record `[coords | n_nbrs | nbr_ids]`. Read-modify-write
  /// that preserves co-resident nodes; extends the file for a new-append slot.
  void write_node(uint32_t id, const float *coords, uint32_t n_nbrs, const uint32_t *nbr_ids) {
    if (n_nbrs > geom_.max_degree) {
      throw std::invalid_argument("DiskPageIO::write_node: n_nbrs exceeds max_degree");
    }
    load_page_for_write(id);
    char *rec = page_buf_ + geom_.offset_to_node(id);
    std::memset(rec, 0, geom_.node_len);  // clear stale bytes (esp. a reused slot's tail)
    pack_node_record(rec, coords, nbr_ids, n_nbrs, geom_.dim);
    write_page(id);
    vec_cache_[id].assign(coords, coords + geom_.dim);  // keep the coords cache coherent
  }

  /// Overwrite only the `n_nbrs` + `nbr_ids` fields, leaving coords untouched.
  void write_node_neighbors(uint32_t id, uint32_t n_nbrs, const uint32_t *nbr_ids) {
    if (n_nbrs > geom_.max_degree) {
      throw std::invalid_argument("DiskPageIO::write_node_neighbors: n_nbrs exceeds max_degree");
    }
    read_page(id);  // must read first to preserve coords + co-resident nodes
    char *rec = page_buf_ + geom_.offset_to_node(id);
    const uint64_t coords_bytes = geom_.dim * sizeof(float);
    // Zero the whole neighbor region first so no stale trailing ids survive.
    std::memset(rec + coords_bytes, 0, geom_.node_len - coords_bytes);
    std::memcpy(rec + coords_bytes, &n_nbrs, sizeof(n_nbrs));
    if (n_nbrs > 0) {
      std::memcpy(rec + coords_bytes + sizeof(uint32_t),
                  nbr_ids,
                  static_cast<size_t>(n_nbrs) * sizeof(uint32_t));
    }
    write_page(id);
  }

  /// Coords of @p id, served from the transient cache when present (design D8).
  /// The reference is valid until the next clear_cache() / write_node(id).
  const std::vector<float> &read_coords_cached(uint32_t id) {
    auto it = vec_cache_.find(id);
    if (it != vec_cache_.end()) {
      return it->second;
    }
    read_page(id);
    const NodeRecordView view{page_buf_ + geom_.offset_to_node(id), geom_.dim};
    auto res = vec_cache_.emplace(id, std::vector<float>(view.coords(), view.coords() + geom_.dim));
    return res.first->second;
  }

  /// Drop the transient coords cache (call between independent update operations).
  void clear_cache() { vec_cache_.clear(); }

  [[nodiscard]] uint64_t file_size() const { return file_size_; }
  [[nodiscard]] const DiskLayoutGeometry &geometry() const { return geom_; }

 private:
  /// Ensure the page holding @p id exists, with page_buf_ holding its current
  /// content (existing page: RMW) or zeros (freshly extended page).
  void load_page_for_write(uint32_t id) {
    const uint64_t page_off = geom_.get_page_offset(id);
    const uint64_t page_end = page_off + geom_.page_size;
    if (page_end <= file_size_) {
      read_page(id);
    } else {
      extend_to(page_end);                         // ftruncate; OS zero-fills the new region
      std::memset(page_buf_, 0, geom_.page_size);  // fresh in-memory page
    }
  }

  // ---- platform-gated syscalls (Linux O_DIRECT) ----
  void open_rw(const std::string &path);
  void read_page(uint32_t id);
  void write_page(uint32_t id);
  void extend_to(uint64_t new_size);
  void close_fd();

  DiskLayoutGeometry geom_;
  int fd_ = -1;
  uint64_t file_size_ = 0;
  char *page_buf_ = nullptr;
  std::unordered_map<uint32_t, std::vector<float>> vec_cache_;
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
  file_size_ = static_cast<uint64_t>(st.st_size);
}

inline void DiskPageIO::read_page(uint32_t id) {
  const uint64_t off = geom_.get_page_offset(id);
  const ssize_t r = ::pread(fd_, page_buf_, geom_.page_size, static_cast<off_t>(off));
  if (r != static_cast<ssize_t>(geom_.page_size)) {
    throw std::runtime_error("DiskPageIO::read_page: short/failed pread at " + std::to_string(off) +
                             " (got " + std::to_string(r) + ")");
  }
}

inline void DiskPageIO::write_page(uint32_t id) {
  const uint64_t off = geom_.get_page_offset(id);
  const ssize_t w = ::pwrite(fd_, page_buf_, geom_.page_size, static_cast<off_t>(off));
  if (w != static_cast<ssize_t>(geom_.page_size)) {
    throw std::runtime_error("DiskPageIO::write_page: short/failed pwrite at " +
                             std::to_string(off) + " (got " + std::to_string(w) + ")");
  }
}

inline void DiskPageIO::extend_to(uint64_t new_size) {
  if (::ftruncate(fd_, static_cast<off_t>(new_size)) != 0) {
    throw std::runtime_error("DiskPageIO::extend_to: ftruncate failed to " +
                             std::to_string(new_size));
  }
  file_size_ = new_size;
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
inline void DiskPageIO::read_page(uint32_t) {
  throw std::runtime_error("DiskPageIO: unsupported platform (needs Linux O_DIRECT)");
}
inline void DiskPageIO::write_page(uint32_t) {
  throw std::runtime_error("DiskPageIO: unsupported platform (needs Linux O_DIRECT)");
}
inline void DiskPageIO::extend_to(uint64_t) {
  throw std::runtime_error("DiskPageIO: unsupported platform (needs Linux O_DIRECT)");
}
inline void DiskPageIO::close_fd() {}

#endif  // __linux__

}  // namespace alaya::diskann
