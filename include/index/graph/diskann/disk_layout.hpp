// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file disk_layout.hpp
 * @brief Sector-aligned on-disk layout for the DiskANN disk index.
 *
 * The disk index file (`diskann.index`) begins with a 4096-byte header sector,
 * followed by node data packed into sector-aligned pages. Each node is a
 * fixed-length record:
 *
 *     [ float32[dim] coords | uint32 n_nbrs | uint32[max_degree] nbr_ids ]
 *
 * The fixed `node_len` lets a node's disk offset be computed directly from its
 * id (no auxiliary index), which is what makes async O_DIRECT reads and future
 * in-place delete/update possible.
 *
 * Geometry (matching the LASER QuantizedGraph convention so both share the same
 * AlignedFileReader contract):
 *   - node_len         = dim*4 + 4 + max_degree*4
 *   - nodes_per_sector = max(1, kSectorLen / node_len)
 *   - page_size        = ceil(nodes_per_sector * node_len / kSectorLen) * kSectorLen
 *
 * The header format is an explicit little-endian byte layout (not a struct
 * dump), so the on-disk format is decoupled from the host ABI.
 */

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace alaya::diskann {

static_assert(std::endian::native == std::endian::little,
              "DiskANN on-disk layout assumes little-endian host");

/// Sector size (O_DIRECT page granularity). All reads must be aligned to this.
inline constexpr uint64_t kSectorLen = 4096;

/// Fixed byte offsets of header fields within the 4096-byte header sector.
namespace header_offset {
inline constexpr size_t kNumPoints = 0;        ///< u64
inline constexpr size_t kDim = 8;              ///< u64
inline constexpr size_t kMedoid = 16;          ///< u32
inline constexpr size_t kMaxDegree = 20;       ///< u32
inline constexpr size_t kNodeLen = 24;         ///< u64
inline constexpr size_t kNodesPerSector = 32;  ///< u64
inline constexpr size_t kTotalFileSize = 40;   ///< u64 (header uses 48 bytes total)
}  // namespace header_offset

/// Parsed header sector contents.
struct DiskLayoutHeader {
  uint64_t num_points = 0;
  uint64_t dim = 0;
  uint32_t medoid = 0;
  uint32_t max_degree = 0;
  uint64_t node_len = 0;
  uint64_t nodes_per_sector = 0;
  uint64_t total_file_size = 0;
};

namespace detail {
inline void put_u64(char *buf, size_t off, uint64_t v) { std::memcpy(buf + off, &v, sizeof(v)); }
inline void put_u32(char *buf, size_t off, uint32_t v) { std::memcpy(buf + off, &v, sizeof(v)); }
inline uint64_t get_u64(const char *buf, size_t off) {
  uint64_t v;
  std::memcpy(&v, buf + off, sizeof(v));
  return v;
}
inline uint32_t get_u32(const char *buf, size_t off) {
  uint32_t v;
  std::memcpy(&v, buf + off, sizeof(v));
  return v;
}
}  // namespace detail

/**
 * @brief Sector geometry derived deterministically from (dim, max_degree).
 *
 * Bundles node_len / nodes_per_sector / page_size and the offset arithmetic so
 * callers never re-derive (or disagree on) the layout math.
 */
struct DiskLayoutGeometry {
  uint64_t dim = 0;
  uint32_t max_degree = 0;
  uint64_t node_len = 0;
  uint64_t nodes_per_sector = 0;
  uint64_t page_size = 0;

  /// Compute geometry for a given vector dimension and max graph degree.
  static DiskLayoutGeometry compute(uint64_t dim, uint32_t max_degree) {
    if (dim == 0) {
      throw std::invalid_argument("DiskLayoutGeometry: dim must be > 0");
    }
    DiskLayoutGeometry g;
    g.dim = dim;
    g.max_degree = max_degree;
    g.node_len = dim * sizeof(float) + sizeof(uint32_t) +
                 static_cast<uint64_t>(max_degree) * sizeof(uint32_t);
    g.nodes_per_sector = std::max<uint64_t>(1, kSectorLen / g.node_len);
    g.page_size = (g.nodes_per_sector * g.node_len + kSectorLen - 1) / kSectorLen * kSectorLen;
    return g;
  }

  /// File offset of the sector page that contains @p node_id.
  [[nodiscard]] uint64_t get_page_offset(uint64_t node_id) const {
    return kSectorLen + page_size * (node_id / nodes_per_sector);
  }

  /// Byte offset of @p node_id within its page.
  [[nodiscard]] uint64_t offset_to_node(uint64_t node_id) const {
    return (node_id % nodes_per_sector) * node_len;
  }

  /// Absolute file offset of @p node_id's record.
  [[nodiscard]] uint64_t file_offset(uint64_t node_id) const {
    return get_page_offset(node_id) + offset_to_node(node_id);
  }

  /// Number of sector pages needed to store @p num_points nodes.
  [[nodiscard]] uint64_t num_pages(uint64_t num_points) const {
    if (num_points == 0) {
      return 0;
    }
    return (num_points + nodes_per_sector - 1) / nodes_per_sector;
  }

  /// Total on-disk file size (header sector + all node pages) for @p num_points.
  [[nodiscard]] uint64_t total_file_size(uint64_t num_points) const {
    return kSectorLen + num_pages(num_points) * page_size;
  }
};

/**
 * @brief Serialize one node record into @p rec (node_len bytes).
 *
 * Writes @c [coords | n_nbrs | nbr_ids]. Unused neighbor slots are left
 * untouched, so the caller must pre-zero @p rec (or its enclosing page/buffer)
 * to satisfy the zero-fill requirement. Used by both the disk writer and the
 * node cache so cached records are byte-identical to on-disk records.
 */
inline void pack_node_record(char *rec,
                             const float *coords,
                             const uint32_t *nbrs,
                             uint32_t n_nbrs,
                             uint64_t dim) {
  const uint64_t coords_bytes = dim * sizeof(float);
  std::memcpy(rec, coords, coords_bytes);
  std::memcpy(rec + coords_bytes, &n_nbrs, sizeof(n_nbrs));
  if (n_nbrs > 0) {
    std::memcpy(rec + coords_bytes + sizeof(uint32_t),
                nbrs,
                static_cast<size_t>(n_nbrs) * sizeof(uint32_t));
  }
}

/**
 * @brief Read-only accessor over a node record (from disk page or cache).
 *
 * The same parser is used whether the @p rec bytes came from a cache hit or an
 * async disk read, guaranteeing the two paths interpret a node identically.
 */
struct NodeRecordView {
  const char *rec = nullptr;
  uint64_t dim = 0;

  /// Pointer to the node's coordinates (dim float32). Suitable for l2_sqr.
  [[nodiscard]] const float *coords() const { return reinterpret_cast<const float *>(rec); }

  /// Number of valid neighbors stored in this record.
  [[nodiscard]] uint32_t n_nbrs() const {
    uint32_t v;
    std::memcpy(&v, rec + dim * sizeof(float), sizeof(v));
    return v;
  }

  /// Pointer to the neighbor id array (n_nbrs() valid entries).
  [[nodiscard]] const uint32_t *nbrs() const {
    return reinterpret_cast<const uint32_t *>(rec + dim * sizeof(float) + sizeof(uint32_t));
  }
};

/// Inputs to write_disk_layout() that are not raw buffers.
struct WriteDiskLayoutParams {
  uint64_t num_points = 0;
  uint64_t dim = 0;
  uint32_t max_degree = 0;
  uint32_t medoid = 0;
};

/**
 * @brief Pack a graph + vectors into a sector-aligned disk index file.
 *
 * @param path     Output file path (overwritten if it exists).
 * @param vectors  Row-major @c num_points*dim float32 coordinates.
 * @param graph    Adjacency lists; graph[i] are the neighbor ids of node i.
 * @param params   num_points / dim / max_degree / medoid.
 *
 * Layout: a 4096-byte header sector, then node pages. Each node record is
 * @c [coords | n_nbrs | nbr_ids] with unused neighbor slots zero-filled.
 *
 * @throws std::invalid_argument if params are inconsistent (e.g. graph size
 *         mismatch, or a node has more than @c max_degree neighbors).
 * @throws std::runtime_error if the file cannot be written.
 */
inline void write_disk_layout(const std::string &path,
                              const float *vectors,
                              const std::vector<std::vector<uint32_t>> &graph,
                              const WriteDiskLayoutParams &params) {
  if (params.dim == 0) {
    throw std::invalid_argument("write_disk_layout: dim must be > 0");
  }
  if (params.num_points == 0) {
    throw std::invalid_argument("write_disk_layout: num_points must be > 0");
  }
  if (vectors == nullptr) {
    throw std::invalid_argument("write_disk_layout: vectors must not be null");
  }
  if (graph.size() != params.num_points) {
    throw std::invalid_argument("write_disk_layout: graph size (" + std::to_string(graph.size()) +
                                ") != num_points (" + std::to_string(params.num_points) + ")");
  }
  if (params.medoid >= params.num_points) {
    throw std::invalid_argument("write_disk_layout: medoid out of range");
  }

  const DiskLayoutGeometry geom = DiskLayoutGeometry::compute(params.dim, params.max_degree);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("write_disk_layout: cannot open " + path);
  }

  // --- Header sector ---
  std::vector<char> header(kSectorLen, 0);
  detail::put_u64(header.data(), header_offset::kNumPoints, params.num_points);
  detail::put_u64(header.data(), header_offset::kDim, params.dim);
  detail::put_u32(header.data(), header_offset::kMedoid, params.medoid);
  detail::put_u32(header.data(), header_offset::kMaxDegree, params.max_degree);
  detail::put_u64(header.data(), header_offset::kNodeLen, geom.node_len);
  detail::put_u64(header.data(), header_offset::kNodesPerSector, geom.nodes_per_sector);
  detail::put_u64(header.data(),
                  header_offset::kTotalFileSize,
                  geom.total_file_size(params.num_points));
  out.write(header.data(), static_cast<std::streamsize>(kSectorLen));

  // --- Node pages ---
  std::vector<char> page(geom.page_size, 0);
  for (uint64_t node_id = 0; node_id < params.num_points; ++node_id) {
    const uint64_t slot = node_id % geom.nodes_per_sector;
    if (slot == 0) {
      std::fill(page.begin(), page.end(), char{0});
    }

    char *rec = page.data() + slot * geom.node_len;
    const auto &nbrs = graph[node_id];
    if (nbrs.size() > params.max_degree) {
      throw std::invalid_argument("write_disk_layout: node " + std::to_string(node_id) +
                                  " degree " + std::to_string(nbrs.size()) +
                                  " exceeds max_degree " + std::to_string(params.max_degree));
    }
    pack_node_record(rec,
                     vectors + node_id * params.dim,
                     nbrs.data(),
                     static_cast<uint32_t>(nbrs.size()),
                     params.dim);

    const bool last_in_page = (slot == geom.nodes_per_sector - 1);
    const bool last_node = (node_id == params.num_points - 1);
    if (last_in_page || last_node) {
      out.write(page.data(), static_cast<std::streamsize>(geom.page_size));
    }
  }

  out.flush();
  if (!out) {
    throw std::runtime_error("write_disk_layout: write failed for " + path);
  }
}

/**
 * @brief Read and validate the header sector of a disk index file.
 *
 * Validates internal consistency: node_len / nodes_per_sector match the
 * geometry derived from (dim, max_degree), the medoid is in range, and
 * total_file_size matches the actual size of the file on disk.
 *
 * @throws std::runtime_error on any inconsistency or read failure.
 */
inline DiskLayoutHeader read_disk_layout_header(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("read_disk_layout_header: cannot open " + path);
  }
  std::vector<char> buf(kSectorLen, 0);
  in.read(buf.data(), static_cast<std::streamsize>(kSectorLen));
  if (in.gcount() != static_cast<std::streamsize>(kSectorLen)) {
    throw std::runtime_error("read_disk_layout_header: short read (truncated header) for " + path);
  }

  DiskLayoutHeader h;
  h.num_points = detail::get_u64(buf.data(), header_offset::kNumPoints);
  h.dim = detail::get_u64(buf.data(), header_offset::kDim);
  h.medoid = detail::get_u32(buf.data(), header_offset::kMedoid);
  h.max_degree = detail::get_u32(buf.data(), header_offset::kMaxDegree);
  h.node_len = detail::get_u64(buf.data(), header_offset::kNodeLen);
  h.nodes_per_sector = detail::get_u64(buf.data(), header_offset::kNodesPerSector);
  h.total_file_size = detail::get_u64(buf.data(), header_offset::kTotalFileSize);

  // --- Consistency checks ---
  if (h.num_points == 0 || h.dim == 0) {
    throw std::runtime_error("read_disk_layout_header: zero num_points/dim in " + path);
  }
  const DiskLayoutGeometry geom = DiskLayoutGeometry::compute(h.dim, h.max_degree);
  if (h.node_len != geom.node_len) {
    throw std::runtime_error("read_disk_layout_header: node_len mismatch (header " +
                             std::to_string(h.node_len) + " vs derived " +
                             std::to_string(geom.node_len) + ")");
  }
  if (h.nodes_per_sector != geom.nodes_per_sector) {
    throw std::runtime_error("read_disk_layout_header: nodes_per_sector mismatch in " + path);
  }
  if (h.medoid >= h.num_points) {
    throw std::runtime_error("read_disk_layout_header: medoid out of range in " + path);
  }
  const uint64_t expected = geom.total_file_size(h.num_points);
  if (h.total_file_size != expected) {
    throw std::runtime_error("read_disk_layout_header: header total_file_size " +
                             std::to_string(h.total_file_size) + " != expected " +
                             std::to_string(expected));
  }
  std::error_code ec;
  const auto actual = std::filesystem::file_size(path, ec);
  if (ec) {
    throw std::runtime_error("read_disk_layout_header: cannot stat " + path);
  }
  if (static_cast<uint64_t>(actual) != h.total_file_size) {
    throw std::runtime_error("read_disk_layout_header: on-disk size " + std::to_string(actual) +
                             " != header total_file_size " + std::to_string(h.total_file_size));
  }
  return h;
}

}  // namespace alaya::diskann
