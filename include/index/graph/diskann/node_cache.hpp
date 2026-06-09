// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file node_cache.hpp
 * @brief BFS node cache for the DiskANN disk index.
 *
 * At build time a BFS from the medoid collects the hottest
 * @c ceil(cache_ratio * num_points) nodes (those near the entry point, which
 * every search traverses). Their full records (@c [coords | n_nbrs | nbr_ids],
 * byte-identical to the on-disk layout) are serialized to @c cache_ids.bin and
 * @c cache_nodes.bin. At load time the records are read into memory and indexed
 * for O(1) lookup, so a cache hit during beam search costs zero disk I/O.
 *
 * The cache is immutable after generate()/load(); concurrent search threads
 * only borrow read-only pointers into it.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "index/graph/diskann/disk_layout.hpp"

namespace alaya::diskann {

/**
 * @brief In-memory cache of hot node records selected by BFS from the medoid.
 *
 * Internally the id->record index stores byte offsets into @c node_data_ (not
 * raw pointers), so the structure stays valid across vector growth and moves;
 * lookup() resolves an offset to a borrowed const pointer.
 */
class NodeCache {
 public:
  NodeCache() = default;

  /**
   * @brief Select and pack the BFS cache from an in-memory graph + vectors.
   *
   * @param graph        Adjacency lists (graph[i] = neighbor ids of node i).
   * @param vectors      Row-major num_points*dim float32 coordinates.
   * @param medoid       BFS root (entry point); always included first.
   * @param num_points   Number of graph nodes.
   * @param dim          Vector dimension.
   * @param max_degree   Graph degree bound (defines node_len).
   * @param cache_ratio  Fraction of nodes to cache; count = ceil(ratio * N),
   *                     capped at N. ratio <= 0 produces an empty cache.
   */
  void generate(const std::vector<std::vector<uint32_t>> &graph,
                const float *vectors,
                uint32_t medoid,
                uint64_t num_points,
                uint64_t dim,
                uint32_t max_degree,
                double cache_ratio) {
    if (graph.size() != num_points) {
      throw std::invalid_argument("NodeCache::generate: graph size != num_points");
    }
    if (num_points == 0 || dim == 0) {
      throw std::invalid_argument("NodeCache::generate: num_points/dim must be > 0");
    }
    if (medoid >= num_points) {
      throw std::invalid_argument("NodeCache::generate: medoid out of range");
    }

    const DiskLayoutGeometry geom = DiskLayoutGeometry::compute(dim, max_degree);
    node_len_ = geom.node_len;

    uint64_t target = 0;
    if (cache_ratio > 0.0) {
      target = static_cast<uint64_t>(std::ceil(cache_ratio * static_cast<double>(num_points)));
      if (target > num_points) {
        target = num_points;
      }
    }

    ids_.clear();
    map_.clear();
    node_data_.clear();
    if (target == 0) {
      return;
    }

    // BFS from medoid, collecting nodes in dequeue (BFS) order.
    std::vector<uint8_t> visited(num_points, 0);
    std::queue<uint32_t> bfs;
    bfs.push(medoid);
    visited[medoid] = 1;
    ids_.reserve(target);
    while (!bfs.empty() && ids_.size() < target) {
      const uint32_t u = bfs.front();
      bfs.pop();
      ids_.push_back(u);
      for (const uint32_t v : graph[u]) {
        if (v < num_points && visited[v] == 0) {
          visited[v] = 1;
          bfs.push(v);
        }
      }
    }

    // Pack records (zero-initialized so unused neighbor slots are zero-filled).
    node_data_.assign(ids_.size() * node_len_, 0);
    map_.reserve(ids_.size() * 2);
    for (size_t i = 0; i < ids_.size(); ++i) {
      const uint32_t id = ids_[i];
      char *rec = node_data_.data() + i * node_len_;
      const auto &nbrs = graph[id];
      const uint32_t n_nbrs =
          static_cast<uint32_t>(nbrs.size() > max_degree ? max_degree : nbrs.size());
      pack_node_record(rec, vectors + static_cast<uint64_t>(id) * dim, nbrs.data(), n_nbrs, dim);
      map_[id] = i * node_len_;
    }
  }

  // --- Persistence ---------------------------------------------------------

  /// Write cache_ids.bin and cache_nodes.bin.
  void save(const std::string &ids_path, const std::string &nodes_path) const {
    const uint64_t count = ids_.size();
    {
      std::ofstream out(ids_path, std::ios::binary | std::ios::trunc);
      if (!out) {
        throw std::runtime_error("NodeCache::save: cannot open " + ids_path);
      }
      out.write(reinterpret_cast<const char *>(&count), sizeof(count));
      if (count > 0) {
        out.write(reinterpret_cast<const char *>(ids_.data()),
                  static_cast<std::streamsize>(count * sizeof(uint32_t)));
      }
      if (!out) {
        throw std::runtime_error("NodeCache::save: write failed for " + ids_path);
      }
    }
    {
      std::ofstream out(nodes_path, std::ios::binary | std::ios::trunc);
      if (!out) {
        throw std::runtime_error("NodeCache::save: cannot open " + nodes_path);
      }
      out.write(reinterpret_cast<const char *>(&count), sizeof(count));
      out.write(reinterpret_cast<const char *>(&node_len_), sizeof(node_len_));
      if (!node_data_.empty()) {
        out.write(node_data_.data(), static_cast<std::streamsize>(node_data_.size()));
      }
      if (!out) {
        throw std::runtime_error("NodeCache::save: write failed for " + nodes_path);
      }
    }
  }

  /// Read cache_ids.bin and cache_nodes.bin and rebuild the lookup index.
  void load(const std::string &ids_path, const std::string &nodes_path) {
    uint64_t id_count = 0;
    {
      std::ifstream in(ids_path, std::ios::binary);
      if (!in) {
        throw std::runtime_error("NodeCache::load: cannot open " + ids_path);
      }
      in.read(reinterpret_cast<char *>(&id_count), sizeof(id_count));
      if (!in) {
        throw std::runtime_error("NodeCache::load: short read (ids count) " + ids_path);
      }
      ids_.assign(id_count, 0);
      if (id_count > 0) {
        in.read(reinterpret_cast<char *>(ids_.data()),
                static_cast<std::streamsize>(id_count * sizeof(uint32_t)));
        if (!in) {
          throw std::runtime_error("NodeCache::load: short read (ids) " + ids_path);
        }
      }
    }

    uint64_t node_count = 0;
    {
      std::ifstream in(nodes_path, std::ios::binary);
      if (!in) {
        throw std::runtime_error("NodeCache::load: cannot open " + nodes_path);
      }
      in.read(reinterpret_cast<char *>(&node_count), sizeof(node_count));
      in.read(reinterpret_cast<char *>(&node_len_), sizeof(node_len_));
      if (!in) {
        throw std::runtime_error("NodeCache::load: short read (nodes header) " + nodes_path);
      }
      if (node_count != id_count) {
        throw std::runtime_error("NodeCache::load: ids/nodes count mismatch (" +
                                 std::to_string(id_count) + " vs " + std::to_string(node_count) +
                                 ")");
      }
      node_data_.assign(node_count * node_len_, 0);
      if (!node_data_.empty()) {
        in.read(node_data_.data(), static_cast<std::streamsize>(node_data_.size()));
        if (!in) {
          throw std::runtime_error("NodeCache::load: short read (node data) " + nodes_path);
        }
      }
    }

    map_.clear();
    map_.reserve(ids_.size() * 2);
    for (size_t i = 0; i < ids_.size(); ++i) {
      map_[ids_[i]] = i * node_len_;
    }
  }

  // --- Runtime lookup ------------------------------------------------------

  /**
   * @brief O(1) cache lookup.
   * @return Pointer to the node's @c node_len -byte record, or nullptr on miss.
   *         The returned pointer is read-only and owned by this cache.
   */
  [[nodiscard]] const char *lookup(uint32_t node_id) const {
    const auto it = map_.find(node_id);
    if (it == map_.end()) {
      return nullptr;
    }
    return node_data_.data() + it->second;
  }

  // --- Accessors -----------------------------------------------------------

  [[nodiscard]] uint64_t size() const { return ids_.size(); }
  [[nodiscard]] uint64_t node_len() const { return node_len_; }
  [[nodiscard]] const std::vector<uint32_t> &ids() const { return ids_; }

 private:
  uint64_t node_len_ = 0;
  std::vector<uint32_t> ids_;                   // cached node ids, BFS order
  std::vector<char> node_data_;                 // size() * node_len_ bytes
  std::unordered_map<uint32_t, uint64_t> map_;  // node_id -> byte offset in node_data_
};

}  // namespace alaya::diskann
