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
#include <utility>
#include <vector>

// VamanaReader — load a DiskANN-compatible Vamana `.index` file produced by
// `alaya::vamana::save_graph` (see `vamana_writer.hpp`) back into the same
// in-memory shape returned by `VamanaBuilder::graph()`.
//
// On-disk byte layout (native byte order, no padding) — must stay in sync
// with `vamana_writer.hpp:29-44`. Any change here must also update the
// writer header layout in lockstep:
//
//   offset 0   size 8   uint64_t expected_file_size  (total bytes incl header)
//   offset 8   size 4   uint32_t max_observed_degree (= R)
//   offset 12  size 4   uint32_t start               (medoid id)
//   offset 16  size 8   uint64_t frozen_pts          (must be 0 in v1)
//   offset 24  ...      per-node records in id order 0..N-1:
//                         uint32_t k
//                         uint32_t neighbors[k]
//
// All structural validation is performed at construction time (D3 in
// design.md). After the constructor returns successfully, the reader's
// graph is known to satisfy every invariant documented in the
// `vamana-reader` spec.

namespace alaya::vamana {

class VamanaReader {
 public:
  // Load and validate a Vamana `.index` file.
  //
  // The constructor performs all I/O and validation in a single pass.
  // On success the reader owns an in-memory adjacency matching
  // `VamanaBuilder::graph()`. On any structural failure it throws
  // `std::runtime_error` whose message identifies the offending byte
  // offset, node id, or field as appropriate.
  //
  // Layout consumed by this constructor — must stay in sync with
  // `vamana_writer.hpp:29-44`:
  //
  //   offset 0   size 8   uint64_t expected_file_size
  //   offset 8   size 4   uint32_t max_observed_degree
  //   offset 12  size 4   uint32_t start
  //   offset 16  size 8   uint64_t frozen_pts
  //   offset 24  ...      per-node (uint32_t k, uint32_t neighbors[k]) for ids 0..N-1
  //
  // See `vamana_writer.hpp` for the corresponding write path.
  explicit VamanaReader(const std::filesystem::path &path) {
    if (!std::filesystem::exists(path)) {
      throw std::runtime_error("VamanaReader: file does not exist: " + path.string());
    }
    if (!std::filesystem::is_regular_file(path)) {
      throw std::runtime_error("VamanaReader: path is not a regular file: " + path.string());
    }

    const std::uintmax_t actual_size_raw = std::filesystem::file_size(path);
    const uint64_t actual_size = static_cast<uint64_t>(actual_size_raw);

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
      throw std::runtime_error("VamanaReader: cannot open file: " + path.string());
    }

    // Reject undersized files up-front so the error names the actual size
    // rather than surfacing as a generic stream-EOF later.
    if (actual_size < kHeaderSize) {
      throw std::runtime_error("VamanaReader: header truncated; actual size " +
                               std::to_string(actual_size) + " < " + std::to_string(kHeaderSize) +
                               " bytes (path: " + path.string() + ")");
    }

    in.read(reinterpret_cast<char *>(&expected_file_size_), sizeof(uint64_t));
    in.read(reinterpret_cast<char *>(&max_degree_), sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&start_), sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&frozen_pts_), sizeof(uint64_t));
    if (!in.good()) {
      throw std::runtime_error("VamanaReader: stream error reading header from " + path.string());
    }

    if (actual_size != expected_file_size_) {
      if (actual_size > expected_file_size_) {
        throw std::runtime_error(
            "VamanaReader: trailing bytes beyond declared end; expected_file_size=" +
            std::to_string(expected_file_size_) + ", actual_size=" + std::to_string(actual_size) +
            " (path: " + path.string() + ")");
      }
      throw std::runtime_error("VamanaReader: file truncated; expected_file_size=" +
                               std::to_string(expected_file_size_) + ", actual_size=" +
                               std::to_string(actual_size) + " (path: " + path.string() + ")");
    }

    // Header-field invariants. `start_` is validated below, after the
    // per-node loop tells us `num_nodes_`.
    if (max_degree_ == 0) {
      throw std::runtime_error(
          "VamanaReader: max_observed_degree is 0; a degree-0 graph cannot be searched (path: " +
          path.string() + ")");
    }
    if (frozen_pts_ != 0) {
      throw std::runtime_error("VamanaReader: frozen_pts=" + std::to_string(frozen_pts_) +
                               " but v1 only supports frozen_pts=0 (path: " + path.string() + ")");
    }

    // Per-node loop: every read is bounds-checked against the records-region
    // byte budget so mid-record truncation surfaces with the offending file
    // offset rather than as a stream-EOF.
    const uint64_t records_total_bytes = expected_file_size_ - kHeaderSize;

    // Each healthy record is ≥ 8 bytes (4 for k, 4 for ≥ 1 neighbor — k=0
    // is rejected below), so records_total_bytes / 8 is a safe upper bound
    // for num_nodes and avoids per-node reallocations of `graph_`.
    const size_t reserve_hint = static_cast<size_t>(records_total_bytes / 8U);
    graph_.reserve(reserve_hint);

    uint64_t consumed = 0;
    while (consumed < records_total_bytes) {
      if (consumed + sizeof(uint32_t) > records_total_bytes) {
        throw std::runtime_error("VamanaReader: mid-record truncation reading k at byte offset " +
                                 std::to_string(kHeaderSize + consumed) +
                                 " (path: " + path.string() + ")");
      }

      uint32_t k = 0;
      in.read(reinterpret_cast<char *>(&k), sizeof(uint32_t));
      if (!in.good()) {
        throw std::runtime_error("VamanaReader: stream error reading k at byte offset " +
                                 std::to_string(kHeaderSize + consumed) +
                                 " (path: " + path.string() + ")");
      }
      consumed += sizeof(uint32_t);

      const uint32_t node_id = static_cast<uint32_t>(graph_.size());

      if (k == 0) {
        throw std::runtime_error("VamanaReader: node " + std::to_string(node_id) +
                                 " has zero out-degree (k=0); v1 rejects zero-degree nodes");
      }
      if (k > max_degree_) {
        throw std::runtime_error("VamanaReader: node " + std::to_string(node_id) +
                                 " has k=" + std::to_string(k) +
                                 " exceeding max_observed_degree=" + std::to_string(max_degree_));
      }

      const uint64_t neighbor_bytes = static_cast<uint64_t>(k) * sizeof(uint32_t);
      if (consumed + neighbor_bytes > records_total_bytes) {
        throw std::runtime_error(
            "VamanaReader: mid-record truncation; node " + std::to_string(node_id) +
            " claims k=" + std::to_string(k) + " neighbors (" + std::to_string(neighbor_bytes) +
            " bytes) but only " + std::to_string(records_total_bytes - consumed) +
            " bytes remain at offset " + std::to_string(kHeaderSize + consumed) +
            " (path: " + path.string() + ")");
      }

      std::vector<uint32_t> adj(k);
      in.read(reinterpret_cast<char *>(adj.data()), static_cast<std::streamsize>(neighbor_bytes));
      if (!in.good()) {
        throw std::runtime_error("VamanaReader: stream error reading neighbors for node " +
                                 std::to_string(node_id) + " (path: " + path.string() + ")");
      }
      consumed += neighbor_bytes;

      graph_.push_back(std::move(adj));
    }

    // Defensive post-condition: the bounds checks above already prevent
    // overruns, but keeping this assertion makes a future refactor that
    // breaks the byte-budget logic fail here rather than silently.
    if (consumed != records_total_bytes) {
      throw std::runtime_error("VamanaReader: size mismatch; consumed " + std::to_string(consumed) +
                               " bytes but expected_file_size - " + std::to_string(kHeaderSize) +
                               " = " + std::to_string(records_total_bytes) +
                               " (path: " + path.string() + ")");
    }

    num_nodes_ = graph_.size();

    // Empty graph: `start` must reference a real id, so 0 nodes is invalid.
    if (num_nodes_ == 0) {
      throw std::runtime_error(
          "VamanaReader: graph has no nodes (records region is empty); "
          "Vamana requires at least one node because `start` must reference a real id");
    }

    // Neighbor-range and self-loop validation has to wait until after the
    // parse loop because `num_nodes` is not known mid-stream.
    for (size_t i = 0; i < graph_.size(); ++i) {
      for (uint32_t n : graph_[i]) {
        if (n >= num_nodes_) {
          throw std::runtime_error("VamanaReader: node " + std::to_string(i) +
                                   " has out-of-range neighbor id " + std::to_string(n) +
                                   " (num_nodes=" + std::to_string(num_nodes_) + ")");
        }
        if (n == static_cast<uint32_t>(i)) {
          throw std::runtime_error(
              "VamanaReader: node " + std::to_string(i) +
              " has a self-loop (neighbor id == node id == " + std::to_string(n) + ")");
        }
      }
    }

    if (static_cast<size_t>(start_) >= num_nodes_) {
      throw std::runtime_error("VamanaReader: start=" + std::to_string(start_) +
                               " is out of range; num_nodes=" + std::to_string(num_nodes_) +
                               " (path: " + path.string() + ")");
    }
  }

  // Non-copyable, non-movable in v1 (design.md D3) — the reader is created
  // in-place and the deleted move ops keep it as a local-only resource.
  VamanaReader(const VamanaReader &) = delete;
  VamanaReader &operator=(const VamanaReader &) = delete;
  VamanaReader(VamanaReader &&) = delete;
  VamanaReader &operator=(VamanaReader &&) = delete;

  uint32_t max_degree() const { return max_degree_; }
  uint32_t start() const { return start_; }
  uint64_t frozen_pts() const { return frozen_pts_; }
  size_t num_nodes() const { return num_nodes_; }
  const std::vector<std::vector<uint32_t>> &graph() const { return graph_; }

 private:
  static constexpr uint64_t kHeaderSize = 24;

  uint64_t expected_file_size_ = 0;
  uint32_t max_degree_ = 0;
  uint32_t start_ = 0;
  uint64_t frozen_pts_ = 0;
  size_t num_nodes_ = 0;
  std::vector<std::vector<uint32_t>> graph_;
};

}  // namespace alaya::vamana
