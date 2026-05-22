// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "index/graph/vamana/vamana_greedy_search.hpp"
#include "index/graph/vamana/vamana_reader.hpp"
#include "storage/mmap_file.hpp"
#include "utils/metric_type.hpp"
#include "utils/platform.hpp"

namespace alaya::disk {

namespace detail {

inline auto is_finite_query_f32(float value) -> bool {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(value));
  return (bits & 0x7F800000U) != 0x7F800000U;
}

inline auto is_nan_query_f32(float value) -> bool {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(value));
  return (bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0;
}

inline auto is_neg_query_f32(float value) -> bool {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(value));
  return (bits & 0x80000000U) != 0;
}

}  // namespace detail

// VamanaSegmentSearcher — adapter that opens a Vamana `seg_<id>/` directory
// and answers `DiskSearchOptions` queries by forwarding to the
// non-polymorphic `VamanaGreedySearch`.
//
// All I/O happens in the constructor (manifest parse, mmap of vectors / ids,
// graph load, greedy-search wiring). After construction, `search()` performs
// no `open(2)`, no manifest parsing, no mmap rebuild — the per-neighbor
// distance loop runs inside `VamanaGreedySearch` whose L2 kernel is held as
// a `simd::L2SqrFunc` function pointer.
//
// v1: L2 only. The constructor rejects `IP` / `COS` manifests with the
// same dual-substring contract used by the rest of the disk subsystem.
class VamanaSegmentSearcher : public SegmentSearcher {
 public:
  explicit VamanaSegmentSearcher(const std::filesystem::path &seg_dir)
      : manifest_(SegmentManifest::load(seg_dir / "manifest.txt")) {
    // Step 1 — engine + shape validation against the parsed manifest.
    if (manifest_.index_type != DiskIndexType::Vamana) {
      throw std::runtime_error(
          "VamanaSegmentSearcher: manifest index_type is not disk_vamana (got '" +
          std::string(index_type_to_string(manifest_.index_type)) + "') in segment " +
          seg_dir.string());
    }
    if (manifest_.dim == 0 || manifest_.count == 0) {
      throw std::runtime_error("VamanaSegmentSearcher: manifest dim or count is zero in segment " +
                               seg_dir.string());
    }
    if (manifest_.dim > kMaxDim) {
      throw std::runtime_error("VamanaSegmentSearcher: manifest dim exceeds uint32 (" +
                               std::to_string(manifest_.dim) + ") in segment " + seg_dir.string());
    }
    if (manifest_.count > kMaxCount) {
      throw std::runtime_error("VamanaSegmentSearcher: manifest count exceeds uint32 (" +
                               std::to_string(manifest_.count) + ") in segment " +
                               seg_dir.string());
    }
    // Step 2 — v1 metric scope. IP / COS surface here with the dual-substring
    // contract before any mmap so an unsupported-metric manifest produces no
    // observable side effects (matching the spec's pre-IO promise).
    if (manifest_.metric != MetricType::L2) {
      const std::string m = (manifest_.metric == MetricType::IP)    ? "ip"
                            : (manifest_.metric == MetricType::COS) ? "cos"
                                                                    : "unknown";
      throw std::runtime_error("VamanaSegmentSearcher: metric '" + m +
                               "' not implemented in v1 (vamana adapter v1 supports L2 only) in "
                               "segment " +
                               seg_dir.string());
    }

    // Step 3 — vectors mmap. MMapFile opens O_RDONLY | O_NOFOLLOW and rejects
    // non-regular files, satisfying disk-collection's symlink-rejection contract.
    const uint64_t expected_vec_bytes = compute_expected_vectors_bytes();
    const uint64_t expected_ids_bytes = compute_expected_ids_bytes();

    vectors_mmap_ = alaya::storage::MMapFile(seg_dir / manifest_.vectors_file);
    if (vectors_mmap_.size() != expected_vec_bytes) {
      throw std::runtime_error("VamanaSegmentSearcher: vectors file size mismatch — expected " +
                               std::to_string(expected_vec_bytes) + " (count×dim×4) but got " +
                               std::to_string(vectors_mmap_.size()) + " for " +
                               (seg_dir / manifest_.vectors_file).string());
    }

    // Step 4 — ids mmap.
    ids_mmap_ = alaya::storage::MMapFile(seg_dir / manifest_.ids_file);
    if (ids_mmap_.size() != expected_ids_bytes) {
      throw std::runtime_error("VamanaSegmentSearcher: ids file size mismatch — expected " +
                               std::to_string(expected_ids_bytes) + " (count×8) but got " +
                               std::to_string(ids_mmap_.size()) + " for " +
                               (seg_dir / manifest_.ids_file).string());
    }

    // Step 5 — graph load. The graph file path lives in the manifest's
    // forward-compat extension namespace under `x_graph_file` to keep the
    // engine-agnostic `SegmentManifest` schema unchanged.
    auto graph_it = manifest_.x_extras.find("x_graph_file");
    if (graph_it == manifest_.x_extras.end()) {
      throw std::runtime_error("VamanaSegmentSearcher: x_graph_file missing in manifest for " +
                               seg_dir.string());
    }
    if (graph_it->second.empty()) {
      throw std::runtime_error(
          "VamanaSegmentSearcher: x_graph_file value is empty in manifest "
          "for " +
          seg_dir.string());
    }
    if (!detail::is_valid_basename(graph_it->second)) {
      throw std::runtime_error(
          "VamanaSegmentSearcher: x_graph_file must be a basename (no '/', '..', NUL): '" +
          graph_it->second + "' in segment " + seg_dir.string());
    }

    // VamanaReader's constructor validates header well-formedness, file size
    // vs `expected_file_size`, neighbor range, no self-loops, and `start <
    // num_nodes`. Any structural fault throws here before greedy_search_ is
    // wired up.
    const auto graph_path = seg_dir / graph_it->second;
    {
      std::error_code ec;
      if (std::filesystem::is_symlink(graph_path, ec)) {
        throw std::runtime_error("VamanaSegmentSearcher: graph file is a symlink: " +
                                 graph_path.string());
      }
    }
    reader_ = std::make_unique<alaya::vamana::VamanaReader>(graph_path);

    // Step 6 — cross-validate manifest count vs graph num_nodes. A mismatch
    // means a corrupt segment (e.g. ids/vectors written from one batch but a
    // graph from a different one) — bail with both numbers in the message.
    if (manifest_.count != reader_->num_nodes()) {
      throw std::runtime_error(
          "VamanaSegmentSearcher: manifest count (" + std::to_string(manifest_.count) +
          ") disagrees with graph num_nodes (" + std::to_string(reader_->num_nodes()) +
          ") in segment " + seg_dir.string());
    }

    // Step 7 — greedy search wiring. Borrows `reader_` and the vectors mmap
    // pointer — both must outlive `greedy_search_`, enforced by member
    // declaration order (greedy_search_ declared LAST, destructed FIRST).
    greedy_search_ =
        std::make_unique<alaya::vamana::VamanaGreedySearch>(*reader_,
                                                            static_cast<const float *>(
                                                                vectors_mmap_.data()),
                                                            static_cast<uint32_t>(manifest_.dim));
  }

  VamanaSegmentSearcher(const VamanaSegmentSearcher &) = delete;
  auto operator=(const VamanaSegmentSearcher &) -> VamanaSegmentSearcher & = delete;
  VamanaSegmentSearcher(VamanaSegmentSearcher &&) = delete;
  auto operator=(VamanaSegmentSearcher &&) -> VamanaSegmentSearcher & = delete;
  ~VamanaSegmentSearcher() override = default;

  auto search(const float *query, const DiskSearchOptions &opts) const
      -> std::vector<DiskSearchHit> override {
    if (opts.top_k == 0) {
      throw std::invalid_argument("VamanaSegmentSearcher: top_k must be > 0");
    }
    if (opts.ef == 0) {
      throw std::invalid_argument("VamanaSegmentSearcher: ef must be > 0");
    }
    if (query == nullptr) {
      throw std::invalid_argument("VamanaSegmentSearcher: query must not be null");
    }
    const uint32_t d = static_cast<uint32_t>(manifest_.dim);
    for (uint32_t c = 0; c < d; ++c) {
      const float v = query[c];
      if (!detail::is_finite_query_f32(v)) {
        const std::string kind =
            detail::is_nan_query_f32(v) ? "NaN" : (detail::is_neg_query_f32(v) ? "-Inf" : "+Inf");
        throw std::invalid_argument(
            "VamanaSegmentSearcher: non-finite query component at "
            "position " +
            std::to_string(c) + " (" + kind + ")");
      }
    }
    const auto count = static_cast<uint32_t>(reader_->num_nodes());
    const auto effective_top_k = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(opts.top_k), static_cast<uint64_t>(count)));
    auto effective_ef = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(opts.ef), static_cast<uint64_t>(count)));
    effective_ef = std::max(effective_ef, effective_top_k);
    // Forward to VamanaGreedySearch — no additional file open, no manifest
    // parse, no mmap rebuild, no metric-string-to-enum lookup. The per-neighbor
    // distance kernel inside greedy search is a `simd::L2SqrFunc` function
    // pointer; the only virtual call on this path is the SegmentSearcher
    // boundary (one per query per segment).
    auto greedy_hits = greedy_search_->search(query, effective_top_k, effective_ef);

    // Internal-id → external-label conversion: a single sequential sweep over
    // the result vector after greedy search returns. The ids mmap is accessed
    // exactly once per hit, with no re-entry into VamanaGreedySearch.
    const auto *ids = static_cast<const uint64_t *>(ids_mmap_.data());
    std::vector<DiskSearchHit> out;
    out.reserve(greedy_hits.size());
    for (const auto &hit : greedy_hits) {
      out.push_back(DiskSearchHit{ids[hit.id], hit.distance});
    }
    return out;
  }

  auto size() const -> uint64_t override { return manifest_.count; }
  auto dim() const -> uint32_t override { return static_cast<uint32_t>(manifest_.dim); }
  auto type() const -> DiskIndexType override { return DiskIndexType::Vamana; }

 private:
  static constexpr uint64_t kMaxDim = static_cast<uint64_t>(UINT32_MAX);
  static constexpr uint64_t kMaxCount = static_cast<uint64_t>(UINT32_MAX);

  auto compute_expected_vectors_bytes() const -> uint64_t {
    uint64_t cd = 0;
    if (alaya_mul_overflow(manifest_.count, manifest_.dim, &cd)) {
      throw std::runtime_error(
          "VamanaSegmentSearcher: manifest dim×count exceeds uint64 range (overflow)");
    }
    uint64_t bytes = 0;
    if (alaya_mul_overflow(cd, static_cast<uint64_t>(sizeof(float)), &bytes)) {
      throw std::runtime_error(
          "VamanaSegmentSearcher: manifest dim×count×4 exceeds uint64 range (overflow)");
    }
    return bytes;
  }

  auto compute_expected_ids_bytes() const -> uint64_t {
    uint64_t bytes = 0;
    if (alaya_mul_overflow(manifest_.count, static_cast<uint64_t>(sizeof(uint64_t)), &bytes)) {
      throw std::runtime_error(
          "VamanaSegmentSearcher: manifest count×8 exceeds uint64 range (overflow)");
    }
    return bytes;
  }

  // Declaration order matters for destruction safety:
  //   manifest_      → destroyed last (no dependency)
  //   vectors_mmap_  → owns the float buffer; greedy_search_ borrows pointer
  //   ids_mmap_      → owns the label buffer; search() reads it directly
  //   reader_        → owns graph adjacency; greedy_search_ borrows by ref
  //   greedy_search_ → declared LAST → destroyed FIRST so the borrowed
  //                    references survive its lifetime
  SegmentManifest manifest_;
  alaya::storage::MMapFile vectors_mmap_;
  alaya::storage::MMapFile ids_mmap_;
  std::unique_ptr<alaya::vamana::VamanaReader> reader_;
  std::unique_ptr<alaya::vamana::VamanaGreedySearch> greedy_search_;
};

}  // namespace alaya::disk
