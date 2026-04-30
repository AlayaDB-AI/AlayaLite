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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "index/disk/disk_flat_builder.hpp"
#include "index/disk/disk_flat_searcher.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "utils/log.hpp"
#include "utils/metric_type.hpp"

namespace alaya::disk {

namespace detail {

inline auto format_segment_id(uint64_t id) -> std::string {
  char buf[16];  // "seg_" + 8 digits + NUL = 13
  std::snprintf(buf, sizeof(buf), "seg_%08llu", static_cast<unsigned long long>(id));
  return std::string(buf);
}

// Standard POSIX rename: atomically replaces destination if it exists.
// Used for collection_manifest.txt updates where the destination intentionally
// exists (segment-publish uses RENAME_NOREPLACE in `disk_flat_builder.hpp`).
inline auto rename_atomic_replace(const std::filesystem::path &from,
                                  const std::filesystem::path &to) -> void {
  if (::rename(from.c_str(), to.c_str()) != 0) {
    int saved = errno;
    throw std::runtime_error("collection_manifest rename failed: " + from.string() + " -> " +
                             to.string() + ": " + std::strerror(saved));
  }
}

// Atomically writes + renames the collection manifest into place. The parent
// directory fsync is split out into a separate step (fsync_collection_dir
// below) because rename atomicity does NOT depend on the parent fsync — the
// new manifest is visible to subsequent opens immediately after rename. The
// parent fsync only promotes durability across a system crash; treating it
// as a soft step allows the in-memory state to commit on rename success.
inline auto publish_collection_manifest_atomic_only(
    const std::filesystem::path &collection_dir, const CollectionManifest &m) -> void {
  const auto pid = static_cast<long long>(::getpid());
  const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::string tmp_name =
      ".tmp_collection_manifest_" + std::to_string(pid) + "_" + std::to_string(ts);
  const auto tmp_path = collection_dir / tmp_name;
  m.save(tmp_path);
  rename_atomic_replace(tmp_path, collection_dir / "collection_manifest.txt");
}

}  // namespace detail

class DiskCollection {
 public:
  // Default soft cap: 512 MiB measured against the D6 formula.
  static constexpr size_t kDefaultMaxPendingBytes = 512ULL * 1024 * 1024;

  // Public constructor: create-only.
  DiskCollection(const std::filesystem::path &path, uint32_t dim, MetricType metric,
                 DiskIndexType index_type) {
    if (dim == 0) {
      throw std::invalid_argument("DiskCollection: dim must be > 0");
    }
    if (metric != MetricType::L2 && metric != MetricType::IP && metric != MetricType::COS) {
      throw std::invalid_argument(
          "DiskCollection: metric must be one of L2, IP, COS (got NONE or unknown)");
    }
    if (index_type != DiskIndexType::Flat) {
      throw std::runtime_error("DiskCollection: index type " +
                               std::string(index_type_to_string(index_type)) +
                               " not implemented in v1");
    }
    {
      std::error_code ec;
      if (std::filesystem::exists(path, ec) || ec) {
        throw std::runtime_error("DiskCollection: target path already exists: " + path.string());
      }
      std::filesystem::create_directories(path / "segments", ec);
      if (ec) {
        throw std::runtime_error("DiskCollection: mkdir failed: " + path.string() + ": " +
                                 ec.message());
      }
    }

    path_ = path;
    manifest_.version = kManifestVersion;
    manifest_.dim = dim;
    manifest_.metric = metric;
    manifest_.index_type = index_type;
    manifest_.next_segment_id = 1;
    manifest_.segment_ids.clear();

    // Constructor publish: atomic write + best-effort durability fsync.
    detail::publish_collection_manifest_atomic_only(path_, manifest_);
    try {
      detail::fsync_dir(path_);
    } catch (const std::exception &e) {
      LOG_WARN("DiskCollection: ctor fsync_dir failed (durability only): {}", e.what());
    }
  }

  static auto open(const std::filesystem::path &path) -> DiskCollection {
    {
      std::error_code ec;
      if (!std::filesystem::exists(path, ec) || ec) {
        throw std::runtime_error("DiskCollection::open: path does not exist: " + path.string());
      }
    }
    DiskCollection col;
    col.path_ = path;
    col.manifest_ = CollectionManifest::load(path / "collection_manifest.txt");
    if (col.manifest_.index_type != DiskIndexType::Flat) {
      throw std::runtime_error("DiskCollection::open: index type " +
                               std::string(index_type_to_string(col.manifest_.index_type)) +
                               " not implemented in v1");
    }
    col.open_listed_segments();
    col.scan_orphans();
    return col;
  }

  DiskCollection(const DiskCollection &) = delete;
  auto operator=(const DiskCollection &) -> DiskCollection & = delete;
  DiskCollection(DiskCollection &&) = default;
  auto operator=(DiskCollection &&) -> DiskCollection & = default;
  ~DiskCollection() = default;

  void add_batch(const float *vectors, const uint64_t *labels, uint64_t n) {
    if (n == 0) {
      return;
    }
    if (vectors == nullptr || labels == nullptr) {
      throw std::invalid_argument("DiskCollection: add_batch with n>0 requires non-null buffers");
    }
    const uint64_t per_row = static_cast<uint64_t>(manifest_.dim) * sizeof(float) +
                             sizeof(uint64_t);
    const uint64_t cap = max_pending_bytes_;

    // Defense in depth: check n * per_row * 2 for overflow before any cap
    // comparison. Without this, a hostile n could wrap to a small value
    // and bypass the cap check. (Final Codex review flagged this as the
    // first of two archive blockers.)
    uint64_t single_batch_bytes = 0;
    if (__builtin_mul_overflow(2ULL, n, &single_batch_bytes) ||
        __builtin_mul_overflow(single_batch_bytes, per_row, &single_batch_bytes)) {
      throw std::runtime_error("DiskCollection: pending size arithmetic overflows uint64 (n=" +
                               std::to_string(n) + ", dim=" + std::to_string(manifest_.dim) + ")");
    }
    if (single_batch_bytes > cap) {
      throw std::runtime_error(
          "DiskCollection: single batch (" + std::to_string(single_batch_bytes) +
          " bytes) exceeds max_pending_bytes (" + std::to_string(cap) +
          " bytes); split the batch or raise max_pending_bytes");
    }

    const uint64_t current_rows = pending_labels_.size();
    const uint64_t current_total = 2ULL * current_rows * per_row;
    uint64_t total_rows = 0;
    uint64_t new_total = 0;
    if (__builtin_add_overflow(current_rows, n, &total_rows) ||
        __builtin_mul_overflow(2ULL, total_rows, &new_total) ||
        __builtin_mul_overflow(new_total, per_row, &new_total)) {
      throw std::runtime_error(
          "DiskCollection: pending+batch arithmetic overflows uint64 (current_rows=" +
          std::to_string(current_rows) + ", n=" + std::to_string(n) + ")");
    }
    if (new_total > cap) {
      const uint64_t remaining = (cap > current_total) ? (cap - current_total) : 0ULL;
      const uint64_t max_addable = (per_row > 0) ? (remaining / (2ULL * per_row)) : 0ULL;
      throw std::runtime_error(
          "DiskCollection: pending buffer would overflow — current=" +
          std::to_string(current_total) + " bytes, cap=" + std::to_string(cap) +
          " bytes, addable_rows_under_dim=" + std::to_string(max_addable));
    }

    // Strong exception safety: reserve both buffers BEFORE any insert. If
    // reserve throws bad_alloc, no state has been mutated. After successful
    // reserve, the inserts cannot allocate again so they can't throw
    // (Codex section-7 med #5).
    pending_vectors_.reserve(pending_vectors_.size() + n * manifest_.dim);
    pending_labels_.reserve(pending_labels_.size() + n);
    pending_vectors_.insert(pending_vectors_.end(), vectors,
                            vectors + n * manifest_.dim);
    pending_labels_.insert(pending_labels_.end(), labels, labels + n);
  }

  void flush() {
    if (pending_labels_.empty()) {
      return;
    }
    // Within-batch label uniqueness.
    std::unordered_set<uint64_t> pending_set;
    pending_set.reserve(pending_labels_.size() * 2);
    for (auto label : pending_labels_) {
      auto [it, inserted] = pending_set.insert(label);
      if (!inserted) {
        throw std::invalid_argument("DiskCollection: duplicate label within pending batch: " +
                                    std::to_string(label));
      }
    }
    // Cross-segment uniqueness — stream each segment's mmap'd ids and probe.
    for (const auto &seg : segments_) {
      auto *flat = dynamic_cast<DiskFlatSegmentSearcher *>(seg.get());
      if (flat == nullptr) {
        continue;
      }
      const uint64_t *labels = flat->labels();
      const uint64_t cnt = flat->size();
      for (uint64_t i = 0; i < cnt; ++i) {
        if (pending_set.contains(labels[i])) {
          throw std::invalid_argument(
              "DiskCollection: duplicate label across segments: " + std::to_string(labels[i]));
        }
      }
    }

    const uint64_t seg_id = manifest_.next_segment_id;
    const std::string seg_basename = detail::format_segment_id(seg_id);
    const auto seg_dir = path_ / "segments" / seg_basename;

    DiskFlatBuilder builder(manifest_.dim, manifest_.metric);
    builder.add_batch(pending_vectors_.data(), pending_labels_.data(), pending_labels_.size());
    auto seg_manifest = builder.finish(seg_dir);
    (void)seg_manifest;
    // Segment is now on disk. Any subsequent failure in this function leaves
    // an orphan segment that will be classified as kind=complete on next open.
    // Eagerly advance in-memory next_segment_id so a retried flush() uses a
    // fresh id rather than colliding with the orphan (Codex section-7 high #1).
    manifest_.next_segment_id = seg_id + 1;

    // Construct the searcher BEFORE updating any other in-memory state.
    // If this throws (e.g., MMapFile open under O_NOFOLLOW fails), the segment
    // is on disk but `segments_` and pending are unchanged; caller can decide
    // whether to retry (after addressing the underlying open failure) — and
    // next_segment_id is already advanced so retry won't collide
    // (Codex section-7 high #2).
    auto searcher = std::make_shared<DiskFlatSegmentSearcher>(seg_dir);

    // Atomic rename of the collection manifest. Once this returns, the segment
    // is officially listed on disk. If this throws, segment is still orphan.
    auto new_manifest = manifest_;
    new_manifest.segment_ids.push_back(seg_basename);
    new_manifest.next_segment_id = seg_id + 1;
    detail::publish_collection_manifest_atomic_only(path_, new_manifest);

    // From this point on, the on-disk state is consistent. Commit in-memory
    // state atomically. Pending is cleared LAST so any throw above leaves it
    // intact for the caller (Codex section-7 high #3 — fsync below is soft).
    manifest_ = std::move(new_manifest);
    segments_.push_back(std::move(searcher));
    pending_vectors_.clear();
    pending_labels_.clear();
    pending_vectors_.shrink_to_fit();
    pending_labels_.shrink_to_fit();

    // Parent-dir fsync is a durability-only step. If it fails, the rename
    // already happened — in-memory state is correct, the only impact is that
    // the rename might not survive a power loss. Log and proceed.
    try {
      detail::fsync_dir(path_);
    } catch (const std::exception &e) {
      LOG_WARN("DiskCollection: collection_manifest fsync_dir failed (durability only): {}",
               e.what());
    }
  }

  auto search(const float *query, const DiskSearchOptions &opts) const
      -> std::vector<DiskSearchHit> {
    if (opts.top_k == 0) {
      throw std::invalid_argument("DiskCollection: top_k must be > 0");
    }
    if (segments_.empty()) {
      return {};
    }

    struct Tagged {
      DiskSearchHit hit;
      uint32_t segment_index;
    };

    std::vector<Tagged> all;
    all.reserve(segments_.size() * opts.top_k);
    for (uint32_t s = 0; s < segments_.size(); ++s) {
      auto seg_hits = segments_[s]->search(query, opts);
      for (auto &h : seg_hits) {
        all.push_back(Tagged{h, s});
      }
    }
    std::sort(all.begin(), all.end(), [](const Tagged &a, const Tagged &b) {
      if (a.hit.distance != b.hit.distance) {
        return a.hit.distance < b.hit.distance;
      }
      if (a.hit.label != b.hit.label) {
        return a.hit.label < b.hit.label;
      }
      return a.segment_index < b.segment_index;
    });

    const size_t k = std::min<size_t>(opts.top_k, all.size());
    std::vector<DiskSearchHit> out;
    out.reserve(k);
    for (size_t i = 0; i < k; ++i) {
      out.push_back(all[i].hit);
    }
    return out;
  }

  // Returns the total number of FLUSHED rows. Pending rows are intentionally
  // excluded — see spec scenario "size() excludes pending rows".
  auto size() const -> uint64_t {
    uint64_t s = 0;
    for (const auto &seg : segments_) {
      s += seg->size();
    }
    return s;
  }

  auto dim() const -> uint32_t { return static_cast<uint32_t>(manifest_.dim); }

  // Tombstone API stubs (v1: not implemented).
  static void mark_deleted(uint64_t /*label*/) {
    throw std::runtime_error("DiskCollection: deletes not implemented in v1");
  }
  static auto is_deleted(uint64_t /*label*/) -> bool { return false; }

 private:
  DiskCollection() = default;

  void open_listed_segments() {
    segments_.reserve(manifest_.segment_ids.size());
    for (const auto &id : manifest_.segment_ids) {
      const auto seg_dir = path_ / "segments" / id;
      // Reject segment directories that are themselves symlinks (D4: a
      // symlink-swap of seg_*/ would otherwise redirect reads outside the
      // collection root — MMapFile's O_NOFOLLOW only protects leaf files).
      std::error_code ec;
      if (std::filesystem::is_symlink(seg_dir, ec)) {
        throw std::runtime_error("DiskCollection: segment directory is a symlink: " +
                                 seg_dir.string());
      }
      auto searcher = std::make_shared<DiskFlatSegmentSearcher>(seg_dir);
      segments_.push_back(std::move(searcher));
    }
  }

  void scan_orphans() {
    const auto segments_dir = path_ / "segments";
    std::error_code ec;
    if (!std::filesystem::is_directory(segments_dir, ec)) {
      return;
    }
    std::set<std::string> listed(manifest_.segment_ids.begin(), manifest_.segment_ids.end());
    uint64_t max_on_disk = 0;
    for (const auto &entry : std::filesystem::directory_iterator(segments_dir, ec)) {
      if (ec) {
        break;
      }
      const auto name = entry.path().filename().string();
      if (name.starts_with(".tmp_")) {
        // Stale tmp from an aborted flush — log as partial and skip.
        LOG_WARN("DiskCollection: orphan tmp dir at {} kind=partial", entry.path().string());
        continue;
      }
      if (!detail::is_valid_segment_id(name)) {
        continue;
      }
      if (listed.contains(name)) {
        // Listed; track its id for next-id calculation.
        const uint64_t id = std::stoull(name.substr(4));
        max_on_disk = std::max(max_on_disk, id);
        continue;
      }
      // Orphan segment. Classify.
      classify_and_log_orphan(entry.path());
      const uint64_t id = std::stoull(name.substr(4));
      max_on_disk = std::max(max_on_disk, id);
    }
    if (max_on_disk + 1 > manifest_.next_segment_id) {
      manifest_.next_segment_id = max_on_disk + 1;
    }
  }

  static void classify_and_log_orphan(const std::filesystem::path &orphan_dir) {
    const auto manifest_path = orphan_dir / "manifest.txt";
    std::error_code ec;
    if (!std::filesystem::exists(manifest_path, ec) || ec) {
      LOG_WARN("DiskCollection: orphan segment at {} kind=partial (no manifest.txt)",
               orphan_dir.string());
      return;
    }
    SegmentManifest sm;
    try {
      sm = SegmentManifest::load(manifest_path);
    } catch (const std::exception &e) {
      LOG_WARN("DiskCollection: orphan segment at {} kind=partial (manifest unparseable: {})",
               orphan_dir.string(), e.what());
      return;
    }
    const auto ids_path = orphan_dir / sm.ids_file;
    const auto vec_path = orphan_dir / sm.vectors_file;
    const auto ids_size_actual = std::filesystem::file_size(ids_path, ec);
    if (ec) {
      LOG_WARN("DiskCollection: orphan segment at {} kind=truncated (ids stat failed)",
               orphan_dir.string());
      return;
    }
    const auto vec_size_actual = std::filesystem::file_size(vec_path, ec);
    if (ec) {
      LOG_WARN("DiskCollection: orphan segment at {} kind=truncated (vectors stat failed)",
               orphan_dir.string());
      return;
    }
    const uint64_t expected_ids = sm.count * sizeof(uint64_t);
    const uint64_t expected_vec = sm.count * sm.dim * sizeof(float);
    if (ids_size_actual != expected_ids || vec_size_actual != expected_vec) {
      LOG_WARN("DiskCollection: orphan segment at {} kind=truncated", orphan_dir.string());
      return;
    }
    LOG_WARN("DiskCollection: orphan segment at {} kind=complete", orphan_dir.string());
  }

  std::filesystem::path path_;
  CollectionManifest manifest_;
  std::vector<std::shared_ptr<SegmentSearcher>> segments_;
  std::vector<float> pending_vectors_;
  std::vector<uint64_t> pending_labels_;
  size_t max_pending_bytes_ = kDefaultMaxPendingBytes;
};

}  // namespace alaya::disk
