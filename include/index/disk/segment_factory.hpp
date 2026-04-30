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

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <stdexcept>
#include <string>
#include "index/disk/disk_flat_builder.hpp"
#include "index/disk/disk_flat_searcher.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "index/disk/vamana_segment_builder.hpp"
#include "index/disk/vamana_segment_searcher.hpp"

namespace alaya::disk {

// engine_supported_v1: v1 capability gate.
//   Flat   → true
//   Vamana → true  (registered via the Vamana adapter; metric scope is L2-only,
//                   surfaced through the engine's own runtime_error rather than
//                   through this gate)
//   Laser  → false (factory unsupported branch throws on use)
[[nodiscard]] constexpr auto engine_supported_v1(DiskIndexType type) noexcept -> bool {
  switch (type) {
    case DiskIndexType::Flat:
      return true;
    case DiskIndexType::Vamana:
      return true;
    case DiskIndexType::Laser:
      return false;
  }
  return false;
}

namespace detail {

// Single source of truth for the unsupported-engine error string. Message MUST
// contain BOTH the lowercase engine identifier from index_type_to_string() AND
// the literal substring "not implemented in v1" — the existing disk-collection
// scenarios pin both.
[[noreturn]] inline auto throw_unsupported_engine(DiskIndexType type) -> void {
  throw std::runtime_error(std::string("DiskSegmentFactory: engine '") +
                           std::string(index_type_to_string(type)) +
                           "' not implemented in v1");
}

}  // namespace detail

// create_segment_from_pending: drive an engine's builder, atomically publish
// segment files under `seg_dir`, and return a ready-to-use `SegmentSearcher`.
//
// v1 routing:
//   Flat   → DiskFlatBuilder + DiskFlatSegmentSearcher
//   Vamana → VamanaSegmentBuilder + VamanaSegmentSearcher (L2 only — IP/COS
//            surface through the engine's own runtime_error before any
//            filesystem mutation)
//   Laser  → throws (no files created at seg_dir)
//
// The factory dispatches on `col_manifest.index_type`; `seg_dir` MUST satisfy
// the format-level constraints enforced by the underlying builder (basename
// matches `^seg_[0-9]{8}$`, parent exists, target does not yet exist).
[[nodiscard]] inline auto create_segment_from_pending(const std::filesystem::path &seg_dir,
                                                      const CollectionManifest &col_manifest,
                                                      const float *vectors,
                                                      const uint64_t *labels,
                                                      uint64_t n_rows)
    -> std::shared_ptr<SegmentSearcher> {
  if (!engine_supported_v1(col_manifest.index_type)) {
    // Throw BEFORE creating any files at seg_dir.
    detail::throw_unsupported_engine(col_manifest.index_type);
  }
  std::shared_ptr<SegmentSearcher> searcher;
  switch (col_manifest.index_type) {
    case DiskIndexType::Flat: {
      DiskFlatBuilder builder(static_cast<uint32_t>(col_manifest.dim), col_manifest.metric);
      builder.add_batch(vectors, labels, n_rows);
      (void)builder.finish(seg_dir);
      searcher = std::make_shared<DiskFlatSegmentSearcher>(seg_dir);
      break;
    }
    case DiskIndexType::Vamana: {
      // Build params hard-coded to VamanaBuilder defaults in v1 — a future
      // change can plumb them through col_manifest.x_extras without touching
      // this branch. The builder rejects IP/COS in its constructor, before
      // any filesystem mutation.
      VamanaSegmentBuilder builder(static_cast<uint32_t>(col_manifest.dim),
                                   col_manifest.metric, VamanaSegmentBuildParams{});
      builder.add_batch(vectors, labels, n_rows);
      (void)builder.finish(seg_dir);
      searcher = std::make_shared<VamanaSegmentSearcher>(seg_dir);
      break;
    }
    case DiskIndexType::Laser:
      // engine_supported_v1 already filtered Laser; the gate above threw
      // before reaching this switch. Hitting this case means the gate was
      // bypassed — surface a loud invariant failure rather than fall through.
      detail::throw_unsupported_engine(col_manifest.index_type);
  }
  // Defence-in-depth: should never fire — engine_supported_v1 plus the
  // engine-specific dispatch above guarantee a Flat or Vamana searcher.
  // A type mismatch here means a future branch was added without going
  // through the gate.
  if (!engine_supported_v1(searcher->type())) {
    throw std::runtime_error(
        "DiskSegmentFactory: invariant violated — searcher engine is not registered in v1");
  }
  return searcher;
}

// load_segment_from_manifest: parse `seg_dir/manifest.txt` to discover the
// engine, then open the matching SegmentSearcher subclass.
//
// v1 routing:
//   Flat   → DiskFlatSegmentSearcher
//   Vamana → VamanaSegmentSearcher
//   Laser  → throws
[[nodiscard]] inline auto load_segment_from_manifest(const std::filesystem::path &seg_dir)
    -> std::shared_ptr<SegmentSearcher> {
  const auto sm = SegmentManifest::load(seg_dir / "manifest.txt");
  if (!engine_supported_v1(sm.index_type)) {
    detail::throw_unsupported_engine(sm.index_type);
  }
  std::shared_ptr<SegmentSearcher> searcher;
  switch (sm.index_type) {
    case DiskIndexType::Flat:
      searcher = std::make_shared<DiskFlatSegmentSearcher>(seg_dir);
      break;
    case DiskIndexType::Vamana:
      searcher = std::make_shared<VamanaSegmentSearcher>(seg_dir);
      break;
    case DiskIndexType::Laser:
      detail::throw_unsupported_engine(sm.index_type);
  }
  if (!engine_supported_v1(searcher->type())) {
    throw std::runtime_error(
        "DiskSegmentFactory: invariant violated — searcher engine is not registered in v1");
  }
  return searcher;
}

// assert_engine_supported_v1: shared throw site for `DiskCollection` ctor /
// open() so the v1-gate exception message is produced by the factory rather
// than duplicated in DiskCollection. Spec D2 pins the dual-substring contract.
inline auto assert_engine_supported_v1(DiskIndexType type) -> void {
  if (!engine_supported_v1(type)) {
    detail::throw_unsupported_engine(type);
  }
}

}  // namespace alaya::disk
