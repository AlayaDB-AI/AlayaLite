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
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include "index/disk/disk_flat_builder.hpp"
#include "index/disk/disk_flat_searcher.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "index/disk/vamana_segment_builder.hpp"
#include "index/disk/vamana_segment_searcher.hpp"

#if defined(ALAYA_ENABLE_LASER) && (ALAYA_ENABLE_LASER + 0) != 0 && defined(__linux__)
  #define ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED 1
#else
  #define ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED 0
#endif

#if ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED
  #include "index/disk/laser_segment_importer.hpp"
  #include "index/disk/laser_segment_searcher.hpp"
#endif

namespace alaya::disk {

// engine_supported_v1: v1 capability gate.
//   Flat   → true
//   Vamana → true  (registered via the Vamana adapter; metric scope is L2-only,
//                   surfaced through the engine's own runtime_error rather than
//                   through this gate)
//   Laser  → build/platform gated. v1 supports load/import only when LASER is
//            compiled into a Linux consumer TU; create-from-pending still throws.
[[nodiscard]] constexpr auto engine_supported_v1(DiskIndexType type) noexcept -> bool {
  switch (type) {
    case DiskIndexType::Flat:
      return true;
    case DiskIndexType::Vamana:
      return true;
    case DiskIndexType::Laser:
#if ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED
      return true;
#else
      return false;
#endif
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
                           std::string(index_type_to_string(type)) + "' not implemented in v1");
}

[[noreturn]] inline auto throw_laser_create_not_implemented() -> void {
  throw std::runtime_error(
      "DiskSegmentFactory: engine 'disk_laser' not implemented in v1; use "
      "import_laser_segment for precomputed LASER artifacts");
}

[[noreturn]] inline auto throw_unsupported_import_path(DiskIndexType type) -> void {
  throw std::runtime_error(std::string("DiskSegmentFactory: unsupported import path for engine '") +
                           std::string(index_type_to_string(type)) + "'");
}

#if ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED
inline auto parse_uint32_extra(const CollectionManifest &manifest,
                               const std::string &key,
                               uint32_t default_value) -> uint32_t {
  const auto it = manifest.x_extras.find(key);
  if (it == manifest.x_extras.end()) {
    return default_value;
  }
  try {
    if (it->second.empty() || it->second.front() == '-') {
      throw std::invalid_argument("expected non-negative uint32");
    }
    size_t pos = 0;
    const auto parsed = std::stoull(it->second, &pos, 10);
    if (pos != it->second.size()) {
      throw std::invalid_argument("trailing characters");
    }
    if (parsed > std::numeric_limits<uint32_t>::max()) {
      throw std::out_of_range("value exceeds uint32");
    }
    return static_cast<uint32_t>(parsed);
  } catch (const std::exception &e) {
    throw std::runtime_error("DiskSegmentFactory: invalid Laser import parameter '" + key +
                             "' value '" + it->second + "': " + e.what());
  }
}

inline auto parse_laser_float_extra(const CollectionManifest &manifest,
                                    const std::string &key,
                                    float default_value) -> float {
  const auto it = manifest.x_extras.find(key);
  if (it == manifest.x_extras.end()) {
    return default_value;
  }
  try {
    if (it->second.empty()) {
      throw std::invalid_argument("expected float");
    }
    size_t pos = 0;
    const auto parsed = std::stof(it->second, &pos);
    if (pos != it->second.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return parsed;
  } catch (const std::exception &e) {
    throw std::runtime_error("DiskSegmentFactory: invalid Laser import parameter '" + key +
                             "' value '" + it->second + "': " + e.what());
  }
}

inline auto parse_bool_extra(const CollectionManifest &manifest,
                             const std::string &key,
                             bool default_value) -> bool {
  const auto it = manifest.x_extras.find(key);
  if (it == manifest.x_extras.end()) {
    return default_value;
  }
  if (it->second == "true" || it->second == "1") {
    return true;
  }
  if (it->second == "false" || it->second == "0") {
    return false;
  }
  throw std::runtime_error("DiskSegmentFactory: invalid Laser import parameter '" + key +
                           "' value '" + it->second + "': expected true/false or 1/0");
}

inline auto laser_import_params_from_manifest(const CollectionManifest &manifest)
    -> LaserSegmentImportParams {
  LaserSegmentImportParams params;
  params.R = parse_uint32_extra(manifest, "x_R", params.R);
  params.main_dim = parse_uint32_extra(manifest, "x_main_dim", params.main_dim);
  params.default_ef = parse_uint32_extra(manifest, "x_default_ef", params.default_ef);
  params.default_beam_width =
      parse_uint32_extra(manifest, "x_default_beam_width", params.default_beam_width);
  params.search_dram_budget_gb = parse_laser_float_extra(manifest,
                                                         "x_laser_search_dram_budget_gb",
                                                         params.search_dram_budget_gb);
  params.copy_files = parse_bool_extra(manifest, "x_laser_copy_files", params.copy_files);
  return params;
}
#endif

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
[[nodiscard]] inline auto create_segment_from_pending(
    const std::filesystem::path &seg_dir,
    const CollectionManifest &col_manifest,
    const float *vectors,
    const uint64_t *labels,
    uint64_t n_rows,
    const VamanaSegmentBuildParams &vamana_params = VamanaSegmentBuildParams{})
    -> std::shared_ptr<SegmentSearcher> {
  if (col_manifest.index_type == DiskIndexType::Laser) {
    // v1 has no in-C++ LASER build path. Reject this before the generic engine
    // gate so supported and unsupported configurations get the same API hint.
    detail::throw_laser_create_not_implemented();
  }
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
      // The builder rejects IP/COS in finish(), before any filesystem
      // mutation. Python validates the same policy at construction time so
      // users get a clean boundary error before a collection directory exists.
      if (n_rows < 2) {
        throw std::runtime_error(
            "DiskSegmentFactory: disk_vamana requires at least 2 rows per segment");
      }
      VamanaSegmentBuilder builder(static_cast<uint32_t>(col_manifest.dim),
                                   col_manifest.metric,
                                   vamana_params);
      builder.add_batch(vectors, labels, n_rows);
      (void)builder.finish(seg_dir);
      searcher = std::make_shared<VamanaSegmentSearcher>(seg_dir);
      break;
    }
    case DiskIndexType::Laser:
      detail::throw_laser_create_not_implemented();
  }
  // Defence-in-depth: should never fire — engine_supported_v1 plus the
  // engine-specific dispatch above guarantee a registered searcher.
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
#if ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED
      searcher = std::make_shared<LaserSegmentSearcher>(seg_dir);
      break;
#else
      detail::throw_unsupported_engine(sm.index_type);
#endif
  }
  if (!engine_supported_v1(searcher->type())) {
    throw std::runtime_error(
        "DiskSegmentFactory: invariant violated — searcher engine is not registered in v1");
  }
  return searcher;
}

// import_segment_from_artifacts: v1 import path for precomputed native LASER
// artifacts. Flat and Vamana do not have artifact importers in v1.
[[nodiscard]] inline auto import_segment_from_artifacts(const std::filesystem::path &seg_dir,
                                                        const CollectionManifest &col_manifest,
                                                        const std::filesystem::path &src_dir,
                                                        const uint64_t *labels,
                                                        uint64_t n_rows)
    -> std::shared_ptr<SegmentSearcher> {
  if (col_manifest.index_type != DiskIndexType::Laser) {
    detail::throw_unsupported_import_path(col_manifest.index_type);
  }
  if (!engine_supported_v1(col_manifest.index_type)) {
    detail::throw_unsupported_engine(col_manifest.index_type);
  }

  std::shared_ptr<SegmentSearcher> searcher;
#if ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED
  auto params = detail::laser_import_params_from_manifest(col_manifest);
  LaserSegmentImporter importer(static_cast<uint32_t>(col_manifest.dim),
                                col_manifest.metric,
                                params);
  (void)importer.import_from(src_dir, labels, n_rows, seg_dir);
  searcher = std::make_shared<LaserSegmentSearcher>(seg_dir);
#else
  detail::throw_unsupported_engine(col_manifest.index_type);
#endif

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

#undef ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED
