// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Reuse the atomic-publish helpers and the bit-pattern finiteness helpers
// from disk_flat_builder.hpp's `detail` namespace (write_all_fsync, fsync_dir,
// rename_no_replace, TmpDirGuard, is_finite_f32 / is_nan_f32 / is_neg_f32).
// disk_flat_builder.hpp is a red-line file — included read-only here.
#include "index/disk/disk_flat_builder.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_writer.hpp"
#include "utils/log.hpp"
#include "utils/metric_type.hpp"
#include "utils/platform.hpp"
#include "utils/platform_fs.hpp"

namespace alaya::disk {

// Build parameters surfaced through the Vamana segment adapter. Defaults
// mirror `alaya::vamana::VamanaBuildParams` so a default-constructed
// `VamanaSegmentBuildParams` produces the same graph as `VamanaBuilder{}`.
struct VamanaSegmentBuildParams {
  uint32_t R = 64;
  uint32_t L = 200;
  float alpha = 1.2F;
  uint32_t num_threads = 0;  // 0 → omp_get_num_procs()
  uint64_t seed = 1234;
};

// VamanaSegmentBuilder — atomically publishes a `seg_<id>/` directory
// containing `manifest.txt`, `ids.u64.bin`, `vectors.f32.bin`, and
// `graph.index`. Wraps `alaya::vamana::VamanaBuilder` + `save_graph`.
//
// v1 metric scope: L2 only. `finish()` rejects `IP` / `COS` before any
// filesystem mutation so direct builder callers can construct the object and
// still observe the atomic-publish precondition.
//
// Lifecycle: construct → add_batch* → finish(seg_dir) → SegmentManifest.
// `finish` is single-shot; subsequent calls throw.
class VamanaSegmentBuilder {
 public:
  VamanaSegmentBuilder(uint32_t dim, MetricType metric, VamanaSegmentBuildParams params)
      : dim_(dim), metric_(metric), params_(params) {
    if (dim == 0) {
      throw std::invalid_argument("VamanaSegmentBuilder: dim must be > 0");
    }
  }

  void add_batch(const float *vectors, const uint64_t *labels, uint64_t n) {
    if (closed_) {
      throw std::runtime_error("VamanaSegmentBuilder: builder is closed (already finished)");
    }
    if (n == 0) {
      return;
    }
    if (vectors == nullptr || labels == nullptr) {
      throw std::invalid_argument(
          "VamanaSegmentBuilder: add_batch with n>0 requires non-null buffers");
    }
    size_t vec_components = 0;
    if (alaya_mul_overflow(static_cast<size_t>(n), static_cast<size_t>(dim_), &vec_components)) {
      throw std::invalid_argument("VamanaSegmentBuilder: n * dim overflows size_t (n=" +
                                  std::to_string(n) + ", dim=" + std::to_string(dim_) + ")");
    }
    // NaN / Inf rejection up-front so the builder cannot ship a corrupted
    // graph: `simd::l2_sqr` returns NaN if either operand has a NaN component
    // and the build's argmin would silently pick that node as the medoid.
    const uint64_t row_offset = pending_labels_.size();
    for (uint64_t r = 0; r < n; ++r) {
      for (uint32_t c = 0; c < dim_; ++c) {
        const float v = vectors[r * dim_ + c];
        if (!detail::is_finite_f32(v)) {
          const uint64_t global_row = row_offset + r;
          if (detail::is_nan_f32(v)) {
            throw std::invalid_argument("VamanaSegmentBuilder: NaN component at row " +
                                        std::to_string(global_row) + " position " +
                                        std::to_string(c));
          }
          const std::string sign = detail::is_neg_f32(v) ? "-Inf" : "+Inf";
          throw std::invalid_argument("VamanaSegmentBuilder: Inf component at row " +
                                      std::to_string(global_row) + " position " +
                                      std::to_string(c) + " (" + sign + ")");
        }
      }
    }
    pending_vectors_.insert(pending_vectors_.end(), vectors, vectors + vec_components);
    pending_labels_.insert(pending_labels_.end(), labels, labels + n);
  }

  auto finish(const std::filesystem::path &seg_dir) -> SegmentManifest {
    if (closed_) {
      throw std::runtime_error("VamanaSegmentBuilder: builder is closed (already finished)");
    }
    ensure_metric_supported();
    if (pending_labels_.empty()) {
      throw std::runtime_error(
          "VamanaSegmentBuilder: finish called with zero rows (count=0 is rejected by manifest)");
    }

    // Step 1 — path validation. Same shape as DiskFlatBuilder::finish.
    const auto parent = seg_dir.parent_path();
    if (parent.empty()) {
      throw std::invalid_argument("VamanaSegmentBuilder: seg_dir must have a parent: " +
                                  seg_dir.string());
    }
    const std::string seg_basename = seg_dir.filename().string();
    if (!detail::is_valid_segment_id(seg_basename)) {
      throw std::invalid_argument(
          "VamanaSegmentBuilder: seg_dir basename must match ^seg_[0-9]{8}$: '" + seg_basename +
          "'");
    }
    {
      std::error_code ec;
      if (std::filesystem::exists(seg_dir, ec)) {
        throw std::runtime_error("VamanaSegmentBuilder: target segment already exists: " +
                                 seg_dir.string());
      }
    }
    {
      std::error_code ec;
      if (!std::filesystem::is_directory(parent, ec) || ec) {
        throw std::runtime_error("VamanaSegmentBuilder: parent directory does not exist: " +
                                 parent.string());
      }
    }

    // Step 2 — tmp directory. The pid+steady_clock suffix avoids collisions
    // when the same caller retries `finish` after a transient error.
    const auto pid = ::alaya::platform::get_pid();
    const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string tmp_name =
        ".tmp_" + seg_basename + "_" + std::to_string(pid) + "_" + std::to_string(ts);
    const auto tmp_dir = parent / tmp_name;
    {
      std::error_code ec;
      if (std::filesystem::exists(tmp_dir, ec)) {
        throw std::runtime_error(
            "VamanaSegmentBuilder: stale tmp directory already exists (refusing to overwrite): " +
            tmp_dir.string());
      }
      std::filesystem::create_directories(tmp_dir, ec);
      if (ec) {
        throw std::runtime_error("VamanaSegmentBuilder: mkdir tmp failed: " + tmp_dir.string() +
                                 ": " + ec.message());
      }
    }
    detail::TmpDirGuard guard(tmp_dir);

    // Step 3 — data files. Both files written + fsync'd before graph build
    // so a build failure leaves them recoverable under the tmp dir for
    // post-mortem inspection (the guard's destructor will still clean up).
    detail::write_all_fsync(tmp_dir / "ids.u64.bin",
                            pending_labels_.data(),
                            pending_labels_.size() * sizeof(uint64_t));
    detail::write_all_fsync(tmp_dir / "vectors.f32.bin",
                            pending_vectors_.data(),
                            pending_vectors_.size() * sizeof(float));

    // Step 4 — graph build. VamanaBuilder borrows `pending_vectors_.data()`
    // for the duration of build(); the buffer outlives the builder.
    alaya::vamana::VamanaBuildParams vp;
    vp.R = params_.R;
    vp.L = params_.L;
    vp.alpha = params_.alpha;
    vp.num_threads = params_.num_threads;
    vp.seed = params_.seed;
    alaya::vamana::VamanaBuilder gb(pending_vectors_.data(), pending_labels_.size(), dim_, vp);
    gb.build();

    // Step 5 — graph file. `save_graph` does NOT fsync the file; we open it
    // RDONLY and fsync explicitly so the publish is durable. The header's
    // `max_observed_degree` field is set to `params_.R` matching the existing
    // convention from `build_dispatch.hpp` and the writer header comment.
    const auto graph_path = tmp_dir / "graph.index";
    alaya::vamana::save_graph(gb.graph(), graph_path, params_.R, gb.medoid());
    fsync_regular_file(graph_path);

    // Step 6 — manifest.
    SegmentManifest manifest{};
    manifest.version = kManifestVersion;
    manifest.segment_id = seg_basename;
    manifest.index_type = DiskIndexType::Vamana;
    manifest.metric = metric_;
    manifest.dim = dim_;
    manifest.count = pending_labels_.size();
    manifest.ids_file = "ids.u64.bin";
    manifest.vectors_file = "vectors.f32.bin";
    manifest.x_extras["x_graph_file"] = "graph.index";
    manifest.x_extras["x_R"] = std::to_string(params_.R);
    manifest.x_extras["x_L"] = std::to_string(params_.L);
    manifest.x_extras["x_alpha"] = std::to_string(params_.alpha);
    manifest.x_extras["x_seed"] = std::to_string(params_.seed);
    manifest.x_extras["x_medoid"] = std::to_string(gb.medoid());
    manifest.save(tmp_dir / "manifest.txt");

    // Step 7 — atomic publish. fsync(tmp), rename, disarm guard, parent fsync.
    detail::fsync_dir(tmp_dir);
    detail::rename_no_replace(tmp_dir, seg_dir);
    guard.disarm();
    closed_ = true;

    // Post-rename parent fsync is durability-only — the rename has already
    // committed visibility. Same warning-not-throw policy as DiskFlatBuilder
    // so the caller can advance `next_segment_id` instead of being stuck.
    try {
      detail::fsync_dir(parent);
    } catch (const std::exception &e) {
      LOG_WARN("VamanaSegmentBuilder: post-rename parent fsync failed (durability only): {}",
               e.what());
    }

    return manifest;
  }

 private:
  auto ensure_metric_supported() const -> void {
    if (metric_ == MetricType::L2) {
      return;
    }
    const std::string m =
        (metric_ == MetricType::IP) ? "ip" : ((metric_ == MetricType::COS) ? "cos" : "unknown");
    throw std::runtime_error("VamanaSegmentBuilder: metric '" + m +
                             "' not implemented in v1 (vamana adapter v1 supports L2 only)");
  }

  static auto fsync_regular_file(const std::filesystem::path &path) -> void {
    ::alaya::platform::sync_file_or_throw(path);
  }

  uint32_t dim_;
  MetricType metric_;
  VamanaSegmentBuildParams params_;
  bool closed_ = false;
  std::vector<float> pending_vectors_;
  std::vector<uint64_t> pending_labels_;
};

}  // namespace alaya::disk
