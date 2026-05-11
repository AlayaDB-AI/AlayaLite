// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "index/disk/disk_flat_builder.hpp"  // bit-pattern helpers in alaya::disk::detail
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "simd/distance_ip.hpp"
#include "simd/distance_l2.hpp"
#include "storage/mmap_file.hpp"
#include "utils/metric_type.hpp"
#include "utils/platform.hpp"

namespace alaya::disk {

static_assert(std::endian::native == std::endian::little,
              "DiskCollection v1 supports only little-endian hosts");

namespace detail {

inline auto compute_expected_vectors_bytes(uint64_t count, uint64_t dim) -> uint64_t {
  uint64_t cd = 0;
  if (__builtin_mul_overflow(count, dim, &cd)) {
    throw std::runtime_error(
        "DiskFlatSegmentSearcher: manifest dim×count exceeds uint64 range (overflow)");
  }
  uint64_t bytes = 0;
  if (__builtin_mul_overflow(cd, static_cast<uint64_t>(sizeof(float)), &bytes)) {
    throw std::runtime_error(
        "DiskFlatSegmentSearcher: manifest dim×count×4 exceeds uint64 range (overflow)");
  }
  return bytes;
}

inline auto compute_expected_ids_bytes(uint64_t count) -> uint64_t {
  uint64_t bytes = 0;
  if (__builtin_mul_overflow(count, static_cast<uint64_t>(sizeof(uint64_t)), &bytes)) {
    throw std::runtime_error(
        "DiskFlatSegmentSearcher: manifest count×8 exceeds uint64 range (overflow)");
  }
  return bytes;
}

struct AlignedFreeDeleter {
  void operator()(void *p) const noexcept { alaya_aligned_free_impl(p); }
};

using AlignedFloatBuffer = std::unique_ptr<float, AlignedFreeDeleter>;

inline auto allocate_aligned_floats(size_t count) -> AlignedFloatBuffer {
  constexpr size_t kAlign = 64;
  const size_t bytes = ((count * sizeof(float) + kAlign - 1) / kAlign) * kAlign;
  void *p = alaya_aligned_alloc_impl(bytes, kAlign);
  if (p == nullptr) {
    throw std::bad_alloc{};
  }
  return AlignedFloatBuffer{static_cast<float *>(p)};
}

}  // namespace detail

class DiskFlatSegmentSearcher : public SegmentSearcher {
 public:
  explicit DiskFlatSegmentSearcher(const std::filesystem::path &segment_dir)
      : manifest_(SegmentManifest::load(segment_dir / "manifest.txt")) {
    if (manifest_.index_type != DiskIndexType::Flat) {
      throw std::runtime_error("DiskFlatSegmentSearcher: manifest index_type is not disk_flat: " +
                               segment_dir.string());
    }
    if (manifest_.dim == 0 || manifest_.count == 0) {
      throw std::runtime_error("DiskFlatSegmentSearcher: manifest dim or count is zero");
    }
    const uint64_t expected_vec_bytes =
        detail::compute_expected_vectors_bytes(manifest_.count, manifest_.dim);
    const uint64_t expected_ids_bytes = detail::compute_expected_ids_bytes(manifest_.count);
    if (manifest_.dim > kMaxDim) {
      throw std::runtime_error("DiskFlatSegmentSearcher: manifest dim exceeds uint32 (" +
                               std::to_string(manifest_.dim) + ")");
    }

    ids_mmap_ = alaya::storage::MMapFile(segment_dir / manifest_.ids_file);
    vectors_mmap_ = alaya::storage::MMapFile(segment_dir / manifest_.vectors_file);

    if (ids_mmap_.size() != expected_ids_bytes) {
      throw std::runtime_error("DiskFlatSegmentSearcher: ids file size mismatch — expected " +
                               std::to_string(expected_ids_bytes) + " (count×8) but got " +
                               std::to_string(ids_mmap_.size()) + " for " + manifest_.ids_file);
    }
    if (vectors_mmap_.size() != expected_vec_bytes) {
      throw std::runtime_error("DiskFlatSegmentSearcher: vectors file size mismatch — expected " +
                               std::to_string(expected_vec_bytes) + " (count×dim×4) but got " +
                               std::to_string(vectors_mmap_.size()) + " for " +
                               manifest_.vectors_file);
    }
  }

  DiskFlatSegmentSearcher(const DiskFlatSegmentSearcher &) = delete;
  auto operator=(const DiskFlatSegmentSearcher &) -> DiskFlatSegmentSearcher & = delete;
  DiskFlatSegmentSearcher(DiskFlatSegmentSearcher &&) = delete;
  auto operator=(DiskFlatSegmentSearcher &&) -> DiskFlatSegmentSearcher & = delete;
  ~DiskFlatSegmentSearcher() override = default;

  auto search(const float *query, const DiskSearchOptions &opts) const
      -> std::vector<DiskSearchHit> override {
    if (opts.top_k == 0) {
      throw std::invalid_argument("DiskFlatSegmentSearcher: top_k must be > 0");
    }
    if (query == nullptr) {
      throw std::invalid_argument("DiskFlatSegmentSearcher: query must not be null");
    }
    const uint32_t d = static_cast<uint32_t>(manifest_.dim);

    for (uint32_t c = 0; c < d; ++c) {
      const float v = query[c];
      if (!detail::is_finite_f32(v)) {
        if (detail::is_nan_f32(v)) {
          throw std::invalid_argument("DiskFlatSegmentSearcher: NaN query component at position " +
                                      std::to_string(c));
        }
        const std::string sign = detail::is_neg_f32(v) ? "-Inf" : "+Inf";
        throw std::invalid_argument("DiskFlatSegmentSearcher: Inf query component at position " +
                                    std::to_string(c) + " (" + sign + ")");
      }
    }

    const float *effective_query = query;
    detail::AlignedFloatBuffer normalized_query;

    if (manifest_.metric == MetricType::COS) {
      double sum_sq = 0.0;
      for (uint32_t c = 0; c < d; ++c) {
        const double v = static_cast<double>(query[c]);
        sum_sq += v * v;
      }
      if (sum_sq == 0.0) {
        throw std::invalid_argument(
            "DiskFlatSegmentSearcher: zero-magnitude query under COS metric");
      }
      const double inv_norm = 1.0 / std::sqrt(sum_sq);
      if (!detail::is_finite_f64(inv_norm)) {
        throw std::runtime_error(
            "DiskFlatSegmentSearcher: non-finite inverse norm of query (defence in depth)");
      }
      normalized_query = detail::allocate_aligned_floats(d);
      for (uint32_t c = 0; c < d; ++c) {
        normalized_query.get()[c] = static_cast<float>(static_cast<double>(query[c]) * inv_norm);
      }
      effective_query = normalized_query.get();
    }

    // Hoist SIMD dispatch out of the per-row loop (D12: no virtual / dispatch
    // call inside the hot loop). The simd::l2_sqr<float,float> wrapper resolves
    // a function pointer via get_l2_sqr_func() on every call; lifting the
    // pointer out of the loop removes one indirection per row.
    using KernelFn = float (*)(const float *__restrict, const float *__restrict, size_t);
    const KernelFn kernel = (manifest_.metric == MetricType::L2)
                                ? static_cast<KernelFn>(simd::get_l2_sqr_func())
                                : static_cast<KernelFn>(simd::get_ip_sqr_func());

    const auto *vectors = static_cast<const float *>(vectors_mmap_.data());
    const auto *ids = static_cast<const uint64_t *>(ids_mmap_.data());
    const uint64_t count = manifest_.count;
    const uint64_t k = std::min<uint64_t>(opts.top_k, count);

    auto cmp = [](const DiskSearchHit &a, const DiskSearchHit &b) {
      if (a.distance != b.distance) {
        return a.distance < b.distance;
      }
      return a.label < b.label;
    };

    std::vector<DiskSearchHit> heap;
    heap.reserve(k);

    for (uint64_t i = 0; i < count; ++i) {
      const float dist = kernel(effective_query, vectors + i * d, d);
      const DiskSearchHit hit{ids[i], dist};
      if (heap.size() < k) {
        heap.push_back(hit);
        std::push_heap(heap.begin(), heap.end(), cmp);
      } else if (cmp(hit, heap.front())) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        heap.back() = hit;
        std::push_heap(heap.begin(), heap.end(), cmp);
      }
    }

    std::sort(heap.begin(), heap.end(), cmp);
    return heap;
  }

  auto size() const -> uint64_t override { return manifest_.count; }
  auto dim() const -> uint32_t override { return static_cast<uint32_t>(manifest_.dim); }
  auto type() const -> DiskIndexType override { return DiskIndexType::Flat; }

  // Read-only view of the segment's external labels (size == size()).
  auto labels() const -> const uint64_t * {
    return static_cast<const uint64_t *>(ids_mmap_.data());
  }

 private:
  static constexpr uint64_t kMaxDim = static_cast<uint64_t>(UINT32_MAX);

  SegmentManifest manifest_;
  alaya::storage::MMapFile ids_mmap_;
  alaya::storage::MMapFile vectors_mmap_;
};

}  // namespace alaya::disk
