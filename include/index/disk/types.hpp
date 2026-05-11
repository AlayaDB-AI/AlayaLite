// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace alaya::disk {

// NOLINTBEGIN
enum class DiskIndexType {
  Flat,
  Vamana,
  Laser,
};
// NOLINTEND

struct DiskSearchOptions {
  uint32_t top_k = 10;
  uint32_t ef = 100;
  uint32_t beam_width = 4;
  bool exact_rerank = true;
};

// Distance contract by metric (smaller-is-better in all three):
//   L2  squared L2 distance between query and stored vector
//   IP  negative inner product, -Sum(q_i * v_i)
//   COS stored vectors and query are L2-normalized internally before
//       scoring; distance is the negative inner product of the
//       normalized pair, -Sum(qn_i * vn_i)
// label is the external label supplied at build time, never the
// internal row index.
struct DiskSearchHit {
  uint64_t label;
  float distance;
};

class SegmentSearcher {
 public:
  virtual ~SegmentSearcher() = default;

  virtual auto search(const float *query, const DiskSearchOptions &opts) const
      -> std::vector<DiskSearchHit> = 0;
  virtual auto size() const -> uint64_t = 0;
  virtual auto dim() const -> uint32_t = 0;
  virtual auto type() const -> DiskIndexType = 0;
};

constexpr auto index_type_to_string(DiskIndexType t) noexcept -> std::string_view {
  switch (t) {
    case DiskIndexType::Flat:
      return "disk_flat";
    case DiskIndexType::Vamana:
      return "disk_vamana";
    case DiskIndexType::Laser:
      return "disk_laser";
  }
  return {};
}

inline auto index_type_from_string(std::string_view s) -> DiskIndexType {
  if (s == "disk_flat") {
    return DiskIndexType::Flat;
  }
  if (s == "disk_vamana") {
    return DiskIndexType::Vamana;
  }
  if (s == "disk_laser") {
    return DiskIndexType::Laser;
  }
  throw std::invalid_argument(std::string("unknown disk index_type string: ") + std::string(s));
}

}  // namespace alaya::disk
