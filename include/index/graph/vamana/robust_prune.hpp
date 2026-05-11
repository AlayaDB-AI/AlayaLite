// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace alaya::vamana {

// Neighbor tuple: id + distance + expansion flag. Byte-compatible with
// DiskANN's struct layout and ordering; ported verbatim so the builder can
// follow DiskANN's pool/queue semantics without translation shims.
struct Neighbor {
  uint32_t id = 0;
  float distance = 0.0f;
  bool expanded = false;

  Neighbor() = default;
  Neighbor(uint32_t id_in, float distance_in) : id(id_in), distance(distance_in) {}

  bool operator<(const Neighbor &other) const {
    return distance < other.distance || (distance == other.distance && id < other.id);
  }
  bool operator==(const Neighbor &other) const { return id == other.id; }
};

// Bounded-capacity priority queue of Neighbors kept in ascending distance
// order. Mirrors DiskANN's `NeighborPriorityQueue` (include/neighbor.h):
// supports O(log n) insertion with dedup-by-id and a cursor (`cur_`) that
// tracks the first unexpanded entry in distance order.
class NeighborPriorityQueue {
 public:
  NeighborPriorityQueue() = default;
  explicit NeighborPriorityQueue(size_t capacity) : capacity_(capacity), data_(capacity + 1) {}

  void reserve(size_t capacity) {
    if (capacity + 1 > data_.size()) {
      data_.resize(capacity + 1);
    }
    capacity_ = capacity;
  }

  void insert(const Neighbor &nbr) {
    if (size_ == capacity_ && data_[size_ - 1] < nbr) {
      return;
    }
    size_t lo = 0;
    size_t hi = size_;
    while (lo < hi) {
      size_t mid = (lo + hi) >> 1;
      if (nbr < data_[mid]) {
        hi = mid;
      } else if (data_[mid].id == nbr.id) {
        return;  // dedup
      } else {
        lo = mid + 1;
      }
    }
    if (lo < capacity_) {
      std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor));
    }
    data_[lo] = Neighbor(nbr.id, nbr.distance);
    if (size_ < capacity_) {
      size_++;
    }
    if (lo < cur_) {
      cur_ = lo;
    }
  }

  Neighbor closest_unexpanded() {
    data_[cur_].expanded = true;
    size_t pre = cur_;
    while (cur_ < size_ && data_[cur_].expanded) {
      cur_++;
    }
    return data_[pre];
  }

  bool has_unexpanded_node() const { return cur_ < size_; }
  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }

  Neighbor &operator[](size_t i) { return data_[i]; }
  const Neighbor &operator[](size_t i) const { return data_[i]; }

  void clear() {
    size_ = 0;
    cur_ = 0;
  }

 private:
  size_t size_ = 0;
  size_t capacity_ = 0;
  size_t cur_ = 0;
  std::vector<Neighbor> data_;
};

// occlude_list — α-scaled RNG pruning over a sorted candidate pool.
//
// DiskANN semantics (src/index.cpp:1072-1165): iterate the pool in ascending
// distance order, selecting candidates greedily. For each already-selected
// entry `i` and each later candidate `j`, update
//   occlude_factor[j] = max(occlude_factor[j], pool[j].distance / d(i, j))
// and skip j when occlude_factor[j] > cur_alpha. cur_alpha ramps 1.0, 1.2,
// 1.44, ... until it exceeds `alpha`, giving α > 1 a second chance to
// retain long-range edges that α = 1 would occlude.
//
// Preconditions:
//   - `pool` is sorted ascending by Neighbor::operator<.
//   - `pool` contains no self-id (caller strips self).
//   - `result` is caller-owned; occlude_list appends to it without clearing.
//   - `occlude_factor` is scratch storage; this function resizes it.
//
// Postcondition: `pool` is truncated to min(size, maxc) (matches DiskANN).
template <typename DistFn>
inline void occlude_list(uint32_t location,
                         std::vector<Neighbor> &pool,
                         float alpha,
                         uint32_t degree,
                         uint32_t maxc,
                         std::vector<uint32_t> &result,
                         std::vector<float> &occlude_factor,
                         DistFn &&dist_fn) {
  if (pool.empty()) {
    return;
  }
  if (pool.size() > maxc) {
    pool.resize(maxc);
  }
  occlude_factor.clear();
  occlude_factor.insert(occlude_factor.end(), pool.size(), 0.0f);

  float cur_alpha = 1.0f;
  while (cur_alpha <= alpha && result.size() < degree) {
    for (auto iter = pool.begin(); result.size() < degree && iter != pool.end(); ++iter) {
      size_t i = static_cast<size_t>(iter - pool.begin());
      if (occlude_factor[i] > cur_alpha) {
        continue;
      }
      occlude_factor[i] = std::numeric_limits<float>::max();
      if (iter->id != location) {
        result.push_back(iter->id);
      }
      for (auto iter2 = iter + 1; iter2 != pool.end(); ++iter2) {
        size_t j = static_cast<size_t>(iter2 - pool.begin());
        if (occlude_factor[j] > alpha) {
          continue;
        }
        float djk = dist_fn(iter2->id, iter->id);
        occlude_factor[j] = (djk == 0.0f) ? std::numeric_limits<float>::max()
                                          : std::max(occlude_factor[j], iter2->distance / djk);
      }
    }
    cur_alpha *= 1.2f;
  }
}

// prune_neighbors — sort pool, invoke occlude_list, emit id list.
// Mirrors DiskANN's `Index::prune_neighbors` (src/index.cpp:1174).
template <typename DistFn>
inline void prune_neighbors(uint32_t location,
                            std::vector<Neighbor> &pool,
                            float alpha,
                            uint32_t range,
                            uint32_t maxc,
                            std::vector<uint32_t> &pruned_list,
                            std::vector<float> &occlude_factor_scratch,
                            DistFn &&dist_fn) {
  pruned_list.clear();
  if (pool.empty()) {
    return;
  }
  std::sort(pool.begin(), pool.end());
  pruned_list.reserve(range);
  occlude_list(location,
               pool,
               alpha,
               range,
               maxc,
               pruned_list,
               occlude_factor_scratch,
               std::forward<DistFn>(dist_fn));
}

}  // namespace alaya::vamana
