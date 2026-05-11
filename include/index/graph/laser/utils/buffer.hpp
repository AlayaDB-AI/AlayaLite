// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file buffer.hpp
 * @brief Sorted buffer implementations for graph-based k-NN search.
 *
 * Provides two buffer types:
 * - SearchBuffer: Beam search frontier (tracks visited/unvisited candidates)
 * - ResultBuffer: Final k-NN results storage
 *
 * Both use binary search insertion to maintain sorted order by distance.
 */

#pragma once

#include <cstddef>
#include <set>
#include <vector>

#include "index/graph/laser/common.hpp"
#include "index/graph/laser/utils/memory.hpp"

namespace alaya::laser::buffer {

/**
 * @brief Sorted linear buffer for beam search candidates.
 *
 * Maintains candidates sorted by distance, with visited/unvisited tracking.
 * The high bit of PID marks whether a candidate has been processed.
 */
class SearchBuffer {
 private:
  std::vector<Candidate<float>, memory::AlignedAllocator<Candidate<float>>> data_;
  size_t size_ = 0, cur_ = 0, capacity_;

  [[nodiscard]] auto binary_search(float dist) const {
    size_t lo = 0;
    size_t len = size_;
    size_t half;
    while (len > 1) {
      half = len >> 1;
      len -= half;
      lo += static_cast<size_t>(data_[lo + half - 1].distance < dist) * half;
    }
    return (lo < size_ && data_[lo].distance < dist) ? lo + 1 : lo;
  }

  static void set_checked(PID &data_id) { data_id |= (1 << 31); }

  [[nodiscard]] static auto is_checked(PID data_id) -> bool {
    return static_cast<bool>(data_id >> 31);
  }

 public:
  SearchBuffer() = default;

  explicit SearchBuffer(size_t capacity) : data_(capacity + 1), capacity_(capacity) {}

  // insert a data point into buffer
  void insert(PID data_id, float dist) {
    size_t lo = binary_search(dist);
    std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Candidate<float>));
    data_[lo] = Candidate<float>(data_id, dist);
    size_ += static_cast<size_t>(size_ < capacity_);
    cur_ = lo < cur_ ? lo : cur_;
  }

  [[nodiscard]] auto is_full(float dist) const -> bool {
    return size_ == capacity_ && dist > data_[size_ - 1].distance;
  }

  // get closest unchecked data point
  PID pop() {
    PID cur_id = data_[cur_].id;
    set_checked(data_[cur_].id);
    ++cur_;
    while (cur_ < size_ && is_checked(data_[cur_].id)) {
      ++cur_;
    }
    return cur_id;
  }

  void clear() {
    size_ = 0;
    cur_ = 0;
  }

  [[nodiscard]] auto next_id() const { return data_[cur_].id; }

  [[nodiscard]] auto has_next() const -> bool { return cur_ < size_; }

  void resize(size_t new_size) {
    this->capacity_ = new_size;
    data_ =
        std::vector<Candidate<float>, memory::AlignedAllocator<Candidate<float>>>(capacity_ + 1);
  }
};

/**
 * @brief Sorted buffer to store final k-NN search results.
 */
class ResultBuffer {
 public:
  explicit ResultBuffer(size_t capacity)
      : ids_(capacity + 1), distances_(capacity + 1), capacity_(capacity) {}

  void insert(PID data_id, float dist) {
    if (size_ == capacity_ && dist > distances_[size_ - 1]) {
      return;
    }
    size_t lo = binary_search(dist);
    std::memmove(&ids_[lo + 1], &ids_[lo], (size_ - lo) * sizeof(PID));
    ids_[lo] = data_id;
    std::memmove(&distances_[lo + 1], &distances_[lo], (size_ - lo) * sizeof(float));
    distances_[lo] = dist;
    size_ += static_cast<size_t>(size_ < capacity_);
  }

  [[nodiscard]] auto is_full() const -> bool { return size_ == capacity_; }

  const std::vector<PID, memory::AlignedAllocator<PID>> &ids() { return ids_; }

  void copy_results(PID *knn) const { std::copy(ids_.begin(), ids_.end() - 1, knn); }

 private:
  std::vector<PID, memory::AlignedAllocator<PID>> ids_;
  std::vector<float, memory::AlignedAllocator<float>> distances_;
  size_t size_ = 0, capacity_;

  [[nodiscard]] auto binary_search(float dist) const -> size_t {
    size_t lo = 0;
    size_t len = size_;
    size_t half;
    while (len > 1) {
      half = len >> 1;
      len -= half;
      lo += static_cast<size_t>(distances_[lo + half - 1] < dist) * half;
    }
    return (lo < size_ && distances_[lo] < dist) ? lo + 1 : lo;
  }
};
}  // namespace alaya::laser::buffer
