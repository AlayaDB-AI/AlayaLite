/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <sys/types.h>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>
#include "../index/neighbor.hpp"
#include "memory.hpp"

#if defined(_MSC_VER)
  #include <intrin.h>
#endif

namespace alaya {

/**
 * @brief Portable function to count trailing zero bits in a 64-bit integer.
 *
 * @param x The 64-bit integer. Assumed to be non-zero.
 * @return The number of trailing zero bits.
 */
inline auto count_trailing_zeros(uint64_t x) -> int {
#if defined(_MSC_VER)
  unsigned long index;  // NOLINT(runtime/int)
  _BitScanForward64(&index, x);
  return static_cast<int>(index);
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_ctzll(x);
#else
  // Fallback for other compilers
  int count = 0;
  while ((x & 1) == 0) {
    x >>= 1;
    count++;
  }
  return count;
#endif
}

/**
 * @brief A dynamic bitset implementation using a vector of uint64_t.
 *
 * Pros:
 * - Dynamic size, can be determined at runtime
 * - Efficient memory usage for dense bit sets
 * - Fast set, get, and reset operations
 *
 * Cons:
 * - Slightly slower than std::bitset for small, fixed-size bit sets
 * - Not optimized for very sparse data
 */
class DynamicBitset {
 private:
  std::vector<uint64_t, AlignAlloc<uint64_t>> data_;
  size_t size_;

 public:
  /**
   * @brief Construct a new Dynamic Bitset object
   *
   * @param num_bits The number of bits in the bitset
   */
  explicit DynamicBitset(size_t num_bits) : size_(num_bits) {
    data_.resize((num_bits + 63) / 64, 0);
  }

  /**
   * @brief Set the bit at the specified position
   *
   * @param pos The position of the bit to set
   */
  void set(size_t pos) noexcept { data_[pos >> 6] |= (1ULL << (pos & 63)); }

  /**
   * @brief Get the value of the bit at the specified position
   *
   * @param pos The position of the bit to get
   * @return true if the bit is set, false otherwise
   */
  [[nodiscard]] auto get(size_t pos) const -> bool {
    return (data_[pos >> 6] >> (pos & 63)) & 1ULL;  // NOLINT
  }

  /**
   * @brief Get the pos address object
   *
   * @param pos The position of the bit to get
   * @return void* The address of the bit
   */
  auto get_address(size_t pos) -> void * { return data_.data() + pos / 64; }

  /**
   * @brief Reset the bit at the specified position
   *
   * @param pos The position of the bit to reset
   */
  void reset(size_t pos) { data_[pos / 64] &= ~(1ULL << (pos % 64)); }
};

/**
 * @brief A sparse bitset implementation using an unordered set.
 *
 * Pros:
 * - Extremely memory efficient for very sparse bit sets
 * - Dynamic size
 *
 * Cons:
 * - Slower than dense bitset implementations for most operations
 * - High memory usage for dense bit sets
 * - No cache locality
 */
class SparseBitset {
 private:
  std::unordered_set<size_t> set_bits_;

 public:
  /**
   * @brief Set the bit at the specified position
   *
   * @param pos The position of the bit to set
   */
  void set(size_t pos) { set_bits_.insert(pos); }

  /**
   * @brief Get the value of the bit at the specified position
   *
   * @param pos The position of the bit to get
   * @return true if the bit is set, false otherwise
   */
  auto get(size_t pos) const -> bool { return set_bits_.find(pos) != set_bits_.end(); }

  /**
   * @brief Reset the bit at the specified position
   *
   * @param pos The position of the bit to reset
   */
  void reset(size_t pos) { set_bits_.erase(pos); }
};

/**
 * @brief A hierarchical bitset implementation for efficient "find first set" operations.
 *
 * Pros:
 * - Very fast "find first set" operation
 * - Good performance for large bitsets
 *
 * Cons:
 * - More complex implementation
 * - Slightly higher memory usage than simple bitset
 * - Set and get operations are slightly slower than simple bitset
 */
class HierarchicalBitset {
 private:
  std::vector<uint64_t> data_;
  std::vector<uint64_t> summary_;
  size_t size_;
  static const size_t kBitsPerBlock = 512;
  static const size_t kSummaryBlockSize = 64;

 public:
  /**
   * @brief Construct a new Hierarchical Bitset object
   *
   * @param num_bits The number of bits in the bitset
   */
  explicit HierarchicalBitset(size_t num_bits) : size_(num_bits) {
    data_.resize((num_bits + 63) / 64, 0);
    summary_.resize((data_.size() + 63) / 64, 0);
  }

  /**
   * @brief Set the bit at the specified position
   *
   * @param pos The position of the bit to set
   */
  void set(size_t pos) {
    size_t block = pos / kBitsPerBlock;
    size_t offset = pos % kBitsPerBlock;
    data_[block * 8 + offset / 64] |= (1ULL << (offset % 64));
    summary_[block / kSummaryBlockSize] |= (1ULL << (block % kSummaryBlockSize));
  }

  /**
   * @brief Get the value of the bit at the specified position
   *
   * @param pos The position of the bit to get
   * @return true if the bit is set, false otherwise
   */
  auto get(size_t pos) const -> bool {
    size_t block = pos / kBitsPerBlock;
    size_t offset = pos % kBitsPerBlock;
    return (data_[(block * 8) + (offset / 64)] & (1ULL << (offset % 64))) != 0;
  }

  /**
   * @brief Find the position of the first set bit
   *
   * @return int The position of the first set bit, or -1 if no bit is set
   */
  auto find_first_set() const -> int {
    for (size_t i = 0; i < summary_.size(); ++i) {
      if (summary_[i] == 0) {
        continue;
      }
      size_t block = (i * kSummaryBlockSize) + count_trailing_zeros(summary_[i]);
      for (size_t j = 0; j < 8; ++j) {
        if (data_[(block * 8) + j] == 0) {
          continue;
        }
        return static_cast<int>((block * kBitsPerBlock) + (j * 64) +
                                count_trailing_zeros(data_[block * 8 + j]));
      }
    }
    return -1;
  }
};

// todo test this class.
template <typename DistanceType, typename IDType>
struct LinearPool {
  LinearPool(IDType n, int capacity) : nb_(n), capacity_(capacity), data_(capacity_ + 1), vis_(n) {}

  auto find_bsearch(DistanceType dist) -> int {
    int l = 0;
    int r = size_;
    while (l < r) {
      int mid = (l + r) / 2;
      if (data_[mid].distance_ > dist) {
        r = mid;
      } else {
        l = mid + 1;
      }
    }
    return l;
  }

  auto insert(IDType u, DistanceType dist) -> bool {
    if (size_ == capacity_ && dist >= data_[size_ - 1].distance_) {
      return false;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor<IDType, DistanceType>));
    data_[lo] = {u, dist};
    if (size_ < capacity_) {
      size_++;
    }
    if (static_cast<size_t>(lo) < cur_) {
      cur_ = lo;
    }
    for (size_t i = 0; i < size_; i++) {
      // LOG_INFO("i {} ,dist is {}", data_[i].id_, data_[i].distance_);
    }
    // LOG_INFO("cur is {} , size {}", cur_, size_);
    return true;
  }

  void emplace_insert(IDType u, DistanceType dist) {
    if (dist >= data_[size_ - 1].distance) {
      return;
    }
    int lo = find_bsearch(dist);
    std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(Neighbor<IDType, DistanceType>));
    data_[lo] = {u, dist};
  }

  auto top() -> IDType { return data_[cur_].id_; }
  auto pop() -> IDType {
    set_checked(data_[cur_].id_);
    int pre = cur_;
    while (cur_ < size_ && is_checked(data_[cur_].id_)) {
      cur_++;
    }

    // LOG_INFO("pop idx is {} , {}",data_[pre].id_, get_id(data_[pre].id_));
    return get_id(data_[pre].id_);
  }

  auto has_next() const -> bool { return cur_ < size_; }
  auto next_id() const -> IDType { return get_id(data_[cur_].id_); }
  auto id(IDType i) const -> IDType { return get_id(data_[i].id_); }
  auto dist(IDType i) const -> DistanceType { return data_[i].distance_; }
  auto size() const -> size_t { return size_; }
  auto capacity() const -> size_t { return capacity_; }

  constexpr static int kMask = 2147483647;
  auto get_id(IDType id) const -> IDType { return id & kMask; }
  // Need to assert IDType is uint32_t instead of uint64_t
  void set_checked(IDType &id) { id |= 1 << 31; }
  auto is_checked(IDType id) -> bool { return (id >> 31 & 1) != 0; }
  auto is_full() -> bool { return size_ == capacity_; }

  size_t nb_, size_ = 0, cur_ = 0, capacity_;
  std::vector<Neighbor<IDType, DistanceType>, AlignAlloc<Neighbor<IDType, DistanceType>>> data_;
  DynamicBitset vis_;
};

}  // namespace alaya
