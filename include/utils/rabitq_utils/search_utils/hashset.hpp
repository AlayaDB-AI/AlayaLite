/*
 * Copyright 2025 VectorDB.NTU
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

#include <sys/mman.h>
#include <climits>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <unordered_set>
#include <vector>

#include "allocator.hpp"

namespace alaya {
// NOLINTBEGIN

/**
 * @brief hash set to record visited vertices
 *
 */
class HashBasedBooleanSet {
 private:
  using PID = uint32_t;
  static constexpr uint32_t kEmptyHashId = 0xFFFFFFFF;
  size_t table_size_ = 0;
  PID mask_ = 0;
  std::vector<PID, AlignedAllocator<PID>> table_;
  std::unordered_set<PID> stl_hash_;

  [[nodiscard]] auto hash1(const PID value) const { return value & mask_; }

 public:
  HashBasedBooleanSet() = default;

  HashBasedBooleanSet(const HashBasedBooleanSet &other)
      : table_size_(other.table_size_),
        mask_(other.mask_),
        table_(other.table_),
        stl_hash_(other.stl_hash_) {}

  HashBasedBooleanSet(HashBasedBooleanSet &&other) noexcept
      : table_size_(other.table_size_),
        mask_(other.mask_),
        table_(std::move(other.table_)),
        stl_hash_(std::move(other.stl_hash_)) {}
  auto operator=(HashBasedBooleanSet &&other) noexcept -> HashBasedBooleanSet & {
    table_size_ = other.table_size_;
    mask_ = other.mask_;
    table_ = std::move(other.table_);
    stl_hash_ = std::move(other.stl_hash_);

    return *this;
  }

  explicit HashBasedBooleanSet(size_t size) {
    size_t bit_size = 0;
    size_t bit = size;
    while (bit != 0) {
      bit_size++;
      bit >>= 1;
    }
    size_t bucket_size = 0x1 << ((bit_size + 4) / 2 + 3);
    initialize(bucket_size);
  }

  void initialize(const size_t table_size) {
    table_size_ = table_size;
    mask_ = static_cast<PID>(table_size_ - 1);
    const PID check_val = hash1(static_cast<PID>(table_size));
    if (check_val != 0) {
      std::cerr << "[WARN] table size is not 2^N :  " << table_size << '\n';
    }

    table_ = std::vector<PID, AlignedAllocator<PID>>(table_size);
    std::fill(table_.begin(), table_.end(), kEmptyHashId);
    stl_hash_.clear();
  }

  void clear() {
    std::fill(table_.begin(), table_.end(), kEmptyHashId);
    stl_hash_.clear();
  }

  // get if data_id is in the hashset
  [[nodiscard]] auto get(PID data_id) const -> bool {
    PID val = this->table_[hash1(data_id)];
    if (val == data_id) {
      return true;
    }
    return (val != kEmptyHashId && stl_hash_.find(data_id) != stl_hash_.end());
  }

  void set(PID data_id) {
    PID &val = table_[hash1(data_id)];
    if (val == data_id) {
      return;
    }
    if (val == kEmptyHashId) {
      val = data_id;
    } else {
      stl_hash_.emplace(data_id);
    }
  }
};
// NOLINTEND
}  // namespace alaya
