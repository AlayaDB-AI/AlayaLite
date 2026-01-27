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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "utils/math.hpp"
#include "utils/platform.hpp"

namespace alaya {

template <typename DataType, typename IDType>
struct SequentialStorage {
  using data_type = DataType;
  using id_type = IDType;
  const size_t kBitEveryByte = 8;

  size_t item_size_{0};
  size_t aligned_item_size_{0};
  size_t capacity_{0};
  size_t pos_{0};
  size_t alignment_{0};
  DataType *data_{nullptr};
  uint8_t *bitmap_{nullptr};

  SequentialStorage(const SequentialStorage &) = delete;
  auto operator=(const SequentialStorage &) -> SequentialStorage & = delete;

  SequentialStorage() = default;
  ~SequentialStorage() {
    alaya_aligned_free_impl(data_);
    alaya_aligned_free_impl(bitmap_);
  }

  auto init(size_t item_size, size_t capacity, char fill = 0, size_t alignment = 64) -> void {
    item_size_ = item_size;
    capacity_ = capacity;
    alignment_ = alignment;
    aligned_item_size_ = math::round_up_pow2(item_size_, alignment);
    data_ = static_cast<DataType *>(
        alaya_aligned_alloc_impl(aligned_item_size_ * capacity_, alignment));
    std::memset(data_, fill, aligned_item_size_ * capacity_);
    auto bitmap_size =
        math::round_up_pow2((capacity_ + kBitEveryByte - 1) / kBitEveryByte, alignment);

    bitmap_ = static_cast<uint8_t *>(alaya_aligned_alloc_impl(bitmap_size, alignment_));
    std::memset(bitmap_, 0, bitmap_size);
  }

  auto operator[](IDType index) const -> DataType * {
    return reinterpret_cast<DataType *>(reinterpret_cast<char *>(data_) +
                                        (index * aligned_item_size_));
  }

  auto is_valid(IDType index) const -> bool {
    return bitmap_[index / kBitEveryByte] & (1 << (index % kBitEveryByte));
  }

  auto insert(const DataType *data) -> IDType {
    if (pos_ >= capacity_) {
      return -1;
    }
    std::memcpy(operator[](pos_), data, item_size_);
    bitmap_[pos_ / kBitEveryByte] |= (1 << (pos_ % kBitEveryByte));
    return pos_++;
  }

  auto reserve() -> IDType {
    if (pos_ >= capacity_) {
      return -1;
    }
    bitmap_[pos_ / kBitEveryByte] |= (1 << (pos_ % kBitEveryByte));
    return pos_++;
  }

  auto remove(IDType id) -> IDType {
    if (!is_valid(id)) {
      return -1;
    }
    bitmap_[id / kBitEveryByte] &= ~(1 << (id % kBitEveryByte));
    return id;
  }

  auto update(IDType id, const DataType *data) -> IDType {
    if (!is_valid(id)) {
      return -1;
    }
    std::memcpy(operator[](id), data, item_size_);
    return id;
  }

  auto save(std::ofstream &writer) const -> void {
    writer.write(reinterpret_cast<const char *>(&item_size_), sizeof(item_size_));
    writer.write(reinterpret_cast<const char *>(&aligned_item_size_), sizeof(aligned_item_size_));
    writer.write(reinterpret_cast<const char *>(&capacity_), sizeof(capacity_));
    writer.write(reinterpret_cast<const char *>(&pos_), sizeof(pos_));
    writer.write(reinterpret_cast<const char *>(&alignment_), sizeof(alignment_));
    writer.write(reinterpret_cast<char *>(data_), aligned_item_size_ * capacity_);
    writer.write(reinterpret_cast<char *>(bitmap_),
                 (capacity_ + kBitEveryByte - 1) / kBitEveryByte);
  }

  auto load(std::ifstream &reader) -> void {
    if (data_ != nullptr) {
      alaya_aligned_free_impl(data_);
    }
    if (bitmap_ != nullptr) {
      alaya_aligned_free_impl(bitmap_);
    }
    reader.read(reinterpret_cast<char *>(&item_size_), sizeof(item_size_));
    reader.read(reinterpret_cast<char *>(&aligned_item_size_), sizeof(aligned_item_size_));
    reader.read(reinterpret_cast<char *>(&capacity_), sizeof(capacity_));
    reader.read(reinterpret_cast<char *>(&pos_), sizeof(pos_));
    reader.read(reinterpret_cast<char *>(&alignment_), sizeof(alignment_));

    data_ = static_cast<DataType *>(
        alaya_aligned_alloc_impl(aligned_item_size_ * capacity_, alignment_));
    reader.read(reinterpret_cast<char *>(data_), aligned_item_size_ * capacity_);

    auto bitmap_size =
        math::round_up_pow2((capacity_ + kBitEveryByte - 1) / kBitEveryByte, alignment_);
    bitmap_ = static_cast<uint8_t *>(alaya_aligned_alloc_impl(bitmap_size, alignment_));
    reader.read(reinterpret_cast<char *>(bitmap_), (capacity_ + kBitEveryByte - 1) / kBitEveryByte);
  }
};

}  // namespace alaya
