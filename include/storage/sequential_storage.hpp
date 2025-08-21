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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <streambuf>
#include <vector>
#include "storage_concept.hpp"
#include "utils/log.hpp"
#include "utils/types.hpp"

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace alaya {

template <typename DataType, typename IDType>
struct SequentialStorage {
  using data_type = DataType;
  using id_type = IDType;

  size_t item_size_{0};
  size_t aligned_item_size_{0};
  size_t capacity_{0};
  size_t pos_{0};
  size_t alignment_{0};
  DataType *data_{nullptr};
  size_t *bitmap_{nullptr};

  ~SequentialStorage() {
    if (data_ != nullptr) {
#ifdef _MSC_VER
      _aligned_free(data_);
#else
      std::free(data_);  // NOLINT
#endif
    }
    if (bitmap_ != nullptr) {
#ifdef _MSC_VER
      _aligned_free(bitmap_);
#else
      std::free(bitmap_);  // NOLINT
#endif
    }
  }

  auto init(size_t item_size, size_t capacity, char fill = 0,
            size_t alignment = 64) -> void {
    item_size_ = item_size;
    capacity_ = capacity;
    alignment_ = alignment;
    aligned_item_size_ = do_align(item_size_, alignment);
#ifdef _MSC_VER
    data_ = static_cast<DataType *>(
        _aligned_malloc(aligned_item_size_ * capacity_, alignment));
#else
    data_ = static_cast<DataType *>(
        std::aligned_alloc(alignment, aligned_item_size_ * capacity_));
#endif
    std::memset(data_, fill, aligned_item_size_ * capacity_);
    auto bitmap_size = do_align(capacity_ / sizeof(char) + 1, alignment);
#ifdef _MSC_VER
    bitmap_ = static_cast<size_t *>(_aligned_malloc(bitmap_size, alignment));
#else
    bitmap_ = static_cast<size_t *>(std::aligned_alloc(alignment, bitmap_size));
#endif
    std::memset(bitmap_, 0, bitmap_size);
  }

  auto operator[](IDType index) const -> DataType * {
    return reinterpret_cast<DataType *>(reinterpret_cast<char *>(data_) +
                                        index * aligned_item_size_);
  }

  auto is_valid(IDType index) const -> bool {
    return bitmap_[index / sizeof(size_t)] & (1 << (index % sizeof(size_t)));
  }

  auto insert(const DataType *data) -> IDType {
    if (pos_ >= capacity_) {
      return -1;
    }
    std::memcpy(operator[](pos_), data, item_size_);
    bitmap_[pos_ / sizeof(size_t)] |= (1 << (pos_ % sizeof(size_t)));
    return pos_++;
  }

  auto reserve() -> IDType {
    if (pos_ >= capacity_) {
      return -1;
    }
    bitmap_[pos_ / sizeof(size_t)] |= (1 << (pos_ % sizeof(size_t)));
    return pos_++;
  }

  auto remove(IDType id) -> IDType {
    if (!is_valid(id)) {
      return -1;
    }
    bitmap_[id / sizeof(size_t)] &= ~(1 << (id % sizeof(size_t)));
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
    writer.write(reinterpret_cast<const char *>(&aligned_item_size_),
                 sizeof(aligned_item_size_));
    writer.write(reinterpret_cast<const char *>(&capacity_), sizeof(capacity_));
    writer.write(reinterpret_cast<const char *>(&pos_), sizeof(pos_));
    writer.write(reinterpret_cast<const char *>(&alignment_), sizeof(alignment_));
    writer.write(reinterpret_cast<char *>(data_),
                 aligned_item_size_ * capacity_);
    writer.write(reinterpret_cast<char *>(bitmap_), capacity_ / sizeof(char) + 1);
  }

  auto load(std::ifstream &reader) -> void {
    if (data_ != nullptr) {
#ifdef _MSC_VER
      _aligned_free(data_);
#else
      std::free(data_);  // NOLINT
#endif
    }
    if (bitmap_ != nullptr) {
#ifdef _MSC_VER
      _aligned_free(bitmap_);
#else
      std::free(bitmap_);  // NOLINT
#endif
    }
    reader.read(reinterpret_cast<char *>(&item_size_), sizeof(item_size_));
    reader.read(reinterpret_cast<char *>(&aligned_item_size_),
                sizeof(aligned_item_size_));
    reader.read(reinterpret_cast<char *>(&capacity_), sizeof(capacity_));
    reader.read(reinterpret_cast<char *>(&pos_), sizeof(pos_));
    reader.read(reinterpret_cast<char *>(&alignment_), sizeof(alignment_));
#ifdef _MSC_VER
    data_ = static_cast<DataType *>(
        _aligned_malloc(aligned_item_size_ * capacity_, alignment_));
#else
    data_ = static_cast<DataType *>(
        std::aligned_alloc(alignment_, aligned_item_size_ * capacity_));
#endif
    reader.read(reinterpret_cast<char *>(data_),
                aligned_item_size_ * capacity_);
    auto bitmap_size = do_align(capacity_ / sizeof(char) + 1, alignment_);
#ifdef _MSC_VER
    bitmap_ = static_cast<size_t *>(_aligned_malloc(bitmap_size, alignment_));
#else
    bitmap_ =
        static_cast<size_t *>(std::aligned_alloc(alignment_, bitmap_size));
#endif
    reader.read(reinterpret_cast<char *>(bitmap_), capacity_ / sizeof(char) + 1);
  }
};

}  // namespace alaya