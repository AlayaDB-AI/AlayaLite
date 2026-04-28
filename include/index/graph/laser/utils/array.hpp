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

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "index/graph/laser/utils/memory.hpp"

namespace alaya::laser::data {

/**
 * @brief Owning flat buffer for an N-D array of trivially-copyable elements.
 *
 * Holds a contiguous allocation of `prod(dims_)` elements obtained from an
 * allocator, with bytewise save/load to std::ofstream / std::ifstream. The
 * allocator is the only knob that controls alignment — pass
 * memory::AlignedAllocator<T, A, HugePage> to get A-byte aligned storage.
 *
 * Non-copyable (owns raw pointer); move-only.
 */
template <typename T, typename Dims = std::vector<size_t>, typename Alloc = memory::Allocator<T>>
class Array {
  static_assert(std::is_trivial_v<T>, "alaya::laser::data::Array requires a trivial element type");

  using traits = std::allocator_traits<Alloc>;

 public:
  using allocator_type = Alloc;
  using value_type = T;
  using pointer = typename traits::pointer;
  using const_pointer = typename traits::const_pointer;
  using reference = T &;
  using const_reference = const T &;

  Array() = default;

  explicit Array(Dims dims, const Alloc &allocator)
      : dims_{std::move(dims)}, allocator_{allocator} {
    const size_t n = element_count();
    pointer_ = (n == 0) ? nullptr : traits::allocate(allocator_, n);
  }

  explicit Array(Dims dims) : Array(std::move(dims), Alloc{}) {}

  Array(const Array &) = delete;
  Array &operator=(const Array &) = delete;

  Array(Array &&other) noexcept
      : pointer_{std::exchange(other.pointer_, nullptr)},
        dims_{std::move(other.dims_)},
        allocator_{std::move(other.allocator_)} {}

  Array &operator=(Array &&other) noexcept {
    if (this != &other) {
      destroy();
      pointer_ = std::exchange(other.pointer_, nullptr);
      dims_ = std::move(other.dims_);
      allocator_ = std::move(other.allocator_);
    }
    return *this;
  }

  ~Array() noexcept { destroy(); }

  [[nodiscard]] pointer data() noexcept { return pointer_; }
  [[nodiscard]] const_pointer data() const noexcept { return pointer_; }

  [[nodiscard]] reference at(size_t idx) noexcept { return pointer_[idx]; }
  [[nodiscard]] const_reference at(size_t idx) const noexcept { return pointer_[idx]; }

  reference operator[](size_t idx) noexcept { return pointer_[idx]; }

  void save(std::ofstream &output) const {
    if (output.good() && pointer_ != nullptr) {
      output.write(reinterpret_cast<const char *>(pointer_),
                   static_cast<std::streamsize>(byte_count()));
    }
  }

  void load(std::ifstream &input) {
    if (pointer_ != nullptr) {
      input.read(reinterpret_cast<char *>(pointer_), static_cast<std::streamsize>(byte_count()));
    }
  }

 private:
  [[nodiscard]] size_t element_count() const noexcept {
    size_t n = 1;
    for (auto d : dims_) {
      n *= d;
    }
    return n;
  }

  [[nodiscard]] size_t byte_count() const noexcept { return sizeof(T) * element_count(); }

  void destroy() noexcept {
    if (pointer_ != nullptr) {
      traits::deallocate(allocator_, pointer_, element_count());
      pointer_ = nullptr;
    }
  }

  pointer pointer_ = nullptr;
  [[no_unique_address]] Dims dims_{};
  [[no_unique_address]] Alloc allocator_{};
};

}  // namespace alaya::laser::data
