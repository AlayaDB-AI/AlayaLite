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

#include <algorithm>
#include <fstream>
#include <utility>
#include <vector>

#include "utils/memory.hpp"

namespace alaya {
template <typename T = char, typename Dims = std::vector<size_t>, typename Alloc = AlignedAlloc<T>>
class StaticStorage {
 private:
  static_assert(std::is_trivial_v<T>);  // only handle trivial types
  static_assert(std::is_same_v<typename Dims::value_type, size_t>);

  void destroy() {
    size_t num_elements = size();
    atraits::deallocate(allocator_, pointer_, num_elements);
    pointer_ = nullptr;
  }

 public:
  using allocator_type = Alloc;
  using atraits = std::allocator_traits<allocator_type>;
  using pointer = typename atraits::pointer;  // T*
  using const_pointer = typename atraits::const_pointer;

  using value_type = T;
  using reference = T &;
  using const_reference = const T &;

  StaticStorage() = default;

  explicit StaticStorage(Dims dims, const Alloc &allocator)
      : dims_(std::move(dims)), allocator_(allocator) {
    size_t num_elements = size();
    pointer_ = atraits::allocate(allocator_, num_elements);
  }

  explicit StaticStorage(Dims dims) : StaticStorage(std::move(dims), Alloc()) {}

  ~StaticStorage() noexcept {
    if (pointer_ != nullptr) {
      destroy();
    }
  }

  /// @brief move constructor
  StaticStorage(StaticStorage &&other) noexcept
      : pointer_{std::exchange(other.pointer_, nullptr)},
        dims_{std::move(other.dims_)},
        allocator_{std::move(other.allocator_)} {}

  auto operator=(StaticStorage &&other) noexcept -> StaticStorage & {
    if (pointer_ != nullptr) {
      destroy();
    }

    if constexpr (atraits::propagate_on_container_move_assignment::value) {
      allocator_ = std::move(other.allocator_);
    }
    dims_ = std::exchange(other.dims_, Dims());
    pointer_ = std::exchange(other.pointer_, nullptr);
    return *this;
  }

  /// @brief num of data objects
  [[nodiscard]] constexpr auto size() const -> size_t {
    // i.e, dims_: std::vector<size_t>{num_points_, data_chunk_size}
    size_t res = 1;
    std::for_each(dims_.begin(), dims_.end(), [&](auto cur_d) -> void {
      res *= cur_d;
    });
    return res;
  }

  /// @brief num of bytes for all data objects
  [[nodiscard]] constexpr auto bytes() const -> size_t { return sizeof(T) * size(); }

  [[nodiscard]] auto data() -> pointer { return pointer_; }
  [[nodiscard]] auto data() const -> const_pointer { return pointer_; }

  [[nodiscard]] auto at(size_t idx) -> reference { return pointer_[idx]; }
  [[nodiscard]] auto at(size_t idx) const -> const_reference { return pointer_[idx]; }

  void save(std::ofstream &output) const {
    if (output.good()) {
      output.write(reinterpret_cast<char *>(pointer_), bytes());
    }
  }
  void load(std::ifstream &input) { input.read(reinterpret_cast<char *>(pointer_), bytes()); }

  auto operator[](size_t idx) -> reference { return pointer_[idx]; }

 private:
  pointer pointer_ = nullptr;
  [[no_unique_address]] Dims dims_;
  [[no_unique_address]] Alloc allocator_;
};

}  // namespace alaya
