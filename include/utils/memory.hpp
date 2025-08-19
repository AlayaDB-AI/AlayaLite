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

#include <sys/mman.h>
#include <cstdio>

#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace alaya {
/**
 * @brief Aligned memory allocator.
 *
 * @tparam ValueType The type of value to allocate memory.
 */
template <typename ValueType>
  requires std::is_trivial_v<ValueType>
struct AlignAlloc {
  using value_type = ValueType;
  ValueType *ptr_ = nullptr;
  auto allocate(int n) -> ValueType * {
    if (n <= 1 << 14) {
      int sz = (n * sizeof(ValueType) + 63) >> 6 << 6;
      return ptr_ = static_cast<ValueType *>(std::aligned_alloc(64, sz));
    }
    int sz = (n * sizeof(ValueType) + (1 << 21) - 1) >> 21 << 21;
    ptr_ = static_cast<ValueType *>(std::aligned_alloc(1 << 21, sz));

#if defined(__linux__)
    madvise(ptr_, sz, MADV_HUGEPAGE);
#endif
    return ptr_;
  }

  void deallocate(ValueType * /*unused*/, int /*unused*/) { free(ptr_); }

  template <typename U>
  struct Rebind {
    using other = AlignAlloc<U>;
  };
  auto operator!=(const AlignAlloc &rhs) -> bool { return ptr_ != rhs.ptr_; }
};

inline auto alloc_2m(size_t nbytes) -> void * {
  size_t len = (nbytes + (1 << 21) - 1) >> 21 << 21;
  auto p = std::aligned_alloc(1 << 21, len);
  std::memset(p, 0, len);
  return p;
}

inline auto alloc_64b(size_t nbytes) -> void * {
  size_t len = (nbytes + (1 << 6) - 1) >> 6 << 6;
  auto p = std::aligned_alloc(1 << 6, len);
  std::memset(p, 0, len);
  return p;
}

}  // namespace alaya
