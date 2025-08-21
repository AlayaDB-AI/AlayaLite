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

#ifdef __linux__
#include <sys/mman.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#ifdef _MSC_VER
#include <malloc.h>
#endif

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
#ifdef _MSC_VER
      return ptr_ = static_cast<ValueType *>(_aligned_malloc(sz, 64));
#else
      return ptr_ = static_cast<ValueType *>(std::aligned_alloc(64, sz));
#endif
    }
    int sz = (n * sizeof(ValueType) + (1 << 21) - 1) >> 21 << 21;
#ifdef _MSC_VER
    ptr_ = static_cast<ValueType *>(_aligned_malloc(sz, 1 << 21));
#else
    ptr_ = static_cast<ValueType *>(std::aligned_alloc(1 << 21, sz));
#endif

#if defined(__linux__)
    madvise(ptr_, sz, MADV_HUGEPAGE);
#endif
    return ptr_;
  }

  void deallocate(ValueType * /*unused*/, int /*unused*/) {
#ifdef _MSC_VER
    _aligned_free(ptr_);
#else
    free(ptr_);
#endif
  }

  template <typename U>
  struct Rebind {
    using other = AlignAlloc<U>;
  };
  auto operator!=(const AlignAlloc &rhs) -> bool { return ptr_ != rhs.ptr_; }
};

inline auto alloc_2m(size_t nbytes) -> void * {
  size_t len = (nbytes + (1 << 21) - 1) >> 21 << 21;
#ifdef _MSC_VER
  auto p = _aligned_malloc(len, 1 << 21);
#else
  auto p = std::aligned_alloc(1 << 21, len);
#endif
  std::memset(p, 0, len);
  return p;
}

inline auto alloc_64b(size_t nbytes) -> void * {
  size_t len = (nbytes + (1 << 6) - 1) >> 6 << 6;
#ifdef _MSC_VER
  auto p = _aligned_malloc(len, 1 << 6);
#else
  auto p = std::aligned_alloc(1 << 6, len);
#endif
  std::memset(p, 0, len);
  return p;
}

}  // namespace alaya
