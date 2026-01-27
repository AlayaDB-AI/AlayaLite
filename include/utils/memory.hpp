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

#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include "utils/math.hpp"
#include "utils/platform.hpp"

namespace alaya {

template <typename T>
class AlignedAlloc {
 private:
  static constexpr size_t kAlignSmall = 64;
  static constexpr size_t kAlignLarge = 2 * 1024 * 1024;   // 2MB
  static constexpr size_t kHugePageThreshold = 16 * 1024;  // 16KB

 public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal = std::true_type;

  template <class U>
  struct rebind {  // NOLINT
    using other = AlignedAlloc<U>;
  };

  constexpr AlignedAlloc() noexcept = default;
  constexpr AlignedAlloc(const AlignedAlloc &) noexcept = default;

  template <typename U>
  constexpr explicit AlignedAlloc(const AlignedAlloc<U> & /*unused*/) noexcept {}

  /**
   * @brief Allocate aligned memory
   *
   * @param n Number of elements to allocate
   * @return Pointer to allocated memory
   * @throws std::bad_alloc on allocation failure
   * @throws std::bad_array_new_length if n is too large
   */
  [[nodiscard]] auto allocate(std::size_t n) -> T * {
    if (n == 0) {
      return nullptr;
    }
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }

    size_t raw_size = n * sizeof(T);
    size_t align = (raw_size >= kHugePageThreshold) ? kAlignLarge : kAlignSmall;
    bool use_huge_page = raw_size >= kHugePageThreshold;

    const auto kNbytes = math::round_up_pow2(raw_size, align);
    void *ptr = alaya_aligned_alloc_impl(kNbytes, align);
    if (ALAYA_UNLIKELY(!ptr)) {
      throw std::bad_alloc();
    }

#ifdef ALAYA_OS_LINUX
    if (use_huge_page) {
      // Suggest using THP (Transparent Huge Pages) for the kernel)
      madvise(ptr, kNbytes, MADV_HUGEPAGE);
    }
#endif
    return reinterpret_cast<T *>(ptr);
  }

  /**
   * @brief Deallocate aligned memory
   * @param ptr Pointer to memory to deallocate
   * @param n Number of elements (not used)
   */
  void deallocate(T *ptr, [[maybe_unused]] std::size_t n) noexcept {
    if (ptr) {
      alaya_aligned_free_impl(ptr);
    }
  }
};
template <typename T, typename U>
auto operator==(const AlignedAlloc<T> & /*unused*/, const AlignedAlloc<U> & /*unused*/) noexcept
    -> bool {
  return true;
}

template <typename T, typename U>
auto operator!=(const AlignedAlloc<T> & /*unused*/, const AlignedAlloc<U> & /*unused*/) noexcept
    -> bool {
  return false;
}

inline auto alloc_2m(size_t nbytes) -> void * {
  auto len = math::round_up_pow2(nbytes, 1 << 21);
  auto p = alaya_aligned_alloc_impl(len, 1 << 21);
  std::memset(p, 0, len);
  return p;
}

inline auto alloc_64b(size_t nbytes) -> void * {
  auto len = math::round_up_pow2(nbytes, 1 << 6);
  auto p = alaya_aligned_alloc_impl(len, 1 << 6);
  std::memset(p, 0, len);
  return p;
}

}  // namespace alaya
