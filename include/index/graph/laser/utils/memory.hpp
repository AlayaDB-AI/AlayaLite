// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file memory.hpp
 * @brief Memory allocation utilities with alignment and huge page support.
 *
 * Provides:
 * - AlignedAllocator: STL-compatible allocator with configurable alignment
 * - align_allocate(): Raw aligned memory allocation
 * - prefetch utilities: L1/L2 cache prefetch hints
 *
 * Alignment is critical for SIMD operations (AVX2 requires 32-byte, AVX512 requires 64-byte).
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>

#include "index/graph/laser/utils/tools.hpp"
#include "utils/platform.hpp"

// _mm_prefetch lives in <xmmintrin.h>/<immintrin.h>; <intrin.h> from platform.hpp
// already pulls it in on MSVC. GCC/Clang use the same header via <immintrin.h>.
#if defined(ALAYA_ARCH_X86)
  #include <immintrin.h>
  #define ALAYA_HAS_MM_PREFETCH 1
#else
  #define ALAYA_HAS_MM_PREFETCH 0
#endif

namespace alaya::laser::memory {
#if defined(_MSC_VER)
  #define PORTABLE_ALIGN32 __declspec(align(32))
  #define PORTABLE_ALIGN64 __declspec(align(64))
#else
  #define PORTABLE_ALIGN32 __attribute__((aligned(32)))
  #define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#endif

/**
 * @brief STL-compatible allocator with configurable alignment and huge page support.
 * @tparam T Element type
 * @tparam Alignment Byte alignment (default 64 for cache line)
 * @tparam HugePage Enable huge page allocation via madvise
 */
template <typename T, size_t Alignment = 64, bool HugePage = false>
class AlignedAllocator {
 private:
  static_assert(Alignment >= alignof(T));

 public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal = std::true_type;

  template <class U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment, HugePage>;
  };

  constexpr AlignedAllocator() noexcept = default;

  constexpr AlignedAllocator(const AlignedAllocator &) noexcept = default;

  template <typename U>
  constexpr explicit AlignedAllocator(AlignedAllocator<U, Alignment, HugePage> const &) noexcept {}

  [[nodiscard]] T *allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }

    auto nbytes = round_up_to_multiple(n * sizeof(T), Alignment);
    auto *ptr = alaya_aligned_alloc_impl(nbytes, Alignment);
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
    if (HugePage) {
#ifdef ALAYA_OS_LINUX
      madvise(ptr, nbytes, MADV_HUGEPAGE);
#endif
    }
    return reinterpret_cast<T *>(ptr);
  }

  void deallocate(T *ptr, [[maybe_unused]] std::size_t bytes) { alaya_aligned_free_impl(ptr); }
};

template <typename T,
          size_t TAlignment,
          bool THugePage,
          typename U,
          size_t UAlignment,
          bool UHugePage>
auto operator==(const AlignedAllocator<T, TAlignment, THugePage> & /*unused*/,
                const AlignedAllocator<U, UAlignment, UHugePage> & /*unused*/) noexcept -> bool {
  return TAlignment == UAlignment && THugePage == UHugePage;
}

template <typename T,
          size_t TAlignment,
          bool THugePage,
          typename U,
          size_t UAlignment,
          bool UHugePage>
auto operator!=(const AlignedAllocator<T, TAlignment, THugePage> &lhs,
                const AlignedAllocator<U, UAlignment, UHugePage> &rhs) noexcept -> bool {
  return !(lhs == rhs);
}

template <typename T>
struct Allocator {
 public:
  using value_type = T;

  constexpr Allocator() noexcept = default;

  template <typename U>
  explicit constexpr Allocator(const Allocator<U> &) noexcept {}

  [[nodiscard]] constexpr T *allocate(std::size_t n) {
    return static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t(alignof(T))));
  }

  constexpr void deallocate(T *ptr, size_t count) noexcept { ::operator delete(ptr, count); }

  // Intercept zero-argument construction to do default initialization.
  template <typename U>
  void construct(U *ptr) noexcept(std::is_nothrow_default_constructible_v<U>) {
    ::new (static_cast<void *>(ptr)) U;
  }
};

/** @brief Allocates aligned memory, optionally with huge page hints. */
template <size_t Alignment>
inline void *align_allocate(size_t nbytes, bool huge_page = false) {
  size_t size = round_up_to_multiple(nbytes, Alignment);
  void *ptr = alaya_aligned_alloc_impl(size, Alignment);
  assert(ptr != nullptr);
  if (huge_page) {
#ifdef ALAYA_OS_LINUX
    madvise(ptr, nbytes, MADV_HUGEPAGE);
#endif
  }
  std::memset(ptr, 0, size);
  return ptr;
}

inline void align_free(void *ptr) noexcept { alaya_aligned_free_impl(ptr); }

static inline void prefetch_l1(const void *addr) {
#if ALAYA_HAS_MM_PREFETCH
  _mm_prefetch(reinterpret_cast<const char *>(addr), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(addr, 0, 3);
#else
  (void)addr;  // No prefetch on this target; correctness unaffected.
#endif
}

static inline void prefetch_l2(const void *addr) {
#if ALAYA_HAS_MM_PREFETCH
  _mm_prefetch(reinterpret_cast<const char *>(addr), _MM_HINT_T1);
#elif defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(addr, 0, 2);
#else
  (void)addr;
#endif
}

inline void mem_prefetch_l1(const char *ptr, size_t num_lines) {
  switch (num_lines) {
    default:
      [[fallthrough]];
    case 20:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 19:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 18:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 17:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 16:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 15:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 14:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 13:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 12:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 11:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 10:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 9:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 8:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 7:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 6:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 5:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 4:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 3:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 2:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 1:
      prefetch_l1(ptr);
      ptr += 64;
      [[fallthrough]];
    case 0:
      break;
  }
}

inline void mem_prefetch_l2(const char *ptr, size_t num_lines) {
  switch (num_lines) {
    default:
      [[fallthrough]];
    case 20:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 19:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 18:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 17:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 16:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 15:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 14:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 13:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 12:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 11:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 10:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 9:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 8:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 7:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 6:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 5:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 4:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 3:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 2:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 1:
      prefetch_l2(ptr);
      ptr += 64;
      [[fallthrough]];
    case 0:
      break;
  }
}
}  // namespace alaya::laser::memory
