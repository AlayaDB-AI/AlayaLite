#pragma once
#include <sys/mman.h>
#include <cstddef>
#include <limits>
#include <new>
#include "utils/rbq_utils/tools.hpp"

namespace alaya {
template <typename T, size_t Alignment = 64, bool HugePage = false>
class AlignedAllocator {
 private:
  static_assert(Alignment >= alignof(T));

 public:
  using value_type = T;

  template <class U>
  struct rebind {  // NOLINT
    using other = AlignedAllocator<U, Alignment>;
  };

  constexpr AlignedAllocator() noexcept = default;

  constexpr AlignedAllocator(const AlignedAllocator &) noexcept = default;

  template <typename U>
  constexpr explicit AlignedAllocator(AlignedAllocator<U, Alignment> const & /*unused*/) noexcept {}

  [[nodiscard]] auto allocate(std::size_t n) -> T * {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }

    auto nbytes = round_up_to_multiple_of<size_t>(n * sizeof(T), Alignment);
    auto *ptr = std::aligned_alloc(Alignment, nbytes);
    if (HugePage) {
      madvise(ptr, nbytes, MADV_HUGEPAGE);
    }
    return reinterpret_cast<T *>(ptr);
  }

  void deallocate(T *ptr, [[maybe_unused]] std::size_t n) { std::free(ptr); }
};
}  // namespace alaya