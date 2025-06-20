#pragma once

#include <sys/mman.h>
#include <climits>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <unordered_set>


#include "utils/rbq_utils/defines.hpp"
#include "utils/rbq_utils/tools.hpp"

namespace alaya {
using IDType = PID;

template <typename T, size_t Alignment = 64, bool HugePage = false>
class AlignedAllocator {
   private:
    static_assert(Alignment >= alignof(T));

   public:
    using value_type = T;

    template <class U>
    struct rebind { // NOLINT
        using other = AlignedAllocator<U, Alignment>;
    };

    constexpr AlignedAllocator() noexcept = default;

    constexpr AlignedAllocator(const AlignedAllocator&) noexcept = default;

    template <typename U>
    constexpr explicit AlignedAllocator(AlignedAllocator<U, Alignment> const& /*unused*/) noexcept {}

    [[nodiscard]] auto allocate(std::size_t n) -> T* {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        auto nbytes = round_up_to_multiple_of<size_t>(n * sizeof(T), Alignment);
        auto* ptr = std::aligned_alloc(Alignment, nbytes);
        if (HugePage) {
            madvise(ptr, nbytes, MADV_HUGEPAGE);
        }
        return reinterpret_cast<T*>(ptr);
    }

    void deallocate(T* ptr, [[maybe_unused]] std::size_t n) { std::free(ptr); }
};

/**
 * @brief hash set to record visited vertices
 *
 */
class HashBasedBooleanSet {
 private:
  size_t table_size_ = 0;
  IDType mask_ = 0;
  std::vector<IDType, AlignedAllocator<IDType>> table_;  // possible error: try original memory.hpp
  std::unordered_set<IDType> stl_hash_;

  [[nodiscard]] auto hash1(const IDType value) const { return value & mask_; }

 public:
  HashBasedBooleanSet() = default;

  HashBasedBooleanSet(const HashBasedBooleanSet &other)  // NOLINT
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
    mask_ = static_cast<IDType>(table_size_ - 1);
    const IDType check_val = hash1(static_cast<IDType>(table_size));  // NOLINT
    if (check_val != 0) {
      std::cerr << "[WARN] table size is not 2^N :  " << table_size << '\n';
    }

    table_ = std::vector<IDType, AlignedAllocator<IDType>>(table_size);
    std::fill(table_.begin(), table_.end(), kPidMax);  // NOLINT
    stl_hash_.clear();
  }

  void clear() {
    std::fill(table_.begin(), table_.end(), kPidMax);  // NOLINT
    stl_hash_.clear();
  }

  // get if data_id is in the hashset
  [[nodiscard]] auto get(IDType data_id) const -> bool {
    IDType val = this->table_[hash1(data_id)];
    if (val == data_id) {
      return true;
    }
    return (val != kPidMax && stl_hash_.find(data_id) != stl_hash_.end());
  }

  void set(IDType data_id) {
    IDType &val = table_[hash1(data_id)];
    if (val == data_id) {
      return;
    }
    if (val == kPidMax) {
      val = data_id;
    } else {
      stl_hash_.emplace(data_id);
    }
  }
};

class VisitedListPool {
  std::deque<HashBasedBooleanSet *> pool_;
  std::mutex poolguard_;
  size_t numelements_;

 public:
  VisitedListPool(size_t initpoolsize, size_t max_elements) {
    numelements_ = max_elements / 10;
    for (size_t i = 0; i < initpoolsize; i++) {
      pool_.push_front(new HashBasedBooleanSet(numelements_));
    }
  }

  auto get_free_vislist() -> HashBasedBooleanSet * {
    HashBasedBooleanSet *rez;
    {
      std::unique_lock<std::mutex> lock(poolguard_);
      if (pool_.size() > 0) {
        rez = pool_.front();
        pool_.pop_front();
      } else {
        rez = new HashBasedBooleanSet(numelements_);
      }
    }
    rez->clear();
    return rez;
  }

  void release_vis_list(HashBasedBooleanSet *vl) {
    std::unique_lock<std::mutex> lock(poolguard_);
    pool_.push_front(vl);
  }

  ~VisitedListPool() {
    while (pool_.size() > 0) {
      HashBasedBooleanSet *rez = pool_.front();
      pool_.pop_front();
      ::delete rez;
    }
  }
};

}  // namespace alaya