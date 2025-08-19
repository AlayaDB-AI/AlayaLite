#pragma once

#include <cstddef>
#include <vector>

#include "allocator.hpp"
#include "index/neighbor.hpp"
#include "utils/rbq_utils//defines.hpp"
namespace alaya {
/**
 * @brief sorted linear buffer, used as beam set for graph-based ANN search. In symphonyqg,
 * the search buffer may contain duplicate id with different distances
 *
 */
template <typename T = float>
class SearchBuffer {
  using ANNCand = Neighbor<PID, T>;

 private:
  std::vector<ANNCand, AlignedAllocator<ANNCand>> data_;
  size_t size_ = 0, cur_ = 0, capacity_;

  [[nodiscard]] auto binary_search(T dist) const {
    size_t lo = 0;
    size_t len = size_;
    size_t half;
    while (len > 1) {
      half = len >> 1;
      len -= half;
      lo += static_cast<size_t>(data_[lo + half - 1].distance_ < dist) * half;
    }
    return (lo < size_ && data_[lo].distance_ < dist) ? lo + 1 : lo;
  }

  // set top bit to 1 as checked
  static void set_checked(PID &data_id) { data_id |= (1 << 31); }

  [[nodiscard]] static auto is_checked(PID data_id) -> bool {
    return static_cast<bool>(data_id >> 31);
  }

 public:
  SearchBuffer() = default;

  explicit SearchBuffer(size_t capacity) : data_(capacity + 1), capacity_(capacity) {}

  // insert a data point into buffer
  void insert(PID data_id, T dist) {
    if (is_full(dist)) {
      return;
    }

    size_t lo = binary_search(dist);
    std::memmove(&data_[lo + 1], &data_[lo], (size_ - lo) * sizeof(ANNCand));
    data_[lo] = ANNCand(data_id, dist);
    size_ += static_cast<size_t>(size_ < capacity_);
    cur_ = lo < cur_ ? lo : cur_;
  }

  // get unchecked candidate with minimum distance
  auto pop() -> PID {
    PID cur_id = data_[cur_].id_;
    set_checked(data_[cur_].id_);
    ++cur_;
    while (cur_ < size_ && is_checked(data_[cur_].id_)) {
      ++cur_;
    }
    return cur_id;
  }

  void clear() {
    size_ = 0;
    cur_ = 0;
  }

  // return candidate id for next pop()
  [[nodiscard]] auto next_id() const { return data_[cur_].id_; }

  [[nodiscard]] auto has_next() const -> bool { return cur_ < size_; }

  void resize(size_t new_size) {
    this->capacity_ = new_size;
    data_ = std::vector<ANNCand, AlignedAllocator<ANNCand>>(capacity_ + 1);
  }

  void copy_results_to(PID *knn) const {
    for (size_t i = 0; i < size_; ++i) {
      knn[i] = data_[i].id_;
    }
  }

  auto top_dist() const -> T {
    return is_full() ? data_[size_ - 1].distance_ : std::numeric_limits<T>::max();
  }

  [[nodiscard]] auto is_full() const -> bool { return size_ == capacity_; }

  // judge if dist can be inserted into buffer
  [[nodiscard]] auto is_full(T dist) const -> bool { return dist > top_dist(); }

  auto data() -> const std::vector<ANNCand, AlignedAllocator<ANNCand>> & { return data_; }

  auto size() -> size_t { return size_; }
};
}  // namespace alaya