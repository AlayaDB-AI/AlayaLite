#pragma once

#include <cstddef>
#include <ctime>
#include <queue>
#include <random>
#include <thread>
#include <type_traits>

namespace alaya {
template <typename T>
inline void assert_integral() {
  static_assert(std::is_integral_v<T>,
                "Template type T must be an integral type (int, long, unsigned, etc.).");
}

template <typename T>
inline void assert_floating() {
  static_assert(std::is_floating_point_v<T>,
                "Template type T must be a floating-point type (float, double, long double).");
}

// thread save rand int
template <typename T>
inline auto rand_integer(T min, T max) -> T {
  static thread_local std::mt19937 generator(
      std::random_device{}() + std::hash<std::thread::id>()(std::this_thread::get_id()));
  std::uniform_int_distribution<T> distribution(min, max);
  return distribution(generator);
}

constexpr auto div_round_up(size_t val, size_t div) -> size_t {
  return (val / div) + static_cast<size_t>((val % div) != 0);
}

constexpr auto round_up_to_multiple(size_t val, size_t multiple_of) -> size_t {
  return multiple_of * (div_round_up(val, multiple_of));
}

inline auto floor_log2(size_t x) -> size_t {
  size_t ret = 0;
  while (x > 1) {
    ret++;
    x >>= 1;
  }
  return ret;
}

inline auto ceil_log2(size_t x) -> size_t {
  size_t ret = floor_log2(x);
  return (1UL << ret) < x ? ret + 1 : ret;
}

inline auto is_powerof2(size_t n) -> bool { return n > 0 && (n & (n - 1)) == 0; }

template <typename T>
constexpr auto div_round_up(T x, T divisor) -> T {
  static_assert(std::is_integral_v<T>, "T must be an integral type");
  return (x / divisor) + static_cast<T>((x % divisor) != 0);
}

template <typename T>
constexpr auto round_up_to_multiple_of(size_t x, size_t multiple_of) -> T {
  return multiple_of * (div_round_up(x, multiple_of));
}

// get number of threads of current sys
inline auto total_threads() -> size_t {
  const auto threads = std::thread::hardware_concurrency();  // NOLINT
  return threads == 0 ? 1 : threads;
}

template <typename T, typename TP>
auto distance_ratio(const T *data, const T *query, const TP *gt, const TP *ann_results, size_t k,
                    size_t dim, T (*dist_func)(const T *, const T *, size_t)) -> float {
  std::priority_queue<float> gt_distances;
  std::priority_queue<float> ann_distances;

  for (size_t i = 0; i < k; ++i) {
    TP gt_id = gt[i];
    TP ann_id = ann_results[i];
    gt_distances.emplace(dist_func(query, data + (gt_id * dim), dim));
    ann_distances.emplace(dist_func(query, data + (ann_id * dim), dim));
  }

  float ret = 0;
  size_t valid_k = 0;

  while (!gt_distances.empty()) {
    if (gt_distances.top() > 1e-5) {
      ret += std::sqrt(ann_distances.top() / gt_distances.top());
      ++valid_k;
    }
    gt_distances.pop();
    ann_distances.pop();
  }

  if (valid_k == 0) {
    return static_cast<float>(k);
  }
  return ret * static_cast<float>(k) / static_cast<float>(valid_k);
}

template <typename T>
auto horizontal_avg(const std::vector<std::vector<T>> &data) -> std::vector<T> {
  size_t rows = data.size();
  size_t cols = data[0].size();

  std::vector<T> avg(cols, 0);
  for (auto &row : data) {
    for (size_t j = 0; j < cols; ++j) {
      avg[j] += row[j];
    }
  }

  for (size_t j = 0; j < cols; ++j) {
    avg[j] /= rows;
  }

  return avg;
}
}  // namespace alaya