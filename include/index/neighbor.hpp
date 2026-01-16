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

#include <cmath>
#include <cstdint>
#include <cstring>

namespace alaya {

/**
 * @brief The unified structure for a neighbor.
 *
 * @tparam IDType The data type for storing IDs is determined by the number of
 vectors that need to be indexed, with the default type being uint64_t.
 */
template <typename IDType = uint64_t, typename DistanceType = float>
struct Neighbor {
  IDType id_;              ///< The id of the current point.
  DistanceType distance_;  ///< The distance between the query point and the current point.
  bool flag_;              ///< The flag identify the current point is visited or not.

  Neighbor() = default;
  Neighbor(IDType id, DistanceType distance, bool f = false)
      : id_(id), distance_(distance), flag_(f) {}

  friend auto operator<(const Neighbor &lhs, const Neighbor &rhs) -> bool {
    return lhs.distance_ < rhs.distance_ || (lhs.distance_ == rhs.distance_ && lhs.id_ < rhs.id_);
  }

  friend auto operator>(const Neighbor &lhs, const Neighbor &rhs) -> bool { return !(lhs < rhs); }
};

template <typename IDType = uint64_t, typename DistanceType = float>
struct Node {
  IDType id_;
  DistanceType distance_;

  Node() = default;
  Node(IDType id, DistanceType distance) : id_(id), distance_(distance) {}

  auto operator<(const Node &other) const -> bool { return distance_ < other.distance_; }
};

/**
 * @brief This is used by nsg.
 *
 * @tparam IDType The data type for storing IDs is determined by the number of
 vectors that need to be indexed, with the default type being uint64_t.
 */
template <typename IDType = uint64_t>
inline auto insert_into_pool(Neighbor<IDType> *addr, int K, Neighbor<IDType> nn) -> int {
  // find the location to insert
  int left = 0;
  int right = K - 1;
  if (addr[left].distance_ > nn.distance_) {
    memmove(&addr[left + 1], &addr[left], K * sizeof(Neighbor<IDType>));
    addr[left] = nn;
    return left;
  }
  if (addr[right].distance_ < nn.distance_) {
    addr[K] = nn;
    return K;
  }
  while (left < right - 1) {
    int mid = (left + right) / 2;
    if (addr[mid].distance_ > nn.distance_) {
      right = mid;
    } else {
      left = mid;
    }
  }
  // check equal ID

  while (left > 0) {
    if (addr[left].distance_ < nn.distance_) {
      break;
    }
    if (addr[left].id_ == nn.id_) {
      return K + 1;
    }
    left--;
  }
  if (addr[left].id_ == nn.id_ || addr[right].id_ == nn.id_) {
    return K + 1;
  }
  memmove(&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor<IDType>));
  addr[right] = nn;
  return right;
}

}  // namespace alaya
