// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file common.hpp
 * @brief Common type definitions and utilities for the library.
 *
 * Defines fundamental types used throughout the codebase:
 * - PID: Point/node identifier type (uint32_t)
 * - Candidate: Distance-sorted candidate for k-NN search
 * - RowMatrix/ColMatrix: Eigen matrix wrappers for vector data
 * - DistFunc: Distance function type alias
 */

#pragma once

#include <cstdint>
#include <functional>

#include <Eigen/Dense>

namespace alaya::laser {
#define RANDOM_QUERY_QUANTIZATION
#define QG_BQUERY 6
#if defined(_MSC_VER) && !defined(__clang__)
  #define FORCE_INLINE __forceinline
  #define LIKELY(x) (x)
  #define UNLIKELY(x) (x)
#else
  #define FORCE_INLINE inline __attribute__((always_inline))
  #define LIKELY(x) __builtin_expect(!!(x), 1)
  #define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

using PID = uint32_t;                     // Point ID type for graph nodes
constexpr uint32_t kPidMax = 0xFFFFFFFF;  // Maximum valid PID value

constexpr float kCacheRatio = 0.15;  // Maximum ratio of nodes to cache in memory (15%)

template <typename T>
using RowMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using ColMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

template <typename T>
using DistFunc = std::function<T(const T *, const T *, size_t)>;

/**
 * @brief Represents a candidate node in k-NN search with its distance to query.
 * @tparam T Distance type (typically float)
 */
template <typename T>
struct Candidate {
  PID id;      // Node identifier
  T distance;  // Distance to query vector

  Candidate() = default;
  explicit Candidate(PID vec_id, T dis) : id(vec_id), distance(dis) {}

  auto operator<(const Candidate &other) const { return distance < other.distance; }

  auto operator>(const Candidate &other) const { return !(*this < other); }
};
}  // namespace alaya::laser
