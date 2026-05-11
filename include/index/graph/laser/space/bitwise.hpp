// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file bitwise.hpp
 * @brief Binary vector operations for RaBitQ quantization.
 *
 * Provides utilities for:
 * - popcount: Count set bits in binary vectors
 * - pack_binary: Convert 0/1 integer arrays to packed uint64 format
 */

#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

namespace alaya::laser::space {

/** @brief Counts total set bits (popcount) across a binary vector. */
inline auto popcount(size_t dim, const uint64_t *__restrict__ data) -> size_t {
  size_t ret = 0;
  for (size_t i = 0; i < dim / 64; ++i) {
    ret += __builtin_popcountll((*data));
    ++data;
  }
  return ret;
}

/** @brief Packs 0/1 integer array into uint64 bit-packed format. */
inline void pack_binary(const int *__restrict__ bin_x,
                        uint64_t *__restrict__ binary,
                        size_t length) {
  for (size_t i = 0; i < length; i += 64) {
    uint64_t cur = 0;
    for (size_t j = 0; j < 64; ++j) {
      cur |= (static_cast<uint64_t>(bin_x[i + j]) << (63 - j));
    }
    *binary = cur;
    ++binary;
  }
}

}  // namespace alaya::laser::space
