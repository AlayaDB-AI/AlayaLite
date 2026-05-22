// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file fastscan_impl.hpp
 * @brief SIMD-optimized fast scan for accumulating quantization codes.
 *
 * Based on Faiss FastScan implementation for efficient product quantization.
 * Packs binary codes into a format suitable for SIMD lookup table accumulation,
 * enabling batch processing of 32 codes simultaneously using AVX2/AVX512.
 *
 * Reference:
 * https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)
 */

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "utils/platform.hpp"

namespace alaya::laser {

#define LOWBIT(x) ((x) & (-(x)))

constexpr static size_t kBatchSize = 32;
constexpr static std::array<int, 16> kPos = {
    3 /*0000*/,
    3 /*0001*/,
    2 /*0010*/,
    3 /*0011*/,
    1 /*0100*/,
    3 /*0101*/,
    2 /*0110*/,
    3 /*0111*/,
    0 /*1000*/,
    3 /*1001*/,
    2 /*1010*/,
    3 /*1011*/,
    1 /*1100*/,
    3 /*1101*/,
    2 /*1110*/,
    3 /*1111*/,
};

constexpr static std::array<int, 16> kPerm0 =
    {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};

template <typename T, class TA>
static inline void get_column(const T *src,
                              size_t rows,
                              size_t cols,
                              size_t row,
                              size_t col,
                              TA &dest) {
  size_t k = 0;
  size_t max_k = std::min(rows - row, dest.size());
  for (; k < max_k; ++k) {
    dest[k] = src[((k + row) * cols) + col];
  }
  if (k < dest.size()) {
    std::fill(dest.begin() + k, dest.end(), 0);
  }
}

/**
 * @brief ack 32 quantization codes in a batch from the quantization codes represented by a
 * sequence of uint8_t variables
 *
 * @param padded_dim    dim of vector
 * @param codes         input quantization codes
 * @param ncode         number of vectors (codes)
 * @param blocks        padded results (block of codes)
 */
static inline void pack_codes_helper(size_t padded_dim,
                                     const uint8_t *codes,
                                     size_t ncode,
                                     uint8_t *blocks) {
  size_t ncode_pad = (ncode + 31) & ~31;
  size_t num_codebook = padded_dim / 4;
  std::memset(blocks, 0, ncode_pad * num_codebook / 2);

  uint8_t *codes2 = blocks;
  for (size_t blk = 0; blk < ncode_pad; blk += kBatchSize) {
    // enumerate i
    for (size_t i = 0; i < num_codebook; i += 2) {
      std::array<uint8_t, 32> col;
      std::array<uint8_t, 32> col_lo;
      std::array<uint8_t, 32> col_hi;
      get_column(codes, ncode, num_codebook / 2, blk, i / 2, col);
      for (int j = 0; j < 32; j++) {
        col_lo[j] = col[j] & 15;
        col_hi[j] = col[j] >> 4;
      }
      for (int j = 0; j < 16; j++) {
        auto val0 = col_lo[kPerm0[j]] | (col_lo[kPerm0[j] + 16] << 4);
        auto val1 = col_hi[kPerm0[j]] | (col_hi[kPerm0[j] + 16] << 4);
        codes2[j] = val0;
        codes2[j + 16] = val1;
      }
      codes2 += 32;
    }
  }
}

inline void pack_codes(size_t padded_dim,
                       const uint64_t *binary_code,
                       size_t ncode,
                       uint8_t *blocks) {
  size_t ncode_pad = (ncode + 31) & ~31;
  std::vector<uint8_t> binary_code_8bit(ncode_pad * padded_dim / 8);
  std::memcpy(binary_code_8bit.data(), binary_code, ncode * padded_dim / 64 * sizeof(uint64_t));

  for (size_t i = 0; i < ncode; ++i) {
    for (size_t j = 0; j < padded_dim / 64; ++j) {
      for (size_t k = 0; k < 4; ++k) {
        std::swap(binary_code_8bit[(i * padded_dim / 8) + (8 * j) + k],
                  binary_code_8bit[(i * padded_dim / 8) + (8 * j) + 8 - k - 1]);
      }
    }
  }

  for (size_t i = 0; i < ncode * padded_dim / 8; ++i) {
    uint8_t val = binary_code_8bit[i];
    uint8_t val_hi = (val >> 4);
    uint8_t val_lo = (val & 15);
    binary_code_8bit[i] = (val_lo << 4) | val_hi;
  }
  pack_codes_helper(padded_dim, binary_code_8bit.data(), ncode, blocks);
}

/** @brief Packs query bytes into lookup table format for SIMD accumulation. */
inline void pack_lut_impl(size_t dim,
                          const uint8_t *ALAYA_RESTRICT byte_query,
                          uint8_t *ALAYA_RESTRICT LUT) {
  size_t num_codebook = dim >> 2;
  for (size_t i = 0; i < num_codebook; ++i) {
    LUT[0] = 0;
    for (int j = 1; j < 16; ++j) {
      LUT[j] = LUT[j - LOWBIT(j)] + byte_query[kPos[j]];
    }
    LUT += 16;
    byte_query += 4;
  }
}
}  // namespace alaya::laser
