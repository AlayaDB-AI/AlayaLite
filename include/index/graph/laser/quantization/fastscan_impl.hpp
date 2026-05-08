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

#include <immintrin.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

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
  size_t ncode_pad = (ncode + 31) & ~31;  // size_t ncode_pad = ((ncode + 31) / 32) * 32;
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
  size_t ncode_pad = (ncode + 31) & ~31;  // size_t ncode_pad = ((ncode + 31) / 32) * 32;
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

/**
 * @brief Accumulates quantization codes using SIMD lookup tables.
 *
 * Uses shuffle-based table lookup to compute inner products between
 * quantized query and database codes in a single pass over 32 vectors.
 */
inline void accumulate_impl(size_t dim,
                            const uint8_t *__restrict__ codes,
                            const uint8_t *__restrict__ LUT,
                            uint16_t *__restrict__ result) {
  size_t code_length = dim << 2;
#if defined(__AVX512F__)
  __m512i c;
  __m512i lo;
  __m512i hi;
  __m512i lut;
  __m512i res_lo;
  __m512i res_hi;

  const __m512i lo_mask = _mm512_set1_epi8(0x0f);
  __m512i accu0 = _mm512_setzero_si512();
  __m512i accu1 = _mm512_setzero_si512();
  __m512i accu2 = _mm512_setzero_si512();
  __m512i accu3 = _mm512_setzero_si512();

  for (size_t i = 0; i < code_length; i += 64) {
    c = _mm512_loadu_si512(&codes[i]);
    lut = _mm512_loadu_si512(&LUT[i]);
    lo = _mm512_and_si512(c, lo_mask);
    hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

    res_lo = _mm512_shuffle_epi8(lut, lo);
    res_hi = _mm512_shuffle_epi8(lut, hi);

    accu0 = _mm512_add_epi16(accu0, res_lo);
    accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
    accu2 = _mm512_add_epi16(accu2, res_hi);
    accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
  }
  accu0 = _mm512_sub_epi16(accu0, _mm512_slli_epi16(accu1, 8));
  accu2 = _mm512_sub_epi16(accu2, _mm512_slli_epi16(accu3, 8));

  __m512i ret1 = _mm512_add_epi16(_mm512_mask_blend_epi64(0b11110000, accu0, accu1),
                                  _mm512_shuffle_i64x2(accu0, accu1, 0b01001110));
  __m512i ret2 = _mm512_add_epi16(_mm512_mask_blend_epi64(0b11110000, accu2, accu3),
                                  _mm512_shuffle_i64x2(accu2, accu3, 0b01001110));
  __m512i ret = _mm512_setzero_si512();

  ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b10001000));
  ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b11011101));

  _mm512_storeu_si512(result, ret);

#elif defined(__AVX2__)
  __m256i c, lo, hi, lut, res_lo, res_hi;

  __m256i low_mask = _mm256_set1_epi8(0xf);
  __m256i accu0 = _mm256_setzero_si256();
  __m256i accu1 = _mm256_setzero_si256();
  __m256i accu2 = _mm256_setzero_si256();
  __m256i accu3 = _mm256_setzero_si256();

  for (size_t i = 0; i < code_length; i += 64) {
    c = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&codes[i]));
    lut = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&LUT[i]));
    lo = _mm256_and_si256(c, low_mask);
    hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

    res_lo = _mm256_shuffle_epi8(lut, lo);
    res_hi = _mm256_shuffle_epi8(lut, hi);

    accu0 = _mm256_add_epi16(accu0, res_lo);
    accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
    accu2 = _mm256_add_epi16(accu2, res_hi);
    accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

    c = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&codes[i + 32]));
    lut = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&LUT[i + 32]));
    lo = _mm256_and_si256(c, low_mask);
    hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

    res_lo = _mm256_shuffle_epi8(lut, lo);
    res_hi = _mm256_shuffle_epi8(lut, hi);

    accu0 = _mm256_add_epi16(accu0, res_lo);
    accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
    accu2 = _mm256_add_epi16(accu2, res_hi);
    accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
  }

  accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
  __m256i dis0 = _mm256_add_epi16(_mm256_permute2f128_si256(accu0, accu1, 0x21),
                                  _mm256_blend_epi32(accu0, accu1, 0xF0));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(result), dis0);

  accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
  __m256i dis1 = _mm256_add_epi16(_mm256_permute2f128_si256(accu2, accu3, 0x21),
                                  _mm256_blend_epi32(accu2, accu3, 0xF0));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(&result[16]), dis1);
#else
  throw std::runtime_error("fastscan_impl: AVX2 or AVX512 required but not detected at build time");
#endif
}

/** @brief Packs query bytes into lookup table format for SIMD accumulation. */
inline void pack_lut_impl(size_t dim,
                          const uint8_t *__restrict__ byte_query,
                          uint8_t *__restrict__ LUT) {
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
