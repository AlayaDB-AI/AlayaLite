/*
 * Copyright 2025 VectorDB.NTU
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

#pragma once

#include <cstddef>
#include <type_traits>
#include "cpu_features.hpp"
#include "platform.hpp"

namespace alaya::simd {

using FHT_Helper_Func = void (*)(float *a);

template <size_t log_n>
inline auto fwht_generic_template(float *buf) -> void {
  size_t n = 1 << log_n;
  for (size_t i = 0; i < log_n; ++i) {
    size_t s1 = 1 << i;
    size_t s2 = s1 << 1;
    for (size_t j = 0; j < n; j += s2) {
      for (size_t k = 0; k < s1; ++k) {
        float u = buf[j + k];
        float v = buf[j + k + s1];
        buf[j + k] = u + v;
        buf[j + k + s1] = u - v;
      }
    }
  }
}

#ifdef ALAYA_X86

ALAYA_NOINLINE
ALAYA_TARGET_AVX2
auto helper_float_6_avx2(float *buf) -> void {  // NOLINT
  for (int j = 0; j < 64; j += 32) {
    for (int k = 0; k < 4; k += 4) {
      __asm__ volatile(
          "movups (%0), %%xmm0\n"
          "movups (%1), %%xmm1\n"
          "movups (%2), %%xmm2\n"
          "movups (%3), %%xmm3\n"
          "movups (%4), %%xmm4\n"
          "movups (%5), %%xmm5\n"
          "movups (%6), %%xmm6\n"
          "movups (%7), %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm0, %%xmm0\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm0, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm1, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm1, %%xmm1\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm1, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm1\n"
          "movaps %%xmm2, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm2, %%xmm2\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm2, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm2\n"
          "movaps %%xmm3, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm3, %%xmm3\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm3, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm3\n"
          "movaps %%xmm4, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm4, %%xmm4\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm4, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm4\n"
          "movaps %%xmm5, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm5, %%xmm5\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm5, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm5\n"
          "movaps %%xmm6, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm6, %%xmm6\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm6, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm6\n"
          "movaps %%xmm7, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm7, %%xmm7\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm7, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm0, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm0, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm0\n"
          "movaps %%xmm1, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm1, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm1, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm1\n"
          "movaps %%xmm2, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm2\n"
          "movaps %%xmm3, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm3, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm3, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm3\n"
          "movaps %%xmm4, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm4, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm4, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm4\n"
          "movaps %%xmm5, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm5, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm5, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm5\n"
          "movaps %%xmm6, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm6, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm6, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm6\n"
          "movaps %%xmm7, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm7, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm7, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm9\n"
          "addps %%xmm1, %%xmm8\n"
          "subps %%xmm1, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "addps %%xmm3, %%xmm10\n"
          "subps %%xmm3, %%xmm11\n"
          "movaps %%xmm4, %%xmm12\n"
          "movaps %%xmm4, %%xmm13\n"
          "addps %%xmm5, %%xmm12\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm6, %%xmm14\n"
          "movaps %%xmm6, %%xmm15\n"
          "addps %%xmm7, %%xmm14\n"
          "subps %%xmm7, %%xmm15\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm8, %%xmm2\n"
          "addps %%xmm10, %%xmm0\n"
          "subps %%xmm10, %%xmm2\n"
          "movaps %%xmm9, %%xmm1\n"
          "movaps %%xmm9, %%xmm3\n"
          "addps %%xmm11, %%xmm1\n"
          "subps %%xmm11, %%xmm3\n"
          "movaps %%xmm12, %%xmm4\n"
          "movaps %%xmm12, %%xmm6\n"
          "addps %%xmm14, %%xmm4\n"
          "subps %%xmm14, %%xmm6\n"
          "movaps %%xmm13, %%xmm5\n"
          "movaps %%xmm13, %%xmm7\n"
          "addps %%xmm15, %%xmm5\n"
          "subps %%xmm15, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm12\n"
          "addps %%xmm4, %%xmm8\n"
          "subps %%xmm4, %%xmm12\n"
          "movaps %%xmm1, %%xmm9\n"
          "movaps %%xmm1, %%xmm13\n"
          "addps %%xmm5, %%xmm9\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm14\n"
          "addps %%xmm6, %%xmm10\n"
          "subps %%xmm6, %%xmm14\n"
          "movaps %%xmm3, %%xmm11\n"
          "movaps %%xmm3, %%xmm15\n"
          "addps %%xmm7, %%xmm11\n"
          "subps %%xmm7, %%xmm15\n"
          "movups %%xmm8, (%0)\n"
          "movups %%xmm9, (%1)\n"
          "movups %%xmm10, (%2)\n"
          "movups %%xmm11, (%3)\n"
          "movups %%xmm12, (%4)\n"
          "movups %%xmm13, (%5)\n"
          "movups %%xmm14, (%6)\n"
          "movups %%xmm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 4),
          "r"(buf + j + k + 8),
          "r"(buf + j + k + 12),
          "r"(buf + j + k + 16),
          "r"(buf + j + k + 20),
          "r"(buf + j + k + 24),
          "r"(buf + j + k + 28)
          : "%xmm0",
            "%xmm1",
            "%xmm2",
            "%xmm3",
            "%xmm4",
            "%xmm5",
            "%xmm6",
            "%xmm7",
            "%xmm8",
            "%xmm9",
            "%xmm10",
            "%xmm11",
            "%xmm12",
            "%xmm13",
            "%xmm14",
            "%xmm15",
            "memory");
    }
  }
  for (int j = 0; j < 64; j += 64) {
    for (int k = 0; k < 32; k += 4) {
      __asm__ volatile(
          "movups (%0), %%xmm0\n"
          "movups (%1), %%xmm1\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm9\n"
          "addps %%xmm1, %%xmm8\n"
          "subps %%xmm1, %%xmm9\n"
          "movups %%xmm8, (%0)\n"
          "movups %%xmm9, (%1)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 32)
          : "%xmm0",
            "%xmm1",
            "%xmm2",
            "%xmm3",
            "%xmm4",
            "%xmm5",
            "%xmm6",
            "%xmm7",
            "%xmm8",
            "%xmm9",
            "%xmm10",
            "%xmm11",
            "%xmm12",
            "%xmm13",
            "%xmm14",
            "%xmm15",
            "memory");
    }
  }
}

ALAYA_NOINLINE
ALAYA_TARGET_AVX2
auto helper_float_7_avx2(float *buf) -> void {  // NOLINT
  for (int j = 0; j < 128; j += 32) {
    for (int k = 0; k < 4; k += 4) {
      __asm__ volatile(
          "movups (%0), %%xmm0\n"
          "movups (%1), %%xmm1\n"
          "movups (%2), %%xmm2\n"
          "movups (%3), %%xmm3\n"
          "movups (%4), %%xmm4\n"
          "movups (%5), %%xmm5\n"
          "movups (%6), %%xmm6\n"
          "movups (%7), %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm0, %%xmm0\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm0, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm1, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm1, %%xmm1\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm1, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm1\n"
          "movaps %%xmm2, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm2, %%xmm2\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm2, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm2\n"
          "movaps %%xmm3, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm3, %%xmm3\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm3, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm3\n"
          "movaps %%xmm4, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm4, %%xmm4\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm4, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm4\n"
          "movaps %%xmm5, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm5, %%xmm5\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm5, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm5\n"
          "movaps %%xmm6, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm6, %%xmm6\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm6, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm6\n"
          "movaps %%xmm7, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm7, %%xmm7\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm7, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm0, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm0, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm0\n"
          "movaps %%xmm1, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm1, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm1, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm1\n"
          "movaps %%xmm2, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm2\n"
          "movaps %%xmm3, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm3, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm3, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm3\n"
          "movaps %%xmm4, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm4, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm4, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm4\n"
          "movaps %%xmm5, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm5, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm5, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm5\n"
          "movaps %%xmm6, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm6, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm6, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm6\n"
          "movaps %%xmm7, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm7, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm7, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm9\n"
          "addps %%xmm1, %%xmm8\n"
          "subps %%xmm1, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "addps %%xmm3, %%xmm10\n"
          "subps %%xmm3, %%xmm11\n"
          "movaps %%xmm4, %%xmm12\n"
          "movaps %%xmm4, %%xmm13\n"
          "addps %%xmm5, %%xmm12\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm6, %%xmm14\n"
          "movaps %%xmm6, %%xmm15\n"
          "addps %%xmm7, %%xmm14\n"
          "subps %%xmm7, %%xmm15\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm8, %%xmm2\n"
          "addps %%xmm10, %%xmm0\n"
          "subps %%xmm10, %%xmm2\n"
          "movaps %%xmm9, %%xmm1\n"
          "movaps %%xmm9, %%xmm3\n"
          "addps %%xmm11, %%xmm1\n"
          "subps %%xmm11, %%xmm3\n"
          "movaps %%xmm12, %%xmm4\n"
          "movaps %%xmm12, %%xmm6\n"
          "addps %%xmm14, %%xmm4\n"
          "subps %%xmm14, %%xmm6\n"
          "movaps %%xmm13, %%xmm5\n"
          "movaps %%xmm13, %%xmm7\n"
          "addps %%xmm15, %%xmm5\n"
          "subps %%xmm15, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm12\n"
          "addps %%xmm4, %%xmm8\n"
          "subps %%xmm4, %%xmm12\n"
          "movaps %%xmm1, %%xmm9\n"
          "movaps %%xmm1, %%xmm13\n"
          "addps %%xmm5, %%xmm9\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm14\n"
          "addps %%xmm6, %%xmm10\n"
          "subps %%xmm6, %%xmm14\n"
          "movaps %%xmm3, %%xmm11\n"
          "movaps %%xmm3, %%xmm15\n"
          "addps %%xmm7, %%xmm11\n"
          "subps %%xmm7, %%xmm15\n"
          "movups %%xmm8, (%0)\n"
          "movups %%xmm9, (%1)\n"
          "movups %%xmm10, (%2)\n"
          "movups %%xmm11, (%3)\n"
          "movups %%xmm12, (%4)\n"
          "movups %%xmm13, (%5)\n"
          "movups %%xmm14, (%6)\n"
          "movups %%xmm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 4),
          "r"(buf + j + k + 8),
          "r"(buf + j + k + 12),
          "r"(buf + j + k + 16),
          "r"(buf + j + k + 20),
          "r"(buf + j + k + 24),
          "r"(buf + j + k + 28)
          : "%xmm0",
            "%xmm1",
            "%xmm2",
            "%xmm3",
            "%xmm4",
            "%xmm5",
            "%xmm6",
            "%xmm7",
            "%xmm8",
            "%xmm9",
            "%xmm10",
            "%xmm11",
            "%xmm12",
            "%xmm13",
            "%xmm14",
            "%xmm15",
            "memory");
    }
  }
  for (int j = 0; j < 128; j += 128) {
    for (int k = 0; k < 32; k += 4) {
      __asm__ volatile(
          "movups (%0), %%xmm0\n"
          "movups (%1), %%xmm1\n"
          "movups (%2), %%xmm2\n"
          "movups (%3), %%xmm3\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm9\n"
          "addps %%xmm1, %%xmm8\n"
          "subps %%xmm1, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "addps %%xmm3, %%xmm10\n"
          "subps %%xmm3, %%xmm11\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm8, %%xmm2\n"
          "addps %%xmm10, %%xmm0\n"
          "subps %%xmm10, %%xmm2\n"
          "movaps %%xmm9, %%xmm1\n"
          "movaps %%xmm9, %%xmm3\n"
          "addps %%xmm11, %%xmm1\n"
          "subps %%xmm11, %%xmm3\n"
          "movups %%xmm0, (%0)\n"
          "movups %%xmm1, (%1)\n"
          "movups %%xmm2, (%2)\n"
          "movups %%xmm3, (%3)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 32),
          "r"(buf + j + k + 64),
          "r"(buf + j + k + 96)
          : "%xmm0",
            "%xmm1",
            "%xmm2",
            "%xmm3",
            "%xmm4",
            "%xmm5",
            "%xmm6",
            "%xmm7",
            "%xmm8",
            "%xmm9",
            "%xmm10",
            "%xmm11",
            "%xmm12",
            "%xmm13",
            "%xmm14",
            "%xmm15",
            "memory");
    }
  }
}

ALAYA_NOINLINE
ALAYA_TARGET_AVX2
auto helper_float_8_avx2(float *buf) -> void {  // NOLINT
  for (int j = 0; j < 256; j += 32) {
    for (int k = 0; k < 4; k += 4) {
      __asm__ volatile(
          "movups (%0), %%xmm0\n"
          "movups (%1), %%xmm1\n"
          "movups (%2), %%xmm2\n"
          "movups (%3), %%xmm3\n"
          "movups (%4), %%xmm4\n"
          "movups (%5), %%xmm5\n"
          "movups (%6), %%xmm6\n"
          "movups (%7), %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm0, %%xmm0\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm0, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm1, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm1, %%xmm1\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm1, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm1\n"
          "movaps %%xmm2, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm2, %%xmm2\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm2, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm2\n"
          "movaps %%xmm3, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm3, %%xmm3\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm3, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm3\n"
          "movaps %%xmm4, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm4, %%xmm4\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm4, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm4\n"
          "movaps %%xmm5, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm5, %%xmm5\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm5, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm5\n"
          "movaps %%xmm6, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm6, %%xmm6\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm6, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm6\n"
          "movaps %%xmm7, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm7, %%xmm7\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm7, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm0, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm0, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm0\n"
          "movaps %%xmm1, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm1, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm1, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm1\n"
          "movaps %%xmm2, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm2\n"
          "movaps %%xmm3, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm3, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm3, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm3\n"
          "movaps %%xmm4, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm4, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm4, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm4\n"
          "movaps %%xmm5, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm5, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm5, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm5\n"
          "movaps %%xmm6, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm6, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm6, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm6\n"
          "movaps %%xmm7, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm7, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm7, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm9\n"
          "addps %%xmm1, %%xmm8\n"
          "subps %%xmm1, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "addps %%xmm3, %%xmm10\n"
          "subps %%xmm3, %%xmm11\n"
          "movaps %%xmm4, %%xmm12\n"
          "movaps %%xmm4, %%xmm13\n"
          "addps %%xmm5, %%xmm12\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm6, %%xmm14\n"
          "movaps %%xmm6, %%xmm15\n"
          "addps %%xmm7, %%xmm14\n"
          "subps %%xmm7, %%xmm15\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm8, %%xmm2\n"
          "addps %%xmm10, %%xmm0\n"
          "subps %%xmm10, %%xmm2\n"
          "movaps %%xmm9, %%xmm1\n"
          "movaps %%xmm9, %%xmm3\n"
          "addps %%xmm11, %%xmm1\n"
          "subps %%xmm11, %%xmm3\n"
          "movaps %%xmm12, %%xmm4\n"
          "movaps %%xmm12, %%xmm6\n"
          "addps %%xmm14, %%xmm4\n"
          "subps %%xmm14, %%xmm6\n"
          "movaps %%xmm13, %%xmm5\n"
          "movaps %%xmm13, %%xmm7\n"
          "addps %%xmm15, %%xmm5\n"
          "subps %%xmm15, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm12\n"
          "addps %%xmm4, %%xmm8\n"
          "subps %%xmm4, %%xmm12\n"
          "movaps %%xmm1, %%xmm9\n"
          "movaps %%xmm1, %%xmm13\n"
          "addps %%xmm5, %%xmm9\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm14\n"
          "addps %%xmm6, %%xmm10\n"
          "subps %%xmm6, %%xmm14\n"
          "movaps %%xmm3, %%xmm11\n"
          "movaps %%xmm3, %%xmm15\n"
          "addps %%xmm7, %%xmm11\n"
          "subps %%xmm7, %%xmm15\n"
          "movups %%xmm8, (%0)\n"
          "movups %%xmm9, (%1)\n"
          "movups %%xmm10, (%2)\n"
          "movups %%xmm11, (%3)\n"
          "movups %%xmm12, (%4)\n"
          "movups %%xmm13, (%5)\n"
          "movups %%xmm14, (%6)\n"
          "movups %%xmm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 4),
          "r"(buf + j + k + 8),
          "r"(buf + j + k + 12),
          "r"(buf + j + k + 16),
          "r"(buf + j + k + 20),
          "r"(buf + j + k + 24),
          "r"(buf + j + k + 28)
          : "%xmm0",
            "%xmm1",
            "%xmm2",
            "%xmm3",
            "%xmm4",
            "%xmm5",
            "%xmm6",
            "%xmm7",
            "%xmm8",
            "%xmm9",
            "%xmm10",
            "%xmm11",
            "%xmm12",
            "%xmm13",
            "%xmm14",
            "%xmm15",
            "memory");
    }
  }
  for (int j = 0; j < 256; j += 256) {
    for (int k = 0; k < 32; k += 4) {
      __asm__ volatile(
          "movups (%0), %%xmm0\n"
          "movups (%1), %%xmm1\n"
          "movups (%2), %%xmm2\n"
          "movups (%3), %%xmm3\n"
          "movups (%4), %%xmm4\n"
          "movups (%5), %%xmm5\n"
          "movups (%6), %%xmm6\n"
          "movups (%7), %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm9\n"
          "addps %%xmm1, %%xmm8\n"
          "subps %%xmm1, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "addps %%xmm3, %%xmm10\n"
          "subps %%xmm3, %%xmm11\n"
          "movaps %%xmm4, %%xmm12\n"
          "movaps %%xmm4, %%xmm13\n"
          "addps %%xmm5, %%xmm12\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm6, %%xmm14\n"
          "movaps %%xmm6, %%xmm15\n"
          "addps %%xmm7, %%xmm14\n"
          "subps %%xmm7, %%xmm15\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm8, %%xmm2\n"
          "addps %%xmm10, %%xmm0\n"
          "subps %%xmm10, %%xmm2\n"
          "movaps %%xmm9, %%xmm1\n"
          "movaps %%xmm9, %%xmm3\n"
          "addps %%xmm11, %%xmm1\n"
          "subps %%xmm11, %%xmm3\n"
          "movaps %%xmm12, %%xmm4\n"
          "movaps %%xmm12, %%xmm6\n"
          "addps %%xmm14, %%xmm4\n"
          "subps %%xmm14, %%xmm6\n"
          "movaps %%xmm13, %%xmm5\n"
          "movaps %%xmm13, %%xmm7\n"
          "addps %%xmm15, %%xmm5\n"
          "subps %%xmm15, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm12\n"
          "addps %%xmm4, %%xmm8\n"
          "subps %%xmm4, %%xmm12\n"
          "movaps %%xmm1, %%xmm9\n"
          "movaps %%xmm1, %%xmm13\n"
          "addps %%xmm5, %%xmm9\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm14\n"
          "addps %%xmm6, %%xmm10\n"
          "subps %%xmm6, %%xmm14\n"
          "movaps %%xmm3, %%xmm11\n"
          "movaps %%xmm3, %%xmm15\n"
          "addps %%xmm7, %%xmm11\n"
          "subps %%xmm7, %%xmm15\n"
          "movups %%xmm8, (%0)\n"
          "movups %%xmm9, (%1)\n"
          "movups %%xmm10, (%2)\n"
          "movups %%xmm11, (%3)\n"
          "movups %%xmm12, (%4)\n"
          "movups %%xmm13, (%5)\n"
          "movups %%xmm14, (%6)\n"
          "movups %%xmm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 32),
          "r"(buf + j + k + 64),
          "r"(buf + j + k + 96),
          "r"(buf + j + k + 128),
          "r"(buf + j + k + 160),
          "r"(buf + j + k + 192),
          "r"(buf + j + k + 224)
          : "%xmm0",
            "%xmm1",
            "%xmm2",
            "%xmm3",
            "%xmm4",
            "%xmm5",
            "%xmm6",
            "%xmm7",
            "%xmm8",
            "%xmm9",
            "%xmm10",
            "%xmm11",
            "%xmm12",
            "%xmm13",
            "%xmm14",
            "%xmm15",
            "memory");
    }
  }
}

ALAYA_NOINLINE
ALAYA_TARGET_AVX2
auto helper_float_9_avx2(float *buf) -> void {  // NOLINT
  for (int j = 0; j < 512; j += 32) {
    for (int k = 0; k < 4; k += 4) {
      __asm__ volatile(
          "movups (%0), %%xmm0\n"
          "movups (%1), %%xmm1\n"
          "movups (%2), %%xmm2\n"
          "movups (%3), %%xmm3\n"
          "movups (%4), %%xmm4\n"
          "movups (%5), %%xmm5\n"
          "movups (%6), %%xmm6\n"
          "movups (%7), %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm0, %%xmm0\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm0, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm1, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm1, %%xmm1\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm1, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm1\n"
          "movaps %%xmm2, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm2, %%xmm2\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm2, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm2\n"
          "movaps %%xmm3, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm3, %%xmm3\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm3, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm3\n"
          "movaps %%xmm4, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm4, %%xmm4\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm4, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm4\n"
          "movaps %%xmm5, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm5, %%xmm5\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm5, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm5\n"
          "movaps %%xmm6, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm6, %%xmm6\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm6, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm6\n"
          "movaps %%xmm7, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm7, %%xmm7\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm7, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm0, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm0, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm0\n"
          "movaps %%xmm1, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm1, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm1, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm1\n"
          "movaps %%xmm2, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm2\n"
          "movaps %%xmm3, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm3, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm3, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm3\n"
          "movaps %%xmm4, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm4, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm4, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm4\n"
          "movaps %%xmm5, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm5, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm5, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm5\n"
          "movaps %%xmm6, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm6, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm6, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm6\n"
          "movaps %%xmm7, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm7, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm7, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm9\n"
          "addps %%xmm1, %%xmm8\n"
          "subps %%xmm1, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "addps %%xmm3, %%xmm10\n"
          "subps %%xmm3, %%xmm11\n"
          "movaps %%xmm4, %%xmm12\n"
          "movaps %%xmm4, %%xmm13\n"
          "addps %%xmm5, %%xmm12\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm6, %%xmm14\n"
          "movaps %%xmm6, %%xmm15\n"
          "addps %%xmm7, %%xmm14\n"
          "subps %%xmm7, %%xmm15\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm8, %%xmm2\n"
          "addps %%xmm10, %%xmm0\n"
          "subps %%xmm10, %%xmm2\n"
          "movaps %%xmm9, %%xmm1\n"
          "movaps %%xmm9, %%xmm3\n"
          "addps %%xmm11, %%xmm1\n"
          "subps %%xmm11, %%xmm3\n"
          "movaps %%xmm12, %%xmm4\n"
          "movaps %%xmm12, %%xmm6\n"
          "addps %%xmm14, %%xmm4\n"
          "subps %%xmm14, %%xmm6\n"
          "movaps %%xmm13, %%xmm5\n"
          "movaps %%xmm13, %%xmm7\n"
          "addps %%xmm15, %%xmm5\n"
          "subps %%xmm15, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm12\n"
          "addps %%xmm4, %%xmm8\n"
          "subps %%xmm4, %%xmm12\n"
          "movaps %%xmm1, %%xmm9\n"
          "movaps %%xmm1, %%xmm13\n"
          "addps %%xmm5, %%xmm9\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm14\n"
          "addps %%xmm6, %%xmm10\n"
          "subps %%xmm6, %%xmm14\n"
          "movaps %%xmm3, %%xmm11\n"
          "movaps %%xmm3, %%xmm15\n"
          "addps %%xmm7, %%xmm11\n"
          "subps %%xmm7, %%xmm15\n"
          "movups %%xmm8, (%0)\n"
          "movups %%xmm9, (%1)\n"
          "movups %%xmm10, (%2)\n"
          "movups %%xmm11, (%3)\n"
          "movups %%xmm12, (%4)\n"
          "movups %%xmm13, (%5)\n"
          "movups %%xmm14, (%6)\n"
          "movups %%xmm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 4),
          "r"(buf + j + k + 8),
          "r"(buf + j + k + 12),
          "r"(buf + j + k + 16),
          "r"(buf + j + k + 20),
          "r"(buf + j + k + 24),
          "r"(buf + j + k + 28)
          : "%xmm0",
            "%xmm1",
            "%xmm2",
            "%xmm3",
            "%xmm4",
            "%xmm5",
            "%xmm6",
            "%xmm7",
            "%xmm8",
            "%xmm9",
            "%xmm10",
            "%xmm11",
            "%xmm12",
            "%xmm13",
            "%xmm14",
            "%xmm15",
            "memory");
    }
  }
  for (int j = 0; j < 512; j += 256) {
    for (int k = 0; k < 32; k += 4) {
      __asm__ volatile(
          "movups (%0), %%xmm0\n"
          "movups (%1), %%xmm1\n"
          "movups (%2), %%xmm2\n"
          "movups (%3), %%xmm3\n"
          "movups (%4), %%xmm4\n"
          "movups (%5), %%xmm5\n"
          "movups (%6), %%xmm6\n"
          "movups (%7), %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm9\n"
          "addps %%xmm1, %%xmm8\n"
          "subps %%xmm1, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "addps %%xmm3, %%xmm10\n"
          "subps %%xmm3, %%xmm11\n"
          "movaps %%xmm4, %%xmm12\n"
          "movaps %%xmm4, %%xmm13\n"
          "addps %%xmm5, %%xmm12\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm6, %%xmm14\n"
          "movaps %%xmm6, %%xmm15\n"
          "addps %%xmm7, %%xmm14\n"
          "subps %%xmm7, %%xmm15\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm8, %%xmm2\n"
          "addps %%xmm10, %%xmm0\n"
          "subps %%xmm10, %%xmm2\n"
          "movaps %%xmm9, %%xmm1\n"
          "movaps %%xmm9, %%xmm3\n"
          "addps %%xmm11, %%xmm1\n"
          "subps %%xmm11, %%xmm3\n"
          "movaps %%xmm12, %%xmm4\n"
          "movaps %%xmm12, %%xmm6\n"
          "addps %%xmm14, %%xmm4\n"
          "subps %%xmm14, %%xmm6\n"
          "movaps %%xmm13, %%xmm5\n"
          "movaps %%xmm13, %%xmm7\n"
          "addps %%xmm15, %%xmm5\n"
          "subps %%xmm15, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm12\n"
          "addps %%xmm4, %%xmm8\n"
          "subps %%xmm4, %%xmm12\n"
          "movaps %%xmm1, %%xmm9\n"
          "movaps %%xmm1, %%xmm13\n"
          "addps %%xmm5, %%xmm9\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm14\n"
          "addps %%xmm6, %%xmm10\n"
          "subps %%xmm6, %%xmm14\n"
          "movaps %%xmm3, %%xmm11\n"
          "movaps %%xmm3, %%xmm15\n"
          "addps %%xmm7, %%xmm11\n"
          "subps %%xmm7, %%xmm15\n"
          "movups %%xmm8, (%0)\n"
          "movups %%xmm9, (%1)\n"
          "movups %%xmm10, (%2)\n"
          "movups %%xmm11, (%3)\n"
          "movups %%xmm12, (%4)\n"
          "movups %%xmm13, (%5)\n"
          "movups %%xmm14, (%6)\n"
          "movups %%xmm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 32),
          "r"(buf + j + k + 64),
          "r"(buf + j + k + 96),
          "r"(buf + j + k + 128),
          "r"(buf + j + k + 160),
          "r"(buf + j + k + 192),
          "r"(buf + j + k + 224)
          : "%xmm0",
            "%xmm1",
            "%xmm2",
            "%xmm3",
            "%xmm4",
            "%xmm5",
            "%xmm6",
            "%xmm7",
            "%xmm8",
            "%xmm9",
            "%xmm10",
            "%xmm11",
            "%xmm12",
            "%xmm13",
            "%xmm14",
            "%xmm15",
            "memory");
    }
  }
  for (int j = 0; j < 512; j += 512) {
    for (int k = 0; k < 256; k += 4) {
      __asm__ volatile(
          "movups (%0), %%xmm0\n"
          "movups (%1), %%xmm1\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm9\n"
          "addps %%xmm1, %%xmm8\n"
          "subps %%xmm1, %%xmm9\n"
          "movups %%xmm8, (%0)\n"
          "movups %%xmm9, (%1)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 256)
          : "%xmm0",
            "%xmm1",
            "%xmm2",
            "%xmm3",
            "%xmm4",
            "%xmm5",
            "%xmm6",
            "%xmm7",
            "%xmm8",
            "%xmm9",
            "%xmm10",
            "%xmm11",
            "%xmm12",
            "%xmm13",
            "%xmm14",
            "%xmm15",
            "memory");
    }
  }
}

ALAYA_NOINLINE
ALAYA_TARGET_AVX2
auto helper_float_10_avx2(float *buf) -> void {  // NOLINT
  for (int j = 0; j < 1024; j += 32) {
    for (int k = 0; k < 4; k += 4) {
      __asm__ volatile(
          "movups (%0), %%xmm0\n"
          "movups (%1), %%xmm1\n"
          "movups (%2), %%xmm2\n"
          "movups (%3), %%xmm3\n"
          "movups (%4), %%xmm4\n"
          "movups (%5), %%xmm5\n"
          "movups (%6), %%xmm6\n"
          "movups (%7), %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm0, %%xmm0\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm0, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm1, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm1, %%xmm1\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm1, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm1\n"
          "movaps %%xmm2, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm2, %%xmm2\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm2, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm2\n"
          "movaps %%xmm3, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm3, %%xmm3\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm3, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm3\n"
          "movaps %%xmm4, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm4, %%xmm4\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm4, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm4\n"
          "movaps %%xmm5, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm5, %%xmm5\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm5, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm5\n"
          "movaps %%xmm6, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm6, %%xmm6\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm6, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm6\n"
          "movaps %%xmm7, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm7, %%xmm7\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm7, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm0, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm0, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm0\n"
          "movaps %%xmm1, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm1, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm1, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm1\n"
          "movaps %%xmm2, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm2\n"
          "movaps %%xmm3, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm3, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm3, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm3\n"
          "movaps %%xmm4, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm4, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm4, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm4\n"
          "movaps %%xmm5, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm5, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm5, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm5\n"
          "movaps %%xmm6, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm6, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm6, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm6\n"
          "movaps %%xmm7, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm7, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm7, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm9\n"
          "addps %%xmm1, %%xmm8\n"
          "subps %%xmm1, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "addps %%xmm3, %%xmm10\n"
          "subps %%xmm3, %%xmm11\n"
          "movaps %%xmm4, %%xmm12\n"
          "movaps %%xmm4, %%xmm13\n"
          "addps %%xmm5, %%xmm12\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm6, %%xmm14\n"
          "movaps %%xmm6, %%xmm15\n"
          "addps %%xmm7, %%xmm14\n"
          "subps %%xmm7, %%xmm15\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm8, %%xmm2\n"
          "addps %%xmm10, %%xmm0\n"
          "subps %%xmm10, %%xmm2\n"
          "movaps %%xmm9, %%xmm1\n"
          "movaps %%xmm9, %%xmm3\n"
          "addps %%xmm11, %%xmm1\n"
          "subps %%xmm11, %%xmm3\n"
          "movaps %%xmm12, %%xmm4\n"
          "movaps %%xmm12, %%xmm6\n"
          "addps %%xmm14, %%xmm4\n"
          "subps %%xmm14, %%xmm6\n"
          "movaps %%xmm13, %%xmm5\n"
          "movaps %%xmm13, %%xmm7\n"
          "addps %%xmm15, %%xmm5\n"
          "subps %%xmm15, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm12\n"
          "addps %%xmm4, %%xmm8\n"
          "subps %%xmm4, %%xmm12\n"
          "movaps %%xmm1, %%xmm9\n"
          "movaps %%xmm1, %%xmm13\n"
          "addps %%xmm5, %%xmm9\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm14\n"
          "addps %%xmm6, %%xmm10\n"
          "subps %%xmm6, %%xmm14\n"
          "movaps %%xmm3, %%xmm11\n"
          "movaps %%xmm3, %%xmm15\n"
          "addps %%xmm7, %%xmm11\n"
          "subps %%xmm7, %%xmm15\n"
          "movups %%xmm8, (%0)\n"
          "movups %%xmm9, (%1)\n"
          "movups %%xmm10, (%2)\n"
          "movups %%xmm11, (%3)\n"
          "movups %%xmm12, (%4)\n"
          "movups %%xmm13, (%5)\n"
          "movups %%xmm14, (%6)\n"
          "movups %%xmm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 4),
          "r"(buf + j + k + 8),
          "r"(buf + j + k + 12),
          "r"(buf + j + k + 16),
          "r"(buf + j + k + 20),
          "r"(buf + j + k + 24),
          "r"(buf + j + k + 28)
          : "%xmm0",
            "%xmm1",
            "%xmm2",
            "%xmm3",
            "%xmm4",
            "%xmm5",
            "%xmm6",
            "%xmm7",
            "%xmm8",
            "%xmm9",
            "%xmm10",
            "%xmm11",
            "%xmm12",
            "%xmm13",
            "%xmm14",
            "%xmm15",
            "memory");
    }
  }
  for (int j = 0; j < 1024; j += 256) {
    for (int k = 0; k < 32; k += 4) {
      __asm__ volatile(
          "movups (%0), %%xmm0\n"
          "movups (%1), %%xmm1\n"
          "movups (%2), %%xmm2\n"
          "movups (%3), %%xmm3\n"
          "movups (%4), %%xmm4\n"
          "movups (%5), %%xmm5\n"
          "movups (%6), %%xmm6\n"
          "movups (%7), %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm9\n"
          "addps %%xmm1, %%xmm8\n"
          "subps %%xmm1, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "addps %%xmm3, %%xmm10\n"
          "subps %%xmm3, %%xmm11\n"
          "movaps %%xmm4, %%xmm12\n"
          "movaps %%xmm4, %%xmm13\n"
          "addps %%xmm5, %%xmm12\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm6, %%xmm14\n"
          "movaps %%xmm6, %%xmm15\n"
          "addps %%xmm7, %%xmm14\n"
          "subps %%xmm7, %%xmm15\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm8, %%xmm2\n"
          "addps %%xmm10, %%xmm0\n"
          "subps %%xmm10, %%xmm2\n"
          "movaps %%xmm9, %%xmm1\n"
          "movaps %%xmm9, %%xmm3\n"
          "addps %%xmm11, %%xmm1\n"
          "subps %%xmm11, %%xmm3\n"
          "movaps %%xmm12, %%xmm4\n"
          "movaps %%xmm12, %%xmm6\n"
          "addps %%xmm14, %%xmm4\n"
          "subps %%xmm14, %%xmm6\n"
          "movaps %%xmm13, %%xmm5\n"
          "movaps %%xmm13, %%xmm7\n"
          "addps %%xmm15, %%xmm5\n"
          "subps %%xmm15, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm12\n"
          "addps %%xmm4, %%xmm8\n"
          "subps %%xmm4, %%xmm12\n"
          "movaps %%xmm1, %%xmm9\n"
          "movaps %%xmm1, %%xmm13\n"
          "addps %%xmm5, %%xmm9\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm14\n"
          "addps %%xmm6, %%xmm10\n"
          "subps %%xmm6, %%xmm14\n"
          "movaps %%xmm3, %%xmm11\n"
          "movaps %%xmm3, %%xmm15\n"
          "addps %%xmm7, %%xmm11\n"
          "subps %%xmm7, %%xmm15\n"
          "movups %%xmm8, (%0)\n"
          "movups %%xmm9, (%1)\n"
          "movups %%xmm10, (%2)\n"
          "movups %%xmm11, (%3)\n"
          "movups %%xmm12, (%4)\n"
          "movups %%xmm13, (%5)\n"
          "movups %%xmm14, (%6)\n"
          "movups %%xmm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 32),
          "r"(buf + j + k + 64),
          "r"(buf + j + k + 96),
          "r"(buf + j + k + 128),
          "r"(buf + j + k + 160),
          "r"(buf + j + k + 192),
          "r"(buf + j + k + 224)
          : "%xmm0",
            "%xmm1",
            "%xmm2",
            "%xmm3",
            "%xmm4",
            "%xmm5",
            "%xmm6",
            "%xmm7",
            "%xmm8",
            "%xmm9",
            "%xmm10",
            "%xmm11",
            "%xmm12",
            "%xmm13",
            "%xmm14",
            "%xmm15",
            "memory");
    }
  }
  for (int j = 0; j < 1024; j += 1024) {
    for (int k = 0; k < 256; k += 4) {
      __asm__ volatile(
          "movups (%0), %%xmm0\n"
          "movups (%1), %%xmm1\n"
          "movups (%2), %%xmm2\n"
          "movups (%3), %%xmm3\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm9\n"
          "addps %%xmm1, %%xmm8\n"
          "subps %%xmm1, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "addps %%xmm3, %%xmm10\n"
          "subps %%xmm3, %%xmm11\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm8, %%xmm2\n"
          "addps %%xmm10, %%xmm0\n"
          "subps %%xmm10, %%xmm2\n"
          "movaps %%xmm9, %%xmm1\n"
          "movaps %%xmm9, %%xmm3\n"
          "addps %%xmm11, %%xmm1\n"
          "subps %%xmm11, %%xmm3\n"
          "movups %%xmm0, (%0)\n"
          "movups %%xmm1, (%1)\n"
          "movups %%xmm2, (%2)\n"
          "movups %%xmm3, (%3)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 256),
          "r"(buf + j + k + 512),
          "r"(buf + j + k + 768)
          : "%xmm0",
            "%xmm1",
            "%xmm2",
            "%xmm3",
            "%xmm4",
            "%xmm5",
            "%xmm6",
            "%xmm7",
            "%xmm8",
            "%xmm9",
            "%xmm10",
            "%xmm11",
            "%xmm12",
            "%xmm13",
            "%xmm14",
            "%xmm15",
            "memory");
    }
  }
}

ALAYA_NOINLINE
ALAYA_TARGET_AVX2
auto helper_float_11_avx2(float *buf) -> void {  // NOLINT
  for (int j = 0; j < 2048; j += 32) {
    for (int k = 0; k < 4; k += 4) {
      __asm__ volatile(
          "movups (%0), %%xmm0\n"
          "movups (%1), %%xmm1\n"
          "movups (%2), %%xmm2\n"
          "movups (%3), %%xmm3\n"
          "movups (%4), %%xmm4\n"
          "movups (%5), %%xmm5\n"
          "movups (%6), %%xmm6\n"
          "movups (%7), %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm0, %%xmm0\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm0, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm1, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm1, %%xmm1\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm1, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm1\n"
          "movaps %%xmm2, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm2, %%xmm2\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm2, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm2\n"
          "movaps %%xmm3, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm3, %%xmm3\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm3, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm3\n"
          "movaps %%xmm4, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm4, %%xmm4\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm4, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm4\n"
          "movaps %%xmm5, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm5, %%xmm5\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm5, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm5\n"
          "movaps %%xmm6, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm6, %%xmm6\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm6, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm6\n"
          "movaps %%xmm7, %%xmm8\n"
          "shufps $160, %%xmm8, %%xmm8\n"
          "shufps $245, %%xmm7, %%xmm7\n"
          "xorps %%xmm9, %%xmm9\n"
          "subps %%xmm7, %%xmm9\n"
          "addsubps %%xmm9, %%xmm8\n"
          "movaps %%xmm8, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm0, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm0, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm0\n"
          "movaps %%xmm1, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm1, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm1, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm1\n"
          "movaps %%xmm2, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm2\n"
          "movaps %%xmm3, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm3, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm3, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm3\n"
          "movaps %%xmm4, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm4, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm4, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm4\n"
          "movaps %%xmm5, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm5, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm5, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm5\n"
          "movaps %%xmm6, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm6, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm6, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm6\n"
          "movaps %%xmm7, %%xmm8\n"
          "shufps $68, %%xmm8, %%xmm8\n"
          "xorps %%xmm9, %%xmm9\n"
          "movaps %%xmm7, %%xmm10\n"
          "shufps $14, %%xmm9, %%xmm10\n"
          "movaps %%xmm7, %%xmm11\n"
          "shufps $224, %%xmm11, %%xmm9\n"
          "addps %%xmm8, %%xmm10\n"
          "subps %%xmm9, %%xmm10\n"
          "movaps %%xmm10, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm9\n"
          "addps %%xmm1, %%xmm8\n"
          "subps %%xmm1, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "addps %%xmm3, %%xmm10\n"
          "subps %%xmm3, %%xmm11\n"
          "movaps %%xmm4, %%xmm12\n"
          "movaps %%xmm4, %%xmm13\n"
          "addps %%xmm5, %%xmm12\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm6, %%xmm14\n"
          "movaps %%xmm6, %%xmm15\n"
          "addps %%xmm7, %%xmm14\n"
          "subps %%xmm7, %%xmm15\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm8, %%xmm2\n"
          "addps %%xmm10, %%xmm0\n"
          "subps %%xmm10, %%xmm2\n"
          "movaps %%xmm9, %%xmm1\n"
          "movaps %%xmm9, %%xmm3\n"
          "addps %%xmm11, %%xmm1\n"
          "subps %%xmm11, %%xmm3\n"
          "movaps %%xmm12, %%xmm4\n"
          "movaps %%xmm12, %%xmm6\n"
          "addps %%xmm14, %%xmm4\n"
          "subps %%xmm14, %%xmm6\n"
          "movaps %%xmm13, %%xmm5\n"
          "movaps %%xmm13, %%xmm7\n"
          "addps %%xmm15, %%xmm5\n"
          "subps %%xmm15, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm12\n"
          "addps %%xmm4, %%xmm8\n"
          "subps %%xmm4, %%xmm12\n"
          "movaps %%xmm1, %%xmm9\n"
          "movaps %%xmm1, %%xmm13\n"
          "addps %%xmm5, %%xmm9\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm14\n"
          "addps %%xmm6, %%xmm10\n"
          "subps %%xmm6, %%xmm14\n"
          "movaps %%xmm3, %%xmm11\n"
          "movaps %%xmm3, %%xmm15\n"
          "addps %%xmm7, %%xmm11\n"
          "subps %%xmm7, %%xmm15\n"
          "movups %%xmm8, (%0)\n"
          "movups %%xmm9, (%1)\n"
          "movups %%xmm10, (%2)\n"
          "movups %%xmm11, (%3)\n"
          "movups %%xmm12, (%4)\n"
          "movups %%xmm13, (%5)\n"
          "movups %%xmm14, (%6)\n"
          "movups %%xmm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 4),
          "r"(buf + j + k + 8),
          "r"(buf + j + k + 12),
          "r"(buf + j + k + 16),
          "r"(buf + j + k + 20),
          "r"(buf + j + k + 24),
          "r"(buf + j + k + 28)
          : "%xmm0",
            "%xmm1",
            "%xmm2",
            "%xmm3",
            "%xmm4",
            "%xmm5",
            "%xmm6",
            "%xmm7",
            "%xmm8",
            "%xmm9",
            "%xmm10",
            "%xmm11",
            "%xmm12",
            "%xmm13",
            "%xmm14",
            "%xmm15",
            "memory");
    }
  }
  for (int j = 0; j < 2048; j += 256) {
    for (int k = 0; k < 32; k += 4) {
      __asm__ volatile(
          "movups (%0), %%xmm0\n"
          "movups (%1), %%xmm1\n"
          "movups (%2), %%xmm2\n"
          "movups (%3), %%xmm3\n"
          "movups (%4), %%xmm4\n"
          "movups (%5), %%xmm5\n"
          "movups (%6), %%xmm6\n"
          "movups (%7), %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm9\n"
          "addps %%xmm1, %%xmm8\n"
          "subps %%xmm1, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "addps %%xmm3, %%xmm10\n"
          "subps %%xmm3, %%xmm11\n"
          "movaps %%xmm4, %%xmm12\n"
          "movaps %%xmm4, %%xmm13\n"
          "addps %%xmm5, %%xmm12\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm6, %%xmm14\n"
          "movaps %%xmm6, %%xmm15\n"
          "addps %%xmm7, %%xmm14\n"
          "subps %%xmm7, %%xmm15\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm8, %%xmm2\n"
          "addps %%xmm10, %%xmm0\n"
          "subps %%xmm10, %%xmm2\n"
          "movaps %%xmm9, %%xmm1\n"
          "movaps %%xmm9, %%xmm3\n"
          "addps %%xmm11, %%xmm1\n"
          "subps %%xmm11, %%xmm3\n"
          "movaps %%xmm12, %%xmm4\n"
          "movaps %%xmm12, %%xmm6\n"
          "addps %%xmm14, %%xmm4\n"
          "subps %%xmm14, %%xmm6\n"
          "movaps %%xmm13, %%xmm5\n"
          "movaps %%xmm13, %%xmm7\n"
          "addps %%xmm15, %%xmm5\n"
          "subps %%xmm15, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm12\n"
          "addps %%xmm4, %%xmm8\n"
          "subps %%xmm4, %%xmm12\n"
          "movaps %%xmm1, %%xmm9\n"
          "movaps %%xmm1, %%xmm13\n"
          "addps %%xmm5, %%xmm9\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm14\n"
          "addps %%xmm6, %%xmm10\n"
          "subps %%xmm6, %%xmm14\n"
          "movaps %%xmm3, %%xmm11\n"
          "movaps %%xmm3, %%xmm15\n"
          "addps %%xmm7, %%xmm11\n"
          "subps %%xmm7, %%xmm15\n"
          "movups %%xmm8, (%0)\n"
          "movups %%xmm9, (%1)\n"
          "movups %%xmm10, (%2)\n"
          "movups %%xmm11, (%3)\n"
          "movups %%xmm12, (%4)\n"
          "movups %%xmm13, (%5)\n"
          "movups %%xmm14, (%6)\n"
          "movups %%xmm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 32),
          "r"(buf + j + k + 64),
          "r"(buf + j + k + 96),
          "r"(buf + j + k + 128),
          "r"(buf + j + k + 160),
          "r"(buf + j + k + 192),
          "r"(buf + j + k + 224)
          : "%xmm0",
            "%xmm1",
            "%xmm2",
            "%xmm3",
            "%xmm4",
            "%xmm5",
            "%xmm6",
            "%xmm7",
            "%xmm8",
            "%xmm9",
            "%xmm10",
            "%xmm11",
            "%xmm12",
            "%xmm13",
            "%xmm14",
            "%xmm15",
            "memory");
    }
  }
  for (int j = 0; j < 2048; j += 2048) {
    for (int k = 0; k < 256; k += 4) {
      __asm__ volatile(
          "movups (%0), %%xmm0\n"
          "movups (%1), %%xmm1\n"
          "movups (%2), %%xmm2\n"
          "movups (%3), %%xmm3\n"
          "movups (%4), %%xmm4\n"
          "movups (%5), %%xmm5\n"
          "movups (%6), %%xmm6\n"
          "movups (%7), %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm9\n"
          "addps %%xmm1, %%xmm8\n"
          "subps %%xmm1, %%xmm9\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm11\n"
          "addps %%xmm3, %%xmm10\n"
          "subps %%xmm3, %%xmm11\n"
          "movaps %%xmm4, %%xmm12\n"
          "movaps %%xmm4, %%xmm13\n"
          "addps %%xmm5, %%xmm12\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm6, %%xmm14\n"
          "movaps %%xmm6, %%xmm15\n"
          "addps %%xmm7, %%xmm14\n"
          "subps %%xmm7, %%xmm15\n"
          "movaps %%xmm8, %%xmm0\n"
          "movaps %%xmm8, %%xmm2\n"
          "addps %%xmm10, %%xmm0\n"
          "subps %%xmm10, %%xmm2\n"
          "movaps %%xmm9, %%xmm1\n"
          "movaps %%xmm9, %%xmm3\n"
          "addps %%xmm11, %%xmm1\n"
          "subps %%xmm11, %%xmm3\n"
          "movaps %%xmm12, %%xmm4\n"
          "movaps %%xmm12, %%xmm6\n"
          "addps %%xmm14, %%xmm4\n"
          "subps %%xmm14, %%xmm6\n"
          "movaps %%xmm13, %%xmm5\n"
          "movaps %%xmm13, %%xmm7\n"
          "addps %%xmm15, %%xmm5\n"
          "subps %%xmm15, %%xmm7\n"
          "movaps %%xmm0, %%xmm8\n"
          "movaps %%xmm0, %%xmm12\n"
          "addps %%xmm4, %%xmm8\n"
          "subps %%xmm4, %%xmm12\n"
          "movaps %%xmm1, %%xmm9\n"
          "movaps %%xmm1, %%xmm13\n"
          "addps %%xmm5, %%xmm9\n"
          "subps %%xmm5, %%xmm13\n"
          "movaps %%xmm2, %%xmm10\n"
          "movaps %%xmm2, %%xmm14\n"
          "addps %%xmm6, %%xmm10\n"
          "subps %%xmm6, %%xmm14\n"
          "movaps %%xmm3, %%xmm11\n"
          "movaps %%xmm3, %%xmm15\n"
          "addps %%xmm7, %%xmm11\n"
          "subps %%xmm7, %%xmm15\n"
          "movups %%xmm8, (%0)\n"
          "movups %%xmm9, (%1)\n"
          "movups %%xmm10, (%2)\n"
          "movups %%xmm11, (%3)\n"
          "movups %%xmm12, (%4)\n"
          "movups %%xmm13, (%5)\n"
          "movups %%xmm14, (%6)\n"
          "movups %%xmm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 256),
          "r"(buf + j + k + 512),
          "r"(buf + j + k + 768),
          "r"(buf + j + k + 1024),
          "r"(buf + j + k + 1280),
          "r"(buf + j + k + 1536),
          "r"(buf + j + k + 1792)
          : "%xmm0",
            "%xmm1",
            "%xmm2",
            "%xmm3",
            "%xmm4",
            "%xmm5",
            "%xmm6",
            "%xmm7",
            "%xmm8",
            "%xmm9",
            "%xmm10",
            "%xmm11",
            "%xmm12",
            "%xmm13",
            "%xmm14",
            "%xmm15",
            "memory");
    }
  }
}

ALAYA_NOINLINE
ALAYA_TARGET_AVX512
inline auto helper_float_6_avx512(float *buf) -> void {  // NOLINT
  for (int j = 0; j < 64; j += 64) {
    for (int k = 0; k < 8; k += 8) {
      __asm__ volatile(
          "vmovups (%0), %%ymm0\n"
          "vmovups (%1), %%ymm1\n"
          "vmovups (%2), %%ymm2\n"
          "vmovups (%3), %%ymm3\n"
          "vmovups (%4), %%ymm4\n"
          "vmovups (%5), %%ymm5\n"
          "vmovups (%6), %%ymm6\n"
          "vmovups (%7), %%ymm7\n"
          "vpermilps $160, %%ymm0, %%ymm8\n"
          "vpermilps $245, %%ymm0, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm0\n"
          "vpermilps $160, %%ymm1, %%ymm8\n"
          "vpermilps $245, %%ymm1, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm1\n"
          "vpermilps $160, %%ymm2, %%ymm8\n"
          "vpermilps $245, %%ymm2, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm2\n"
          "vpermilps $160, %%ymm3, %%ymm8\n"
          "vpermilps $245, %%ymm3, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm3\n"
          "vpermilps $160, %%ymm4, %%ymm8\n"
          "vpermilps $245, %%ymm4, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm4\n"
          "vpermilps $160, %%ymm5, %%ymm8\n"
          "vpermilps $245, %%ymm5, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm5\n"
          "vpermilps $160, %%ymm6, %%ymm8\n"
          "vpermilps $245, %%ymm6, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm6\n"
          "vpermilps $160, %%ymm7, %%ymm8\n"
          "vpermilps $245, %%ymm7, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm7\n"
          "vpermilps $68, %%ymm0, %%ymm8\n"
          "vpermilps $238, %%ymm0, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm0\n"
          "vpermilps $68, %%ymm1, %%ymm8\n"
          "vpermilps $238, %%ymm1, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm1\n"
          "vpermilps $68, %%ymm2, %%ymm8\n"
          "vpermilps $238, %%ymm2, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm2\n"
          "vpermilps $68, %%ymm3, %%ymm8\n"
          "vpermilps $238, %%ymm3, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm3\n"
          "vpermilps $68, %%ymm4, %%ymm8\n"
          "vpermilps $238, %%ymm4, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm4\n"
          "vpermilps $68, %%ymm5, %%ymm8\n"
          "vpermilps $238, %%ymm5, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm5\n"
          "vpermilps $68, %%ymm6, %%ymm8\n"
          "vpermilps $238, %%ymm6, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm6\n"
          "vpermilps $68, %%ymm7, %%ymm8\n"
          "vpermilps $238, %%ymm7, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm7\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm0, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm0, %%ymm0, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm0, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm0\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm1, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm1, %%ymm1, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm1, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm1\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm2, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm2, %%ymm2, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm2, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm2\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm3, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm3, %%ymm3, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm3, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm3\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm4, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm4, %%ymm4, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm4, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm4\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm5, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm5, %%ymm5, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm5, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm5\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm6, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm6, %%ymm6, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm6, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm6\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm7, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm7, %%ymm7, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm7, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm7\n"
          "vaddps %%ymm1, %%ymm0, %%ymm8\n"
          "vsubps %%ymm1, %%ymm0, %%ymm9\n"
          "vaddps %%ymm3, %%ymm2, %%ymm10\n"
          "vsubps %%ymm3, %%ymm2, %%ymm11\n"
          "vaddps %%ymm5, %%ymm4, %%ymm12\n"
          "vsubps %%ymm5, %%ymm4, %%ymm13\n"
          "vaddps %%ymm7, %%ymm6, %%ymm14\n"
          "vsubps %%ymm7, %%ymm6, %%ymm15\n"
          "vaddps %%ymm10, %%ymm8, %%ymm0\n"
          "vsubps %%ymm10, %%ymm8, %%ymm2\n"
          "vaddps %%ymm11, %%ymm9, %%ymm1\n"
          "vsubps %%ymm11, %%ymm9, %%ymm3\n"
          "vaddps %%ymm14, %%ymm12, %%ymm4\n"
          "vsubps %%ymm14, %%ymm12, %%ymm6\n"
          "vaddps %%ymm15, %%ymm13, %%ymm5\n"
          "vsubps %%ymm15, %%ymm13, %%ymm7\n"
          "vaddps %%ymm4, %%ymm0, %%ymm8\n"
          "vsubps %%ymm4, %%ymm0, %%ymm12\n"
          "vaddps %%ymm5, %%ymm1, %%ymm9\n"
          "vsubps %%ymm5, %%ymm1, %%ymm13\n"
          "vaddps %%ymm6, %%ymm2, %%ymm10\n"
          "vsubps %%ymm6, %%ymm2, %%ymm14\n"
          "vaddps %%ymm7, %%ymm3, %%ymm11\n"
          "vsubps %%ymm7, %%ymm3, %%ymm15\n"
          "vmovups %%ymm8, (%0)\n"
          "vmovups %%ymm9, (%1)\n"
          "vmovups %%ymm10, (%2)\n"
          "vmovups %%ymm11, (%3)\n"
          "vmovups %%ymm12, (%4)\n"
          "vmovups %%ymm13, (%5)\n"
          "vmovups %%ymm14, (%6)\n"
          "vmovups %%ymm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 8),
          "r"(buf + j + k + 16),
          "r"(buf + j + k + 24),
          "r"(buf + j + k + 32),
          "r"(buf + j + k + 40),
          "r"(buf + j + k + 48),
          "r"(buf + j + k + 56)
          : "%ymm0",
            "%ymm1",
            "%ymm2",
            "%ymm3",
            "%ymm4",
            "%ymm5",
            "%ymm6",
            "%ymm7",
            "%ymm8",
            "%ymm9",
            "%ymm10",
            "%ymm11",
            "%ymm12",
            "%ymm13",
            "%ymm14",
            "%ymm15",
            "memory");
    }
  }
}

ALAYA_NOINLINE
ALAYA_TARGET_AVX512
inline auto helper_float_7_avx512(float *buf) -> void {  // NOLINT
  for (int j = 0; j < 128; j += 64) {
    for (int k = 0; k < 8; k += 8) {
      __asm__ volatile(
          "vmovups (%0), %%ymm0\n"
          "vmovups (%1), %%ymm1\n"
          "vmovups (%2), %%ymm2\n"
          "vmovups (%3), %%ymm3\n"
          "vmovups (%4), %%ymm4\n"
          "vmovups (%5), %%ymm5\n"
          "vmovups (%6), %%ymm6\n"
          "vmovups (%7), %%ymm7\n"
          "vpermilps $160, %%ymm0, %%ymm8\n"
          "vpermilps $245, %%ymm0, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm0\n"
          "vpermilps $160, %%ymm1, %%ymm8\n"
          "vpermilps $245, %%ymm1, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm1\n"
          "vpermilps $160, %%ymm2, %%ymm8\n"
          "vpermilps $245, %%ymm2, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm2\n"
          "vpermilps $160, %%ymm3, %%ymm8\n"
          "vpermilps $245, %%ymm3, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm3\n"
          "vpermilps $160, %%ymm4, %%ymm8\n"
          "vpermilps $245, %%ymm4, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm4\n"
          "vpermilps $160, %%ymm5, %%ymm8\n"
          "vpermilps $245, %%ymm5, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm5\n"
          "vpermilps $160, %%ymm6, %%ymm8\n"
          "vpermilps $245, %%ymm6, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm6\n"
          "vpermilps $160, %%ymm7, %%ymm8\n"
          "vpermilps $245, %%ymm7, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm7\n"
          "vpermilps $68, %%ymm0, %%ymm8\n"
          "vpermilps $238, %%ymm0, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm0\n"
          "vpermilps $68, %%ymm1, %%ymm8\n"
          "vpermilps $238, %%ymm1, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm1\n"
          "vpermilps $68, %%ymm2, %%ymm8\n"
          "vpermilps $238, %%ymm2, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm2\n"
          "vpermilps $68, %%ymm3, %%ymm8\n"
          "vpermilps $238, %%ymm3, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm3\n"
          "vpermilps $68, %%ymm4, %%ymm8\n"
          "vpermilps $238, %%ymm4, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm4\n"
          "vpermilps $68, %%ymm5, %%ymm8\n"
          "vpermilps $238, %%ymm5, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm5\n"
          "vpermilps $68, %%ymm6, %%ymm8\n"
          "vpermilps $238, %%ymm6, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm6\n"
          "vpermilps $68, %%ymm7, %%ymm8\n"
          "vpermilps $238, %%ymm7, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm7\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm0, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm0, %%ymm0, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm0, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm0\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm1, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm1, %%ymm1, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm1, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm1\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm2, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm2, %%ymm2, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm2, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm2\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm3, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm3, %%ymm3, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm3, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm3\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm4, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm4, %%ymm4, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm4, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm4\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm5, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm5, %%ymm5, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm5, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm5\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm6, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm6, %%ymm6, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm6, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm6\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm7, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm7, %%ymm7, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm7, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm7\n"
          "vaddps %%ymm1, %%ymm0, %%ymm8\n"
          "vsubps %%ymm1, %%ymm0, %%ymm9\n"
          "vaddps %%ymm3, %%ymm2, %%ymm10\n"
          "vsubps %%ymm3, %%ymm2, %%ymm11\n"
          "vaddps %%ymm5, %%ymm4, %%ymm12\n"
          "vsubps %%ymm5, %%ymm4, %%ymm13\n"
          "vaddps %%ymm7, %%ymm6, %%ymm14\n"
          "vsubps %%ymm7, %%ymm6, %%ymm15\n"
          "vaddps %%ymm10, %%ymm8, %%ymm0\n"
          "vsubps %%ymm10, %%ymm8, %%ymm2\n"
          "vaddps %%ymm11, %%ymm9, %%ymm1\n"
          "vsubps %%ymm11, %%ymm9, %%ymm3\n"
          "vaddps %%ymm14, %%ymm12, %%ymm4\n"
          "vsubps %%ymm14, %%ymm12, %%ymm6\n"
          "vaddps %%ymm15, %%ymm13, %%ymm5\n"
          "vsubps %%ymm15, %%ymm13, %%ymm7\n"
          "vaddps %%ymm4, %%ymm0, %%ymm8\n"
          "vsubps %%ymm4, %%ymm0, %%ymm12\n"
          "vaddps %%ymm5, %%ymm1, %%ymm9\n"
          "vsubps %%ymm5, %%ymm1, %%ymm13\n"
          "vaddps %%ymm6, %%ymm2, %%ymm10\n"
          "vsubps %%ymm6, %%ymm2, %%ymm14\n"
          "vaddps %%ymm7, %%ymm3, %%ymm11\n"
          "vsubps %%ymm7, %%ymm3, %%ymm15\n"
          "vmovups %%ymm8, (%0)\n"
          "vmovups %%ymm9, (%1)\n"
          "vmovups %%ymm10, (%2)\n"
          "vmovups %%ymm11, (%3)\n"
          "vmovups %%ymm12, (%4)\n"
          "vmovups %%ymm13, (%5)\n"
          "vmovups %%ymm14, (%6)\n"
          "vmovups %%ymm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 8),
          "r"(buf + j + k + 16),
          "r"(buf + j + k + 24),
          "r"(buf + j + k + 32),
          "r"(buf + j + k + 40),
          "r"(buf + j + k + 48),
          "r"(buf + j + k + 56)
          : "%ymm0",
            "%ymm1",
            "%ymm2",
            "%ymm3",
            "%ymm4",
            "%ymm5",
            "%ymm6",
            "%ymm7",
            "%ymm8",
            "%ymm9",
            "%ymm10",
            "%ymm11",
            "%ymm12",
            "%ymm13",
            "%ymm14",
            "%ymm15",
            "memory");
    }
  }
  for (int j = 0; j < 128; j += 128) {
    for (int k = 0; k < 64; k += 8) {
      __asm__ volatile(
          "vmovups (%0), %%ymm0\n"
          "vmovups (%1), %%ymm1\n"
          "vaddps %%ymm1, %%ymm0, %%ymm8\n"
          "vsubps %%ymm1, %%ymm0, %%ymm9\n"
          "vmovups %%ymm8, (%0)\n"
          "vmovups %%ymm9, (%1)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 64)
          : "%ymm0",
            "%ymm1",
            "%ymm2",
            "%ymm3",
            "%ymm4",
            "%ymm5",
            "%ymm6",
            "%ymm7",
            "%ymm8",
            "%ymm9",
            "%ymm10",
            "%ymm11",
            "%ymm12",
            "%ymm13",
            "%ymm14",
            "%ymm15",
            "memory");
    }
  }
}

ALAYA_ALWAYS_INLINE
ALAYA_TARGET_AVX512
auto helper_float_8_avx512_recursive(float *buf, int depth) -> void {  // NOLINT
  if (depth == 6) {
    for (int j = 0; j < 64; j += 64) {
      for (int k = 0; k < 8; k += 8) {
        __asm__ volatile(
            "vmovups (%0), %%ymm0\n"
            "vmovups (%1), %%ymm1\n"
            "vmovups (%2), %%ymm2\n"
            "vmovups (%3), %%ymm3\n"
            "vmovups (%4), %%ymm4\n"
            "vmovups (%5), %%ymm5\n"
            "vmovups (%6), %%ymm6\n"
            "vmovups (%7), %%ymm7\n"
            "vpermilps $160, %%ymm0, %%ymm8\n"
            "vpermilps $245, %%ymm0, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vaddsubps %%ymm11, %%ymm8, %%ymm0\n"
            "vpermilps $160, %%ymm1, %%ymm8\n"
            "vpermilps $245, %%ymm1, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vaddsubps %%ymm11, %%ymm8, %%ymm1\n"
            "vpermilps $160, %%ymm2, %%ymm8\n"
            "vpermilps $245, %%ymm2, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vaddsubps %%ymm11, %%ymm8, %%ymm2\n"
            "vpermilps $160, %%ymm3, %%ymm8\n"
            "vpermilps $245, %%ymm3, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vaddsubps %%ymm11, %%ymm8, %%ymm3\n"
            "vpermilps $160, %%ymm4, %%ymm8\n"
            "vpermilps $245, %%ymm4, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vaddsubps %%ymm11, %%ymm8, %%ymm4\n"
            "vpermilps $160, %%ymm5, %%ymm8\n"
            "vpermilps $245, %%ymm5, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vaddsubps %%ymm11, %%ymm8, %%ymm5\n"
            "vpermilps $160, %%ymm6, %%ymm8\n"
            "vpermilps $245, %%ymm6, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vaddsubps %%ymm11, %%ymm8, %%ymm6\n"
            "vpermilps $160, %%ymm7, %%ymm8\n"
            "vpermilps $245, %%ymm7, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vaddsubps %%ymm11, %%ymm8, %%ymm7\n"
            "vpermilps $68, %%ymm0, %%ymm8\n"
            "vpermilps $238, %%ymm0, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
            "vaddps %%ymm8, %%ymm12, %%ymm0\n"
            "vpermilps $68, %%ymm1, %%ymm8\n"
            "vpermilps $238, %%ymm1, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
            "vaddps %%ymm8, %%ymm12, %%ymm1\n"
            "vpermilps $68, %%ymm2, %%ymm8\n"
            "vpermilps $238, %%ymm2, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
            "vaddps %%ymm8, %%ymm12, %%ymm2\n"
            "vpermilps $68, %%ymm3, %%ymm8\n"
            "vpermilps $238, %%ymm3, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
            "vaddps %%ymm8, %%ymm12, %%ymm3\n"
            "vpermilps $68, %%ymm4, %%ymm8\n"
            "vpermilps $238, %%ymm4, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
            "vaddps %%ymm8, %%ymm12, %%ymm4\n"
            "vpermilps $68, %%ymm5, %%ymm8\n"
            "vpermilps $238, %%ymm5, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
            "vaddps %%ymm8, %%ymm12, %%ymm5\n"
            "vpermilps $68, %%ymm6, %%ymm8\n"
            "vpermilps $238, %%ymm6, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
            "vaddps %%ymm8, %%ymm12, %%ymm6\n"
            "vpermilps $68, %%ymm7, %%ymm8\n"
            "vpermilps $238, %%ymm7, %%ymm9\n"
            "vxorps %%ymm10, %%ymm10, %%ymm10\n"
            "vsubps %%ymm9, %%ymm10, %%ymm11\n"
            "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
            "vaddps %%ymm8, %%ymm12, %%ymm7\n"
            "vxorps %%ymm8, %%ymm8, %%ymm8\n"
            "vsubps %%ymm0, %%ymm8, %%ymm9\n"
            "vperm2f128 $0, %%ymm0, %%ymm0, %%ymm10\n"
            "vperm2f128 $49, %%ymm9, %%ymm0, %%ymm11\n"
            "vaddps %%ymm10, %%ymm11, %%ymm0\n"
            "vxorps %%ymm8, %%ymm8, %%ymm8\n"
            "vsubps %%ymm1, %%ymm8, %%ymm9\n"
            "vperm2f128 $0, %%ymm1, %%ymm1, %%ymm10\n"
            "vperm2f128 $49, %%ymm9, %%ymm1, %%ymm11\n"
            "vaddps %%ymm10, %%ymm11, %%ymm1\n"
            "vxorps %%ymm8, %%ymm8, %%ymm8\n"
            "vsubps %%ymm2, %%ymm8, %%ymm9\n"
            "vperm2f128 $0, %%ymm2, %%ymm2, %%ymm10\n"
            "vperm2f128 $49, %%ymm9, %%ymm2, %%ymm11\n"
            "vaddps %%ymm10, %%ymm11, %%ymm2\n"
            "vxorps %%ymm8, %%ymm8, %%ymm8\n"
            "vsubps %%ymm3, %%ymm8, %%ymm9\n"
            "vperm2f128 $0, %%ymm3, %%ymm3, %%ymm10\n"
            "vperm2f128 $49, %%ymm9, %%ymm3, %%ymm11\n"
            "vaddps %%ymm10, %%ymm11, %%ymm3\n"
            "vxorps %%ymm8, %%ymm8, %%ymm8\n"
            "vsubps %%ymm4, %%ymm8, %%ymm9\n"
            "vperm2f128 $0, %%ymm4, %%ymm4, %%ymm10\n"
            "vperm2f128 $49, %%ymm9, %%ymm4, %%ymm11\n"
            "vaddps %%ymm10, %%ymm11, %%ymm4\n"
            "vxorps %%ymm8, %%ymm8, %%ymm8\n"
            "vsubps %%ymm5, %%ymm8, %%ymm9\n"
            "vperm2f128 $0, %%ymm5, %%ymm5, %%ymm10\n"
            "vperm2f128 $49, %%ymm9, %%ymm5, %%ymm11\n"
            "vaddps %%ymm10, %%ymm11, %%ymm5\n"
            "vxorps %%ymm8, %%ymm8, %%ymm8\n"
            "vsubps %%ymm6, %%ymm8, %%ymm9\n"
            "vperm2f128 $0, %%ymm6, %%ymm6, %%ymm10\n"
            "vperm2f128 $49, %%ymm9, %%ymm6, %%ymm11\n"
            "vaddps %%ymm10, %%ymm11, %%ymm6\n"
            "vxorps %%ymm8, %%ymm8, %%ymm8\n"
            "vsubps %%ymm7, %%ymm8, %%ymm9\n"
            "vperm2f128 $0, %%ymm7, %%ymm7, %%ymm10\n"
            "vperm2f128 $49, %%ymm9, %%ymm7, %%ymm11\n"
            "vaddps %%ymm10, %%ymm11, %%ymm7\n"
            "vaddps %%ymm1, %%ymm0, %%ymm8\n"
            "vsubps %%ymm1, %%ymm0, %%ymm9\n"
            "vaddps %%ymm3, %%ymm2, %%ymm10\n"
            "vsubps %%ymm3, %%ymm2, %%ymm11\n"
            "vaddps %%ymm5, %%ymm4, %%ymm12\n"
            "vsubps %%ymm5, %%ymm4, %%ymm13\n"
            "vaddps %%ymm7, %%ymm6, %%ymm14\n"
            "vsubps %%ymm7, %%ymm6, %%ymm15\n"
            "vaddps %%ymm10, %%ymm8, %%ymm0\n"
            "vsubps %%ymm10, %%ymm8, %%ymm2\n"
            "vaddps %%ymm11, %%ymm9, %%ymm1\n"
            "vsubps %%ymm11, %%ymm9, %%ymm3\n"
            "vaddps %%ymm14, %%ymm12, %%ymm4\n"
            "vsubps %%ymm14, %%ymm12, %%ymm6\n"
            "vaddps %%ymm15, %%ymm13, %%ymm5\n"
            "vsubps %%ymm15, %%ymm13, %%ymm7\n"
            "vaddps %%ymm4, %%ymm0, %%ymm8\n"
            "vsubps %%ymm4, %%ymm0, %%ymm12\n"
            "vaddps %%ymm5, %%ymm1, %%ymm9\n"
            "vsubps %%ymm5, %%ymm1, %%ymm13\n"
            "vaddps %%ymm6, %%ymm2, %%ymm10\n"
            "vsubps %%ymm6, %%ymm2, %%ymm14\n"
            "vaddps %%ymm7, %%ymm3, %%ymm11\n"
            "vsubps %%ymm7, %%ymm3, %%ymm15\n"
            "vmovups %%ymm8, (%0)\n"
            "vmovups %%ymm9, (%1)\n"
            "vmovups %%ymm10, (%2)\n"
            "vmovups %%ymm11, (%3)\n"
            "vmovups %%ymm12, (%4)\n"
            "vmovups %%ymm13, (%5)\n"
            "vmovups %%ymm14, (%6)\n"
            "vmovups %%ymm15, (%7)\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 8),
            "r"(buf + j + k + 16),
            "r"(buf + j + k + 24),
            "r"(buf + j + k + 32),
            "r"(buf + j + k + 40),
            "r"(buf + j + k + 48),
            "r"(buf + j + k + 56)
            : "%ymm0",
              "%ymm1",
              "%ymm2",
              "%ymm3",
              "%ymm4",
              "%ymm5",
              "%ymm6",
              "%ymm7",
              "%ymm8",
              "%ymm9",
              "%ymm10",
              "%ymm11",
              "%ymm12",
              "%ymm13",
              "%ymm14",
              "%ymm15",
              "memory");
      }
    }
    return;
  }
  if (depth == 8) {
    helper_float_8_avx512_recursive(buf + 0, 6);
    helper_float_8_avx512_recursive(buf + 64, 6);
    helper_float_8_avx512_recursive(buf + 128, 6);
    helper_float_8_avx512_recursive(buf + 192, 6);
    for (int j = 0; j < 256; j += 256) {
      for (int k = 0; k < 64; k += 8) {
        __asm__ volatile(
            "vmovups (%0), %%ymm0\n"
            "vmovups (%1), %%ymm1\n"
            "vmovups (%2), %%ymm2\n"
            "vmovups (%3), %%ymm3\n"
            "vaddps %%ymm1, %%ymm0, %%ymm8\n"
            "vsubps %%ymm1, %%ymm0, %%ymm9\n"
            "vaddps %%ymm3, %%ymm2, %%ymm10\n"
            "vsubps %%ymm3, %%ymm2, %%ymm11\n"
            "vaddps %%ymm10, %%ymm8, %%ymm0\n"
            "vsubps %%ymm10, %%ymm8, %%ymm2\n"
            "vaddps %%ymm11, %%ymm9, %%ymm1\n"
            "vsubps %%ymm11, %%ymm9, %%ymm3\n"
            "vmovups %%ymm0, (%0)\n"
            "vmovups %%ymm1, (%1)\n"
            "vmovups %%ymm2, (%2)\n"
            "vmovups %%ymm3, (%3)\n" ::"r"(buf + j + k + 0),
            "r"(buf + j + k + 64),
            "r"(buf + j + k + 128),
            "r"(buf + j + k + 192)
            : "%ymm0",
              "%ymm1",
              "%ymm2",
              "%ymm3",
              "%ymm4",
              "%ymm5",
              "%ymm6",
              "%ymm7",
              "%ymm8",
              "%ymm9",
              "%ymm10",
              "%ymm11",
              "%ymm12",
              "%ymm13",
              "%ymm14",
              "%ymm15",
              "memory");
      }
    }
    return;
  }
}

ALAYA_NOINLINE
ALAYA_TARGET_AVX512
inline auto helper_float_8_avx512(float *buf) -> void { helper_float_8_avx512_recursive(buf, 8); }

ALAYA_NOINLINE
ALAYA_TARGET_AVX512
inline auto helper_float_9_avx512(float *buf) -> void {  // NOLINT
  for (int j = 0; j < 512; j += 64) {
    for (int k = 0; k < 8; k += 8) {
      __asm__ volatile(
          "vmovups (%0), %%ymm0\n"
          "vmovups (%1), %%ymm1\n"
          "vmovups (%2), %%ymm2\n"
          "vmovups (%3), %%ymm3\n"
          "vmovups (%4), %%ymm4\n"
          "vmovups (%5), %%ymm5\n"
          "vmovups (%6), %%ymm6\n"
          "vmovups (%7), %%ymm7\n"
          "vpermilps $160, %%ymm0, %%ymm8\n"
          "vpermilps $245, %%ymm0, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm0\n"
          "vpermilps $160, %%ymm1, %%ymm8\n"
          "vpermilps $245, %%ymm1, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm1\n"
          "vpermilps $160, %%ymm2, %%ymm8\n"
          "vpermilps $245, %%ymm2, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm2\n"
          "vpermilps $160, %%ymm3, %%ymm8\n"
          "vpermilps $245, %%ymm3, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm3\n"
          "vpermilps $160, %%ymm4, %%ymm8\n"
          "vpermilps $245, %%ymm4, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm4\n"
          "vpermilps $160, %%ymm5, %%ymm8\n"
          "vpermilps $245, %%ymm5, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm5\n"
          "vpermilps $160, %%ymm6, %%ymm8\n"
          "vpermilps $245, %%ymm6, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm6\n"
          "vpermilps $160, %%ymm7, %%ymm8\n"
          "vpermilps $245, %%ymm7, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm7\n"
          "vpermilps $68, %%ymm0, %%ymm8\n"
          "vpermilps $238, %%ymm0, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm0\n"
          "vpermilps $68, %%ymm1, %%ymm8\n"
          "vpermilps $238, %%ymm1, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm1\n"
          "vpermilps $68, %%ymm2, %%ymm8\n"
          "vpermilps $238, %%ymm2, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm2\n"
          "vpermilps $68, %%ymm3, %%ymm8\n"
          "vpermilps $238, %%ymm3, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm3\n"
          "vpermilps $68, %%ymm4, %%ymm8\n"
          "vpermilps $238, %%ymm4, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm4\n"
          "vpermilps $68, %%ymm5, %%ymm8\n"
          "vpermilps $238, %%ymm5, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm5\n"
          "vpermilps $68, %%ymm6, %%ymm8\n"
          "vpermilps $238, %%ymm6, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm6\n"
          "vpermilps $68, %%ymm7, %%ymm8\n"
          "vpermilps $238, %%ymm7, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm7\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm0, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm0, %%ymm0, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm0, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm0\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm1, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm1, %%ymm1, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm1, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm1\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm2, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm2, %%ymm2, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm2, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm2\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm3, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm3, %%ymm3, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm3, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm3\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm4, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm4, %%ymm4, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm4, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm4\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm5, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm5, %%ymm5, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm5, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm5\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm6, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm6, %%ymm6, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm6, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm6\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm7, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm7, %%ymm7, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm7, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm7\n"
          "vaddps %%ymm1, %%ymm0, %%ymm8\n"
          "vsubps %%ymm1, %%ymm0, %%ymm9\n"
          "vaddps %%ymm3, %%ymm2, %%ymm10\n"
          "vsubps %%ymm3, %%ymm2, %%ymm11\n"
          "vaddps %%ymm5, %%ymm4, %%ymm12\n"
          "vsubps %%ymm5, %%ymm4, %%ymm13\n"
          "vaddps %%ymm7, %%ymm6, %%ymm14\n"
          "vsubps %%ymm7, %%ymm6, %%ymm15\n"
          "vaddps %%ymm10, %%ymm8, %%ymm0\n"
          "vsubps %%ymm10, %%ymm8, %%ymm2\n"
          "vaddps %%ymm11, %%ymm9, %%ymm1\n"
          "vsubps %%ymm11, %%ymm9, %%ymm3\n"
          "vaddps %%ymm14, %%ymm12, %%ymm4\n"
          "vsubps %%ymm14, %%ymm12, %%ymm6\n"
          "vaddps %%ymm15, %%ymm13, %%ymm5\n"
          "vsubps %%ymm15, %%ymm13, %%ymm7\n"
          "vaddps %%ymm4, %%ymm0, %%ymm8\n"
          "vsubps %%ymm4, %%ymm0, %%ymm12\n"
          "vaddps %%ymm5, %%ymm1, %%ymm9\n"
          "vsubps %%ymm5, %%ymm1, %%ymm13\n"
          "vaddps %%ymm6, %%ymm2, %%ymm10\n"
          "vsubps %%ymm6, %%ymm2, %%ymm14\n"
          "vaddps %%ymm7, %%ymm3, %%ymm11\n"
          "vsubps %%ymm7, %%ymm3, %%ymm15\n"
          "vmovups %%ymm8, (%0)\n"
          "vmovups %%ymm9, (%1)\n"
          "vmovups %%ymm10, (%2)\n"
          "vmovups %%ymm11, (%3)\n"
          "vmovups %%ymm12, (%4)\n"
          "vmovups %%ymm13, (%5)\n"
          "vmovups %%ymm14, (%6)\n"
          "vmovups %%ymm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 8),
          "r"(buf + j + k + 16),
          "r"(buf + j + k + 24),
          "r"(buf + j + k + 32),
          "r"(buf + j + k + 40),
          "r"(buf + j + k + 48),
          "r"(buf + j + k + 56)
          : "%ymm0",
            "%ymm1",
            "%ymm2",
            "%ymm3",
            "%ymm4",
            "%ymm5",
            "%ymm6",
            "%ymm7",
            "%ymm8",
            "%ymm9",
            "%ymm10",
            "%ymm11",
            "%ymm12",
            "%ymm13",
            "%ymm14",
            "%ymm15",
            "memory");
    }
  }
  for (int j = 0; j < 512; j += 512) {
    for (int k = 0; k < 64; k += 8) {
      __asm__ volatile(
          "vmovups (%0), %%ymm0\n"
          "vmovups (%1), %%ymm1\n"
          "vmovups (%2), %%ymm2\n"
          "vmovups (%3), %%ymm3\n"
          "vmovups (%4), %%ymm4\n"
          "vmovups (%5), %%ymm5\n"
          "vmovups (%6), %%ymm6\n"
          "vmovups (%7), %%ymm7\n"
          "vaddps %%ymm1, %%ymm0, %%ymm8\n"
          "vsubps %%ymm1, %%ymm0, %%ymm9\n"
          "vaddps %%ymm3, %%ymm2, %%ymm10\n"
          "vsubps %%ymm3, %%ymm2, %%ymm11\n"
          "vaddps %%ymm5, %%ymm4, %%ymm12\n"
          "vsubps %%ymm5, %%ymm4, %%ymm13\n"
          "vaddps %%ymm7, %%ymm6, %%ymm14\n"
          "vsubps %%ymm7, %%ymm6, %%ymm15\n"
          "vaddps %%ymm10, %%ymm8, %%ymm0\n"
          "vsubps %%ymm10, %%ymm8, %%ymm2\n"
          "vaddps %%ymm11, %%ymm9, %%ymm1\n"
          "vsubps %%ymm11, %%ymm9, %%ymm3\n"
          "vaddps %%ymm14, %%ymm12, %%ymm4\n"
          "vsubps %%ymm14, %%ymm12, %%ymm6\n"
          "vaddps %%ymm15, %%ymm13, %%ymm5\n"
          "vsubps %%ymm15, %%ymm13, %%ymm7\n"
          "vaddps %%ymm4, %%ymm0, %%ymm8\n"
          "vsubps %%ymm4, %%ymm0, %%ymm12\n"
          "vaddps %%ymm5, %%ymm1, %%ymm9\n"
          "vsubps %%ymm5, %%ymm1, %%ymm13\n"
          "vaddps %%ymm6, %%ymm2, %%ymm10\n"
          "vsubps %%ymm6, %%ymm2, %%ymm14\n"
          "vaddps %%ymm7, %%ymm3, %%ymm11\n"
          "vsubps %%ymm7, %%ymm3, %%ymm15\n"
          "vmovups %%ymm8, (%0)\n"
          "vmovups %%ymm9, (%1)\n"
          "vmovups %%ymm10, (%2)\n"
          "vmovups %%ymm11, (%3)\n"
          "vmovups %%ymm12, (%4)\n"
          "vmovups %%ymm13, (%5)\n"
          "vmovups %%ymm14, (%6)\n"
          "vmovups %%ymm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 64),
          "r"(buf + j + k + 128),
          "r"(buf + j + k + 192),
          "r"(buf + j + k + 256),
          "r"(buf + j + k + 320),
          "r"(buf + j + k + 384),
          "r"(buf + j + k + 448)
          : "%ymm0",
            "%ymm1",
            "%ymm2",
            "%ymm3",
            "%ymm4",
            "%ymm5",
            "%ymm6",
            "%ymm7",
            "%ymm8",
            "%ymm9",
            "%ymm10",
            "%ymm11",
            "%ymm12",
            "%ymm13",
            "%ymm14",
            "%ymm15",
            "memory");
    }
  }
}

ALAYA_NOINLINE
ALAYA_TARGET_AVX512
inline auto helper_float_10_avx512(float *buf) -> void {  // NOLINT
  for (int j = 0; j < 1024; j += 64) {
    for (int k = 0; k < 8; k += 8) {
      __asm__ volatile(
          "vmovups (%0), %%ymm0\n"
          "vmovups (%1), %%ymm1\n"
          "vmovups (%2), %%ymm2\n"
          "vmovups (%3), %%ymm3\n"
          "vmovups (%4), %%ymm4\n"
          "vmovups (%5), %%ymm5\n"
          "vmovups (%6), %%ymm6\n"
          "vmovups (%7), %%ymm7\n"
          "vpermilps $160, %%ymm0, %%ymm8\n"
          "vpermilps $245, %%ymm0, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm0\n"
          "vpermilps $160, %%ymm1, %%ymm8\n"
          "vpermilps $245, %%ymm1, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm1\n"
          "vpermilps $160, %%ymm2, %%ymm8\n"
          "vpermilps $245, %%ymm2, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm2\n"
          "vpermilps $160, %%ymm3, %%ymm8\n"
          "vpermilps $245, %%ymm3, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm3\n"
          "vpermilps $160, %%ymm4, %%ymm8\n"
          "vpermilps $245, %%ymm4, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm4\n"
          "vpermilps $160, %%ymm5, %%ymm8\n"
          "vpermilps $245, %%ymm5, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm5\n"
          "vpermilps $160, %%ymm6, %%ymm8\n"
          "vpermilps $245, %%ymm6, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm6\n"
          "vpermilps $160, %%ymm7, %%ymm8\n"
          "vpermilps $245, %%ymm7, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm7\n"
          "vpermilps $68, %%ymm0, %%ymm8\n"
          "vpermilps $238, %%ymm0, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm0\n"
          "vpermilps $68, %%ymm1, %%ymm8\n"
          "vpermilps $238, %%ymm1, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm1\n"
          "vpermilps $68, %%ymm2, %%ymm8\n"
          "vpermilps $238, %%ymm2, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm2\n"
          "vpermilps $68, %%ymm3, %%ymm8\n"
          "vpermilps $238, %%ymm3, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm3\n"
          "vpermilps $68, %%ymm4, %%ymm8\n"
          "vpermilps $238, %%ymm4, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm4\n"
          "vpermilps $68, %%ymm5, %%ymm8\n"
          "vpermilps $238, %%ymm5, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm5\n"
          "vpermilps $68, %%ymm6, %%ymm8\n"
          "vpermilps $238, %%ymm6, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm6\n"
          "vpermilps $68, %%ymm7, %%ymm8\n"
          "vpermilps $238, %%ymm7, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm7\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm0, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm0, %%ymm0, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm0, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm0\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm1, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm1, %%ymm1, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm1, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm1\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm2, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm2, %%ymm2, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm2, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm2\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm3, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm3, %%ymm3, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm3, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm3\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm4, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm4, %%ymm4, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm4, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm4\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm5, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm5, %%ymm5, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm5, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm5\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm6, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm6, %%ymm6, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm6, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm6\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm7, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm7, %%ymm7, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm7, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm7\n"
          "vaddps %%ymm1, %%ymm0, %%ymm8\n"
          "vsubps %%ymm1, %%ymm0, %%ymm9\n"
          "vaddps %%ymm3, %%ymm2, %%ymm10\n"
          "vsubps %%ymm3, %%ymm2, %%ymm11\n"
          "vaddps %%ymm5, %%ymm4, %%ymm12\n"
          "vsubps %%ymm5, %%ymm4, %%ymm13\n"
          "vaddps %%ymm7, %%ymm6, %%ymm14\n"
          "vsubps %%ymm7, %%ymm6, %%ymm15\n"
          "vaddps %%ymm10, %%ymm8, %%ymm0\n"
          "vsubps %%ymm10, %%ymm8, %%ymm2\n"
          "vaddps %%ymm11, %%ymm9, %%ymm1\n"
          "vsubps %%ymm11, %%ymm9, %%ymm3\n"
          "vaddps %%ymm14, %%ymm12, %%ymm4\n"
          "vsubps %%ymm14, %%ymm12, %%ymm6\n"
          "vaddps %%ymm15, %%ymm13, %%ymm5\n"
          "vsubps %%ymm15, %%ymm13, %%ymm7\n"
          "vaddps %%ymm4, %%ymm0, %%ymm8\n"
          "vsubps %%ymm4, %%ymm0, %%ymm12\n"
          "vaddps %%ymm5, %%ymm1, %%ymm9\n"
          "vsubps %%ymm5, %%ymm1, %%ymm13\n"
          "vaddps %%ymm6, %%ymm2, %%ymm10\n"
          "vsubps %%ymm6, %%ymm2, %%ymm14\n"
          "vaddps %%ymm7, %%ymm3, %%ymm11\n"
          "vsubps %%ymm7, %%ymm3, %%ymm15\n"
          "vmovups %%ymm8, (%0)\n"
          "vmovups %%ymm9, (%1)\n"
          "vmovups %%ymm10, (%2)\n"
          "vmovups %%ymm11, (%3)\n"
          "vmovups %%ymm12, (%4)\n"
          "vmovups %%ymm13, (%5)\n"
          "vmovups %%ymm14, (%6)\n"
          "vmovups %%ymm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 8),
          "r"(buf + j + k + 16),
          "r"(buf + j + k + 24),
          "r"(buf + j + k + 32),
          "r"(buf + j + k + 40),
          "r"(buf + j + k + 48),
          "r"(buf + j + k + 56)
          : "%ymm0",
            "%ymm1",
            "%ymm2",
            "%ymm3",
            "%ymm4",
            "%ymm5",
            "%ymm6",
            "%ymm7",
            "%ymm8",
            "%ymm9",
            "%ymm10",
            "%ymm11",
            "%ymm12",
            "%ymm13",
            "%ymm14",
            "%ymm15",
            "memory");
    }
  }
  for (int j = 0; j < 1024; j += 512) {
    for (int k = 0; k < 64; k += 8) {
      __asm__ volatile(
          "vmovups (%0), %%ymm0\n"
          "vmovups (%1), %%ymm1\n"
          "vmovups (%2), %%ymm2\n"
          "vmovups (%3), %%ymm3\n"
          "vmovups (%4), %%ymm4\n"
          "vmovups (%5), %%ymm5\n"
          "vmovups (%6), %%ymm6\n"
          "vmovups (%7), %%ymm7\n"
          "vaddps %%ymm1, %%ymm0, %%ymm8\n"
          "vsubps %%ymm1, %%ymm0, %%ymm9\n"
          "vaddps %%ymm3, %%ymm2, %%ymm10\n"
          "vsubps %%ymm3, %%ymm2, %%ymm11\n"
          "vaddps %%ymm5, %%ymm4, %%ymm12\n"
          "vsubps %%ymm5, %%ymm4, %%ymm13\n"
          "vaddps %%ymm7, %%ymm6, %%ymm14\n"
          "vsubps %%ymm7, %%ymm6, %%ymm15\n"
          "vaddps %%ymm10, %%ymm8, %%ymm0\n"
          "vsubps %%ymm10, %%ymm8, %%ymm2\n"
          "vaddps %%ymm11, %%ymm9, %%ymm1\n"
          "vsubps %%ymm11, %%ymm9, %%ymm3\n"
          "vaddps %%ymm14, %%ymm12, %%ymm4\n"
          "vsubps %%ymm14, %%ymm12, %%ymm6\n"
          "vaddps %%ymm15, %%ymm13, %%ymm5\n"
          "vsubps %%ymm15, %%ymm13, %%ymm7\n"
          "vaddps %%ymm4, %%ymm0, %%ymm8\n"
          "vsubps %%ymm4, %%ymm0, %%ymm12\n"
          "vaddps %%ymm5, %%ymm1, %%ymm9\n"
          "vsubps %%ymm5, %%ymm1, %%ymm13\n"
          "vaddps %%ymm6, %%ymm2, %%ymm10\n"
          "vsubps %%ymm6, %%ymm2, %%ymm14\n"
          "vaddps %%ymm7, %%ymm3, %%ymm11\n"
          "vsubps %%ymm7, %%ymm3, %%ymm15\n"
          "vmovups %%ymm8, (%0)\n"
          "vmovups %%ymm9, (%1)\n"
          "vmovups %%ymm10, (%2)\n"
          "vmovups %%ymm11, (%3)\n"
          "vmovups %%ymm12, (%4)\n"
          "vmovups %%ymm13, (%5)\n"
          "vmovups %%ymm14, (%6)\n"
          "vmovups %%ymm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 64),
          "r"(buf + j + k + 128),
          "r"(buf + j + k + 192),
          "r"(buf + j + k + 256),
          "r"(buf + j + k + 320),
          "r"(buf + j + k + 384),
          "r"(buf + j + k + 448)
          : "%ymm0",
            "%ymm1",
            "%ymm2",
            "%ymm3",
            "%ymm4",
            "%ymm5",
            "%ymm6",
            "%ymm7",
            "%ymm8",
            "%ymm9",
            "%ymm10",
            "%ymm11",
            "%ymm12",
            "%ymm13",
            "%ymm14",
            "%ymm15",
            "memory");
    }
  }
  for (int j = 0; j < 1024; j += 1024) {
    for (int k = 0; k < 512; k += 8) {
      __asm__ volatile(
          "vmovups (%0), %%ymm0\n"
          "vmovups (%1), %%ymm1\n"
          "vaddps %%ymm1, %%ymm0, %%ymm8\n"
          "vsubps %%ymm1, %%ymm0, %%ymm9\n"
          "vmovups %%ymm8, (%0)\n"
          "vmovups %%ymm9, (%1)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 512)
          : "%ymm0",
            "%ymm1",
            "%ymm2",
            "%ymm3",
            "%ymm4",
            "%ymm5",
            "%ymm6",
            "%ymm7",
            "%ymm8",
            "%ymm9",
            "%ymm10",
            "%ymm11",
            "%ymm12",
            "%ymm13",
            "%ymm14",
            "%ymm15",
            "memory");
    }
  }
}

ALAYA_NOINLINE
ALAYA_TARGET_AVX512
inline auto helper_float_11_avx512(float *buf) -> void {  // NOLINT
  for (int j = 0; j < 2048; j += 64) {
    for (int k = 0; k < 8; k += 8) {
      __asm__ volatile(
          "vmovups (%0), %%ymm0\n"
          "vmovups (%1), %%ymm1\n"
          "vmovups (%2), %%ymm2\n"
          "vmovups (%3), %%ymm3\n"
          "vmovups (%4), %%ymm4\n"
          "vmovups (%5), %%ymm5\n"
          "vmovups (%6), %%ymm6\n"
          "vmovups (%7), %%ymm7\n"
          "vpermilps $160, %%ymm0, %%ymm8\n"
          "vpermilps $245, %%ymm0, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm0\n"
          "vpermilps $160, %%ymm1, %%ymm8\n"
          "vpermilps $245, %%ymm1, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm1\n"
          "vpermilps $160, %%ymm2, %%ymm8\n"
          "vpermilps $245, %%ymm2, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm2\n"
          "vpermilps $160, %%ymm3, %%ymm8\n"
          "vpermilps $245, %%ymm3, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm3\n"
          "vpermilps $160, %%ymm4, %%ymm8\n"
          "vpermilps $245, %%ymm4, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm4\n"
          "vpermilps $160, %%ymm5, %%ymm8\n"
          "vpermilps $245, %%ymm5, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm5\n"
          "vpermilps $160, %%ymm6, %%ymm8\n"
          "vpermilps $245, %%ymm6, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm6\n"
          "vpermilps $160, %%ymm7, %%ymm8\n"
          "vpermilps $245, %%ymm7, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vaddsubps %%ymm11, %%ymm8, %%ymm7\n"
          "vpermilps $68, %%ymm0, %%ymm8\n"
          "vpermilps $238, %%ymm0, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm0\n"
          "vpermilps $68, %%ymm1, %%ymm8\n"
          "vpermilps $238, %%ymm1, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm1\n"
          "vpermilps $68, %%ymm2, %%ymm8\n"
          "vpermilps $238, %%ymm2, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm2\n"
          "vpermilps $68, %%ymm3, %%ymm8\n"
          "vpermilps $238, %%ymm3, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm3\n"
          "vpermilps $68, %%ymm4, %%ymm8\n"
          "vpermilps $238, %%ymm4, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm4\n"
          "vpermilps $68, %%ymm5, %%ymm8\n"
          "vpermilps $238, %%ymm5, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm5\n"
          "vpermilps $68, %%ymm6, %%ymm8\n"
          "vpermilps $238, %%ymm6, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm6\n"
          "vpermilps $68, %%ymm7, %%ymm8\n"
          "vpermilps $238, %%ymm7, %%ymm9\n"
          "vxorps %%ymm10, %%ymm10, %%ymm10\n"
          "vsubps %%ymm9, %%ymm10, %%ymm11\n"
          "vblendps $204, %%ymm11, %%ymm9, %%ymm12\n"
          "vaddps %%ymm8, %%ymm12, %%ymm7\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm0, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm0, %%ymm0, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm0, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm0\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm1, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm1, %%ymm1, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm1, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm1\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm2, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm2, %%ymm2, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm2, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm2\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm3, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm3, %%ymm3, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm3, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm3\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm4, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm4, %%ymm4, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm4, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm4\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm5, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm5, %%ymm5, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm5, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm5\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm6, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm6, %%ymm6, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm6, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm6\n"
          "vxorps %%ymm8, %%ymm8, %%ymm8\n"
          "vsubps %%ymm7, %%ymm8, %%ymm9\n"
          "vperm2f128 $0, %%ymm7, %%ymm7, %%ymm10\n"
          "vperm2f128 $49, %%ymm9, %%ymm7, %%ymm11\n"
          "vaddps %%ymm10, %%ymm11, %%ymm7\n"
          "vaddps %%ymm1, %%ymm0, %%ymm8\n"
          "vsubps %%ymm1, %%ymm0, %%ymm9\n"
          "vaddps %%ymm3, %%ymm2, %%ymm10\n"
          "vsubps %%ymm3, %%ymm2, %%ymm11\n"
          "vaddps %%ymm5, %%ymm4, %%ymm12\n"
          "vsubps %%ymm5, %%ymm4, %%ymm13\n"
          "vaddps %%ymm7, %%ymm6, %%ymm14\n"
          "vsubps %%ymm7, %%ymm6, %%ymm15\n"
          "vaddps %%ymm10, %%ymm8, %%ymm0\n"
          "vsubps %%ymm10, %%ymm8, %%ymm2\n"
          "vaddps %%ymm11, %%ymm9, %%ymm1\n"
          "vsubps %%ymm11, %%ymm9, %%ymm3\n"
          "vaddps %%ymm14, %%ymm12, %%ymm4\n"
          "vsubps %%ymm14, %%ymm12, %%ymm6\n"
          "vaddps %%ymm15, %%ymm13, %%ymm5\n"
          "vsubps %%ymm15, %%ymm13, %%ymm7\n"
          "vaddps %%ymm4, %%ymm0, %%ymm8\n"
          "vsubps %%ymm4, %%ymm0, %%ymm12\n"
          "vaddps %%ymm5, %%ymm1, %%ymm9\n"
          "vsubps %%ymm5, %%ymm1, %%ymm13\n"
          "vaddps %%ymm6, %%ymm2, %%ymm10\n"
          "vsubps %%ymm6, %%ymm2, %%ymm14\n"
          "vaddps %%ymm7, %%ymm3, %%ymm11\n"
          "vsubps %%ymm7, %%ymm3, %%ymm15\n"
          "vmovups %%ymm8, (%0)\n"
          "vmovups %%ymm9, (%1)\n"
          "vmovups %%ymm10, (%2)\n"
          "vmovups %%ymm11, (%3)\n"
          "vmovups %%ymm12, (%4)\n"
          "vmovups %%ymm13, (%5)\n"
          "vmovups %%ymm14, (%6)\n"
          "vmovups %%ymm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 8),
          "r"(buf + j + k + 16),
          "r"(buf + j + k + 24),
          "r"(buf + j + k + 32),
          "r"(buf + j + k + 40),
          "r"(buf + j + k + 48),
          "r"(buf + j + k + 56)
          : "%ymm0",
            "%ymm1",
            "%ymm2",
            "%ymm3",
            "%ymm4",
            "%ymm5",
            "%ymm6",
            "%ymm7",
            "%ymm8",
            "%ymm9",
            "%ymm10",
            "%ymm11",
            "%ymm12",
            "%ymm13",
            "%ymm14",
            "%ymm15",
            "memory");
    }
  }
  for (int j = 0; j < 2048; j += 512) {
    for (int k = 0; k < 64; k += 8) {
      __asm__ volatile(
          "vmovups (%0), %%ymm0\n"
          "vmovups (%1), %%ymm1\n"
          "vmovups (%2), %%ymm2\n"
          "vmovups (%3), %%ymm3\n"
          "vmovups (%4), %%ymm4\n"
          "vmovups (%5), %%ymm5\n"
          "vmovups (%6), %%ymm6\n"
          "vmovups (%7), %%ymm7\n"
          "vaddps %%ymm1, %%ymm0, %%ymm8\n"
          "vsubps %%ymm1, %%ymm0, %%ymm9\n"
          "vaddps %%ymm3, %%ymm2, %%ymm10\n"
          "vsubps %%ymm3, %%ymm2, %%ymm11\n"
          "vaddps %%ymm5, %%ymm4, %%ymm12\n"
          "vsubps %%ymm5, %%ymm4, %%ymm13\n"
          "vaddps %%ymm7, %%ymm6, %%ymm14\n"
          "vsubps %%ymm7, %%ymm6, %%ymm15\n"
          "vaddps %%ymm10, %%ymm8, %%ymm0\n"
          "vsubps %%ymm10, %%ymm8, %%ymm2\n"
          "vaddps %%ymm11, %%ymm9, %%ymm1\n"
          "vsubps %%ymm11, %%ymm9, %%ymm3\n"
          "vaddps %%ymm14, %%ymm12, %%ymm4\n"
          "vsubps %%ymm14, %%ymm12, %%ymm6\n"
          "vaddps %%ymm15, %%ymm13, %%ymm5\n"
          "vsubps %%ymm15, %%ymm13, %%ymm7\n"
          "vaddps %%ymm4, %%ymm0, %%ymm8\n"
          "vsubps %%ymm4, %%ymm0, %%ymm12\n"
          "vaddps %%ymm5, %%ymm1, %%ymm9\n"
          "vsubps %%ymm5, %%ymm1, %%ymm13\n"
          "vaddps %%ymm6, %%ymm2, %%ymm10\n"
          "vsubps %%ymm6, %%ymm2, %%ymm14\n"
          "vaddps %%ymm7, %%ymm3, %%ymm11\n"
          "vsubps %%ymm7, %%ymm3, %%ymm15\n"
          "vmovups %%ymm8, (%0)\n"
          "vmovups %%ymm9, (%1)\n"
          "vmovups %%ymm10, (%2)\n"
          "vmovups %%ymm11, (%3)\n"
          "vmovups %%ymm12, (%4)\n"
          "vmovups %%ymm13, (%5)\n"
          "vmovups %%ymm14, (%6)\n"
          "vmovups %%ymm15, (%7)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 64),
          "r"(buf + j + k + 128),
          "r"(buf + j + k + 192),
          "r"(buf + j + k + 256),
          "r"(buf + j + k + 320),
          "r"(buf + j + k + 384),
          "r"(buf + j + k + 448)
          : "%ymm0",
            "%ymm1",
            "%ymm2",
            "%ymm3",
            "%ymm4",
            "%ymm5",
            "%ymm6",
            "%ymm7",
            "%ymm8",
            "%ymm9",
            "%ymm10",
            "%ymm11",
            "%ymm12",
            "%ymm13",
            "%ymm14",
            "%ymm15",
            "memory");
    }
  }
  for (int j = 0; j < 2048; j += 2048) {
    for (int k = 0; k < 512; k += 8) {
      __asm__ volatile(
          "vmovups (%0), %%ymm0\n"
          "vmovups (%1), %%ymm1\n"
          "vmovups (%2), %%ymm2\n"
          "vmovups (%3), %%ymm3\n"
          "vaddps %%ymm1, %%ymm0, %%ymm8\n"
          "vsubps %%ymm1, %%ymm0, %%ymm9\n"
          "vaddps %%ymm3, %%ymm2, %%ymm10\n"
          "vsubps %%ymm3, %%ymm2, %%ymm11\n"
          "vaddps %%ymm10, %%ymm8, %%ymm0\n"
          "vsubps %%ymm10, %%ymm8, %%ymm2\n"
          "vaddps %%ymm11, %%ymm9, %%ymm1\n"
          "vsubps %%ymm11, %%ymm9, %%ymm3\n"
          "vmovups %%ymm0, (%0)\n"
          "vmovups %%ymm1, (%1)\n"
          "vmovups %%ymm2, (%2)\n"
          "vmovups %%ymm3, (%3)\n" ::"r"(buf + j + k + 0),
          "r"(buf + j + k + 512),
          "r"(buf + j + k + 1024),
          "r"(buf + j + k + 1536)
          : "%ymm0",
            "%ymm1",
            "%ymm2",
            "%ymm3",
            "%ymm4",
            "%ymm5",
            "%ymm6",
            "%ymm7",
            "%ymm8",
            "%ymm9",
            "%ymm10",
            "%ymm11",
            "%ymm12",
            "%ymm13",
            "%ymm14",
            "%ymm15",
            "memory");
    }
  }
}
#endif  // ALAYA_X86

inline auto helper_float_6(float *buf) -> void {  // NOLINT
  static const FHT_Helper_Func kFunc = []() -> FHT_Helper_Func {
#ifdef ALAYA_X86
    const auto &f = get_cpu_features();
    if (f.avx512f_) {
      return helper_float_6_avx512;
    }
    if (f.avx2_) {
      return helper_float_6_avx2;
    }
#endif
    return fwht_generic_template<6>;
  }();
  kFunc(buf);
}
inline auto helper_float_7(float *buf) -> void {
  static const FHT_Helper_Func kFunc = []() -> FHT_Helper_Func {
#ifdef ALAYA_X86
    const auto &f = get_cpu_features();
    if (f.avx512f_) {
      return helper_float_7_avx512;
    }
    if (f.avx2_) {
      return helper_float_7_avx2;
    }
#endif
    return fwht_generic_template<7>;
  }();
  kFunc(buf);
}
inline auto helper_float_8(float *buf) -> void {
  static const FHT_Helper_Func kFunc = []() -> FHT_Helper_Func {
#ifdef ALAYA_X86
    const auto &f = get_cpu_features();
    if (f.avx512f_) {
      return helper_float_8_avx512;
    }
    if (f.avx2_) {
      return helper_float_8_avx2;
    }
#endif
    return fwht_generic_template<8>;
  }();
  kFunc(buf);
}

inline auto helper_float_9(float *buf) -> void {
  static const FHT_Helper_Func kFunc = []() -> FHT_Helper_Func {
#ifdef ALAYA_X86
    const auto &f = get_cpu_features();
    if (f.avx512f_) {
      return helper_float_9_avx512;
    }
    if (f.avx2_) {
      return helper_float_9_avx2;
    }
#endif
    return fwht_generic_template<9>;
  }();
  kFunc(buf);
}
inline auto helper_float_10(float *buf) -> void {
  static const FHT_Helper_Func kFunc = []() -> FHT_Helper_Func {
#ifdef ALAYA_X86
    const auto &f = get_cpu_features();
    if (f.avx512f_) {
      return helper_float_10_avx512;
    }
    if (f.avx2_) {
      return helper_float_10_avx2;
    }
#endif
    return fwht_generic_template<10>;
  }();
  kFunc(buf);
}
inline auto helper_float_11(float *buf) -> void {
  static const FHT_Helper_Func kFunc = []() -> FHT_Helper_Func {
#ifdef ALAYA_X86
    const auto &f = get_cpu_features();
    if (f.avx512f_) {
      return helper_float_11_avx512;
    }
    if (f.avx2_) {
      return helper_float_11_avx2;
    }
#endif
    return fwht_generic_template<11>;
  }();
  kFunc(buf);
}

inline auto fht_float(float *buf, int log_n) -> int {
  switch (log_n) {
    case 6:
      helper_float_6(buf);
      return 0; 
    case 7:
      helper_float_7(buf);
      return 0; 
    case 8:
      helper_float_8(buf);
      return 0; 
    case 9:
      helper_float_9(buf);
      return 0; 
    case 10:
      helper_float_10(buf);
      return 0; 
    case 11:
      helper_float_11(buf);
      return 0; 
    default:
      return 1;
  }
  return 0;
}

}  // namespace alaya::simd
