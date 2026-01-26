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

#include <cstdint>
#include "platform.hpp"

namespace alaya::simd {

// ============================================================================
// CPU Feature Detection
// ============================================================================
struct CpuFeatures {
  bool avx512f_ = false;
  bool avx2_ = false;
  bool fma_ = false;
  bool sse4_1_ = false;

  static auto detect() -> CpuFeatures {
    CpuFeatures features;

#ifdef ALAYA_X86
  #if defined(__GNUC__) || defined(__clang__)
    if (__builtin_cpu_supports("avx512f")) {
      features.avx512f_ = true;
    }
    if (__builtin_cpu_supports("avx2")) {
      features.avx2_ = true;
    }
    if (__builtin_cpu_supports("fma")) {
      features.fma_ = true;
    }
    if (__builtin_cpu_supports("sse4.1")) {
      features.sse4_1_ = true;
    }
  #elif defined(_MSC_VER)
    int cpu_info[4];
    __cpuid(cpu_info, 0);
    int max_func = cpu_info[0];

    if (max_func >= 1) {
      __cpuid(cpu_info, 1);
      features.sse4_1_ = (cpu_info[2] & (1 << 19)) != 0;
      features.fma_ = (cpu_info[2] & (1 << 12)) != 0;
    }
    if (max_func >= 7) {
      __cpuidex(cpu_info, 7, 0);
      features.avx512f_ = (cpu_info[1] & (1 << 16)) != 0;
      features.avx2_ = (cpu_info[1] & (1 << 5)) != 0;
    }
  #endif
#endif

    return features;
  }
};

inline auto get_cpu_features() -> const CpuFeatures & {
  static const CpuFeatures kFeatures = CpuFeatures::detect();
  return kFeatures;
}

// ============================================================================
// SIMD Level Enum
// ============================================================================
enum class SimdLevel : std::uint8_t { kGeneric, kSse4, kAvx2, kAvx512 };

inline auto get_simd_level() -> SimdLevel {
#ifdef ALAYA_X86
  const auto &f = get_cpu_features();
  if (f.avx512f_) {
    return SimdLevel::kAvx512;
  }
  if (f.avx2_ && f.fma_) {
    return SimdLevel::kAvx2;
  }
  if (f.sse4_1_) {
    return SimdLevel::kSse4;
  }
#endif
  return SimdLevel::kGeneric;
}

inline auto get_simd_level_name() -> const char * {
  switch (get_simd_level()) {
    case SimdLevel::kAvx512:
      return "AVX-512";
    case SimdLevel::kAvx2:
      return "AVX2+FMA";
    case SimdLevel::kSse4:
      return "SSE4.1";
    default:
      return "Generic";
  }
}

}  // namespace alaya::simd
