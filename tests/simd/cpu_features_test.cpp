// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "simd/cpu_features.hpp"
#include <gtest/gtest.h>

namespace alaya::simd {

TEST(CpuFeaturesTest, GetSimdLevelPrefersHighestAvailableCapability) {
  CpuFeatures generic;
  EXPECT_EQ(get_simd_level(generic), SimdLevel::kGeneric);

  CpuFeatures sse4;
  sse4.sse4_1_ = true;
  EXPECT_EQ(get_simd_level(sse4), SimdLevel::kSse4);

  CpuFeatures avx2;
  avx2.avx2_ = true;
  avx2.fma_ = true;
  avx2.sse4_1_ = true;
  EXPECT_EQ(get_simd_level(avx2), SimdLevel::kAvx2);

  CpuFeatures avx512;
  avx512.avx512f_ = true;
  avx512.avx2_ = true;
  avx512.fma_ = true;
  avx512.sse4_1_ = true;
  EXPECT_EQ(get_simd_level(avx512), SimdLevel::kAvx512);
}

TEST(CpuFeaturesTest, GetSimdLevelNameMatchesEachEnum) {
  EXPECT_STREQ(get_simd_level_name(SimdLevel::kGeneric), "Generic");
  EXPECT_STREQ(get_simd_level_name(SimdLevel::kSse4), "SSE4.1");
  EXPECT_STREQ(get_simd_level_name(SimdLevel::kAvx2), "AVX2+FMA");
  EXPECT_STREQ(get_simd_level_name(SimdLevel::kAvx512), "AVX-512");
}

TEST(CpuFeaturesTest, RuntimeHelpersStayConsistent) {
  const auto &features = get_cpu_features();
  const auto level = get_simd_level();

  EXPECT_STREQ(get_simd_level_name(level), get_simd_level_name());

#ifdef ALAYA_ARCH_X86
  if (features.avx512f_) {
    EXPECT_EQ(level, SimdLevel::kAvx512);
  } else if (features.avx2_ && features.fma_) {
    EXPECT_EQ(level, SimdLevel::kAvx2);
  } else if (features.sse4_1_) {
    EXPECT_EQ(level, SimdLevel::kSse4);
  } else {
    EXPECT_EQ(level, SimdLevel::kGeneric);
  }
#else
  EXPECT_EQ(level, SimdLevel::kGeneric);
#endif
}

TEST(CpuFeaturesTest, Fp32DistanceDispatchDefaultsToAvx2WhenAvailable) {
  CpuFeatures avx512;
  avx512.avx512f_ = true;
  avx512.avx2_ = true;
  avx512.fma_ = true;
  avx512.sse4_1_ = true;

  EXPECT_EQ(select_fp32_distance_level(avx512, DistanceDispatchPolicy::kPreferStableThroughput),
            SimdLevel::kAvx2);
  EXPECT_EQ(select_fp32_distance_level(avx512, DistanceDispatchPolicy::kPreferAvx512),
            SimdLevel::kAvx512);
}

TEST(CpuFeaturesTest, ParseDistanceDispatchPolicyRecognizesAvx512Override) {
  EXPECT_EQ(parse_distance_dispatch_policy(nullptr),
            DistanceDispatchPolicy::kPreferStableThroughput);
  EXPECT_EQ(parse_distance_dispatch_policy("avx512"), DistanceDispatchPolicy::kPreferAvx512);
  EXPECT_EQ(parse_distance_dispatch_policy("unknown"),
            DistanceDispatchPolicy::kPreferStableThroughput);
}

}  // namespace alaya::simd
