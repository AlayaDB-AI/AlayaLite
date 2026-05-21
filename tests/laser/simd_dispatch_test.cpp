// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "index/graph/laser/qg/qg_scanner.hpp"
#include "index/graph/laser/quantization/fastscan_impl.hpp"
#include "index/graph/laser/space/l2.hpp"
#include "index/graph/laser/utils/rotator.hpp"
#include "index/graph/laser/utils/scalar_quantize.hpp"
#include "simd/cpu_features.hpp"
#include "simd/laser_dispatch.hpp"

namespace alaya::laser {
namespace {

auto bits_equal(float lhs, float rhs) -> bool {
  return std::bit_cast<uint32_t>(lhs) == std::bit_cast<uint32_t>(rhs);
}

void expect_near_rel(float lhs, float rhs, float rel = 1.0e-5F) {
  const float scale = std::max({1.0F, std::abs(lhs), std::abs(rhs)});
  EXPECT_LE(std::abs(lhs - rhs), rel * scale) << lhs << " vs " << rhs;
}

auto has_avx512_bw() -> bool {
  const auto &features = alaya::simd::get_cpu_features();
  return features.avx512f_ && features.avx512bw_;
}

TEST(LaserSimdDispatchTest, FactoriesSelectHighestSupportedIsa) {
  const auto &features = alaya::simd::get_cpu_features();
  if (features.avx512f_ && features.avx512bw_) {
    EXPECT_STREQ(simd::get_laser_simd_name(), "avx512");
    EXPECT_EQ(simd::get_accumulate_func(), simd::detail::accumulate_impl_avx512);
    EXPECT_EQ(simd::get_appro_dist_func(), simd::detail::appro_dist_impl_avx512);
    EXPECT_EQ(simd::get_convert_func(), simd::detail::convert_accum_to_float_avx512);
    EXPECT_EQ(simd::get_rotate_loop_func(), simd::detail::rotate_loop_avx512);
    EXPECT_EQ(simd::get_data_range_func(), simd::detail::data_range_avx512);
    EXPECT_EQ(simd::get_l2_sqr_single_func(), simd::detail::l2_sqr_single_avx512);
    return;
  }
  if (features.avx2_ && features.fma_) {
    EXPECT_STREQ(simd::get_laser_simd_name(), "avx2");
    EXPECT_EQ(simd::get_accumulate_func(), simd::detail::accumulate_impl_avx2);
    EXPECT_EQ(simd::get_appro_dist_func(), simd::detail::appro_dist_impl_avx2);
    EXPECT_EQ(simd::get_convert_func(), simd::detail::convert_accum_to_float_avx2);
    EXPECT_EQ(simd::get_rotate_loop_func(), simd::detail::rotate_loop_avx2);
    EXPECT_EQ(simd::get_data_range_func(), simd::detail::data_range_avx2);
    EXPECT_EQ(simd::get_l2_sqr_single_func(), simd::detail::l2_sqr_single_avx2);
    return;
  }
  EXPECT_THROW((void)simd::get_laser_simd_name(), std::runtime_error);
  EXPECT_THROW((void)simd::get_accumulate_func(), std::runtime_error);
}

TEST(LaserSimdDispatchTest, AccumulateAvx512MatchesAvx2) {
  if (!has_avx512_bw()) {
    GTEST_SKIP() << "AVX-512F+BW is not available on this CPU";
  }

  constexpr size_t kDim = 16;
  constexpr size_t kCodeLength = kDim << 2;
  std::array<uint8_t, kCodeLength> codes{};
  std::array<uint8_t, kCodeLength> lut{};
  for (size_t i = 0; i < kCodeLength; ++i) {
    codes[i] = static_cast<uint8_t>((i * 17U + 3U) & 0xFFU);
    lut[i] = static_cast<uint8_t>((i * 11U + 5U) & 0xFFU);
  }

  std::array<uint16_t, kBatchSize> avx512{};
  std::array<uint16_t, kBatchSize> avx2{};
  simd::detail::accumulate_impl_avx512(kDim, codes.data(), lut.data(), avx512.data());
  simd::detail::accumulate_impl_avx2(kDim, codes.data(), lut.data(), avx2.data());

  EXPECT_EQ(avx512, avx2);
}

TEST(LaserSimdDispatchTest, ApproDistAndConvertAvx512MatchAvx2) {
  if (!has_avx512_bw()) {
    GTEST_SKIP() << "AVX-512F+BW is not available on this CPU";
  }

  constexpr size_t kN = 32;
  std::array<uint16_t, kN> accum{};
  std::array<float, kN> converted512{};
  std::array<float, kN> converted2{};
  for (size_t i = 0; i < kN; ++i) {
    accum[i] = static_cast<uint16_t>(1000U + i * 13U);
  }
  simd::detail::convert_accum_to_float_avx512(kN, accum.data(), 37, converted512.data());
  simd::detail::convert_accum_to_float_avx2(kN, accum.data(), 37, converted2.data());
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_TRUE(bits_equal(converted512[i], converted2[i])) << i;
  }

  std::array<float, kN> triple_x{};
  std::array<float, kN> fac_dq{};
  std::array<float, kN> fac_vq{};
  std::array<float, kN> dist512{};
  std::array<float, kN> dist2{};
  for (size_t i = 0; i < kN; ++i) {
    triple_x[i] = 0.25F + static_cast<float>(i) * 0.5F;
    fac_dq[i] = 0.01F * static_cast<float>((i % 7) + 1);
    fac_vq[i] = 0.02F * static_cast<float>((i % 5) + 1);
  }
  simd::detail::appro_dist_impl_avx512(kN,
                                       1.25F,
                                       0.125F,
                                       2.5F,
                                       0.75F,
                                       converted512.data(),
                                       triple_x.data(),
                                       fac_dq.data(),
                                       fac_vq.data(),
                                       dist512.data());
  simd::detail::appro_dist_impl_avx2(kN,
                                     1.25F,
                                     0.125F,
                                     2.5F,
                                     0.75F,
                                     converted2.data(),
                                     triple_x.data(),
                                     fac_dq.data(),
                                     fac_vq.data(),
                                     dist2.data());
  for (size_t i = 0; i < kN; ++i) {
    expect_near_rel(dist512[i], dist2[i]);
  }
}

TEST(LaserSimdDispatchTest, DataRangeAndL2SingleAvx512MatchAvx2) {
  if (!has_avx512_bw()) {
    GTEST_SKIP() << "AVX-512F+BW is not available on this CPU";
  }

  constexpr size_t kN = 33;
  std::array<float, kN> values{};
  for (size_t i = 0; i < kN; ++i) {
    values[i] = (static_cast<float>(i % 11) - 5.0F) * 0.25F;
  }

  float lo512 = 0.0F;
  float hi512 = 0.0F;
  float lo2 = 0.0F;
  float hi2 = 0.0F;
  simd::detail::data_range_avx512(values.data(), values.size(), lo512, hi512);
  simd::detail::data_range_avx2(values.data(), values.size(), lo2, hi2);
  EXPECT_TRUE(bits_equal(lo512, lo2));
  EXPECT_TRUE(bits_equal(hi512, hi2));

  const float norm512 = simd::detail::l2_sqr_single_avx512(values.data(), values.size());
  const float norm2 = simd::detail::l2_sqr_single_avx2(values.data(), values.size());
  expect_near_rel(norm512, norm2);
}

TEST(LaserSimdDispatchTest, RotateLoopAvx512MatchesAvx2) {
  if (!has_avx512_bw()) {
    GTEST_SKIP() << "AVX-512F+BW is not available on this CPU";
  }

  constexpr size_t kN = 32;
  std::array<float, kN> src{};
  std::array<float, kN> signs{};
  std::array<float, kN> dst512{};
  std::array<float, kN> dst2{};
  for (size_t i = 0; i < kN; ++i) {
    src[i] = static_cast<float>(i + 1) * 0.125F;
    signs[i] = (i % 2 == 0) ? 0.25F : -0.25F;
  }

  const size_t done512 =
      simd::detail::rotate_loop_avx512(src.data(), signs.data(), kN, dst512.data());
  const size_t done2 = simd::detail::rotate_loop_avx2(src.data(), signs.data(), kN, dst2.data());
  ASSERT_EQ(done512, kN);
  ASSERT_EQ(done2, kN);
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_TRUE(bits_equal(dst512[i], dst2[i])) << i;
  }
}

}  // namespace
}  // namespace alaya::laser
