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

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include "simd/fht.hpp"

// ============================================================================
// FHT (Fast Hadamard Transform) Tests
// ============================================================================

class FHTTest : public ::testing::Test {
 protected:
  static void fill_random(std::vector<float>& v, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto& x : v) {
      x = dist(rng);
    }
  }

  // Reference implementation: naive O(n^2) Hadamard transform
  static void reference_fwht(std::vector<float>& a) {
    size_t n = a.size();
    for (size_t h = 1; h < n; h <<= 1) {
      for (size_t i = 0; i < n; i += (h << 1)) {
        for (size_t j = i; j < i + h; ++j) {
          float x = a[j];
          float y = a[j + h];
          a[j] = x + y;
          a[j + h] = x - y;
        }
      }
    }
  }

  static auto vectors_equal(const std::vector<float>& a, const std::vector<float>& b,
                            float tolerance = 1e-4F) -> bool {
    if (a.size() != b.size()) {
      return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
      if (std::abs(a[i] - b[i]) > tolerance) {
        return false;
      }
    }
    return true;
  }
};

TEST_F(FHTTest, SimdLevelDetection) {
  const char* level = alaya::simd::get_simd_level_name();
  std::cout << "Detected SIMD level: " << level << '\n';
  EXPECT_NE(level, nullptr);
}

TEST_F(FHTTest, GenericCorrectness) {
  constexpr size_t kLogN = 8;
  constexpr size_t kN = 1ULL << kLogN;  // 256

  std::vector<float> input(kN);
  fill_random(input, 42);

  std::vector<float> expected = input;
  reference_fwht(expected);

  std::vector<float> result = input;
  alaya::simd::fwht_generic_template<kLogN>(result.data());

  EXPECT_TRUE(vectors_equal(result, expected, 1e-4F));
}

TEST_F(FHTTest, Helper6Correctness) {
  const size_t kN = 64;  // 2^6

  std::vector<float> input(kN);
  fill_random(input, 1);

  std::vector<float> expected = input;
  reference_fwht(expected);

  std::vector<float> result = input;
  alaya::simd::helper_float_6(result.data());

  EXPECT_TRUE(vectors_equal(result, expected, 1e-4F));
}

TEST_F(FHTTest, Helper7Correctness) {
  const size_t kN = 128;  // 2^7

  std::vector<float> input(kN);
  fill_random(input, 2);

  std::vector<float> expected = input;
  reference_fwht(expected);

  std::vector<float> result = input;
  alaya::simd::helper_float_7(result.data());

  EXPECT_TRUE(vectors_equal(result, expected, 1e-4F));
}

TEST_F(FHTTest, Helper8Correctness) {
  const size_t kN = 256;  // 2^8

  std::vector<float> input(kN);
  fill_random(input, 3);

  std::vector<float> expected = input;
  reference_fwht(expected);

  std::vector<float> result = input;
  alaya::simd::helper_float_8(result.data());

  EXPECT_TRUE(vectors_equal(result, expected, 1e-4F));
}

TEST_F(FHTTest, Helper9Correctness) {
  const size_t kN = 512;  // 2^9

  std::vector<float> input(kN);
  fill_random(input, 4);

  std::vector<float> expected = input;
  reference_fwht(expected);

  std::vector<float> result = input;
  alaya::simd::helper_float_9(result.data());

  EXPECT_TRUE(vectors_equal(result, expected, 1e-4F));
}

TEST_F(FHTTest, Helper10Correctness) {
  const size_t kN = 1024;  // 2^10

  std::vector<float> input(kN);
  fill_random(input, 5);

  std::vector<float> expected = input;
  reference_fwht(expected);

  std::vector<float> result = input;
  alaya::simd::helper_float_10(result.data());

  EXPECT_TRUE(vectors_equal(result, expected, 1e-4F));
}

TEST_F(FHTTest, Helper11Correctness) {
  const size_t kN = 2048;  // 2^11

  std::vector<float> input(kN);
  fill_random(input, 6);

  std::vector<float> expected = input;
  reference_fwht(expected);

  std::vector<float> result = input;
  alaya::simd::helper_float_11(result.data());

  EXPECT_TRUE(vectors_equal(result, expected, 1e-4F));
}

TEST_F(FHTTest, ZeroVector) {
  const size_t kN = 256;
  std::vector<float> input(kN, 0.0F);

  alaya::simd::helper_float_8(input.data());

  for (size_t i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(input[i], 0.0F);
  }
}

TEST_F(FHTTest, UnitImpulse) {
  const size_t kN = 64;
  std::vector<float> input(kN, 0.0F);
  input[0] = 1.0F;

  std::vector<float> expected = input;
  reference_fwht(expected);

  alaya::simd::helper_float_6(input.data());

  EXPECT_TRUE(vectors_equal(input, expected, 1e-6F));

  // For unit impulse at index 0, all outputs should be 1.0
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_FLOAT_EQ(input[i], 1.0F);
  }
}

TEST_F(FHTTest, Involution) {
  // FHT is its own inverse (up to scaling by n)
  const size_t kN = 128;
  std::vector<float> original(kN);
  fill_random(original, 123);

  std::vector<float> transformed = original;
  alaya::simd::helper_float_7(transformed.data());
  alaya::simd::helper_float_7(transformed.data());

  // After two transforms, result should be n * original
  for (size_t i = 0; i < kN; ++i) {
    EXPECT_NEAR(transformed[i], original[i] * static_cast<float>(kN), 1e-3F);
  }
}

TEST_F(FHTTest, GenericTemplateVariousSizes) {
  // Test template version with various sizes
  auto test_size = [](auto log_n_tag) -> void {
    constexpr size_t kLogN = decltype(log_n_tag)::value;
    constexpr size_t kN = 1ULL << kLogN;

    std::vector<float> input(kN);
    std::mt19937 rng(kN);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto& x : input) {
      x = dist(rng);
    }

    std::vector<float> expected = input;
    size_t n = expected.size();
    for (size_t h = 1; h < n; h <<= 1) {
      for (size_t i = 0; i < n; i += (h << 1)) {
        for (size_t j = i; j < i + h; ++j) {
          float x = expected[j];
          float y = expected[j + h];
          expected[j] = x + y;
          expected[j + h] = x - y;
        }
      }
    }

    std::vector<float> result = input;
    alaya::simd::fwht_generic_template<kLogN>(result.data());

    for (size_t i = 0; i < kN; ++i) {
      EXPECT_NEAR(result[i], expected[i], 1e-4F) << "Failed for log_n=" << kLogN;
    }
  };

  test_size(std::integral_constant<size_t, 2>{});   // 4
  test_size(std::integral_constant<size_t, 3>{});   // 8
  test_size(std::integral_constant<size_t, 4>{});   // 16
  test_size(std::integral_constant<size_t, 5>{});   // 32
  test_size(std::integral_constant<size_t, 6>{});   // 64
  test_size(std::integral_constant<size_t, 7>{});   // 128
  test_size(std::integral_constant<size_t, 8>{});   // 256
  test_size(std::integral_constant<size_t, 9>{});   // 512
  test_size(std::integral_constant<size_t, 10>{});  // 1024
}

#ifdef ALAYA_ARCH_X86
TEST_F(FHTTest, AVX2Correctness) {
  const auto& features = alaya::simd::get_cpu_features();
  if (!features.avx2_) {
    GTEST_SKIP() << "AVX2 not available";
  }

  // Test all AVX2 helper functions
  std::vector<std::pair<size_t, alaya::simd::FHT_Helper_Func>> helpers = {
      {64, alaya::simd::helper_float_6_avx2},   {128, alaya::simd::helper_float_7_avx2},
      {256, alaya::simd::helper_float_8_avx2},  {512, alaya::simd::helper_float_9_avx2},
      {1024, alaya::simd::helper_float_10_avx2}, {2048, alaya::simd::helper_float_11_avx2},
  };

  for (const auto& [n, func] : helpers) {
    std::vector<float> input(n);
    fill_random(input, n);

    std::vector<float> expected = input;
    reference_fwht(expected);

    std::vector<float> result = input;
    func(result.data());

    EXPECT_TRUE(vectors_equal(result, expected, 1e-4F))
        << "AVX2 failed for size=" << n;
  }
}

TEST_F(FHTTest, AVX512Correctness) {
  const auto& features = alaya::simd::get_cpu_features();
  if (!features.avx512f_) {
    GTEST_SKIP() << "AVX-512 not available";
  }

  // Test all AVX512 helper functions
  std::vector<std::pair<size_t, alaya::simd::FHT_Helper_Func>> helpers = {
      {64, alaya::simd::helper_float_6_avx512},   {128, alaya::simd::helper_float_7_avx512},
      {256, alaya::simd::helper_float_8_avx512},  {512, alaya::simd::helper_float_9_avx512},
      {1024, alaya::simd::helper_float_10_avx512}, {2048, alaya::simd::helper_float_11_avx512},
  };

  for (const auto& [n, func] : helpers) {
    std::vector<float> input(n);
    fill_random(input, n);

    std::vector<float> expected = input;
    reference_fwht(expected);

    std::vector<float> result = input;
    func(result.data());

    EXPECT_TRUE(vectors_equal(result, expected, 1e-4F))
        << "AVX512 failed for size=" << n;
  }
}
#endif

// ============================================================================
// fht_float unified API tests
// ============================================================================

TEST_F(FHTTest, FhtFloatCorrectness) {
  // Test fht_float for all supported sizes (log_n = 6 to 11)
  for (int log_n = 6; log_n <= 11; ++log_n) {
    const size_t kN = 1ULL << log_n;

    std::vector<float> input(kN);
    fill_random(input, log_n);

    std::vector<float> expected = input;
    reference_fwht(expected);

    std::vector<float> result = input;
    int ret = alaya::simd::fht_float(result.data(), log_n);

    EXPECT_EQ(ret, 0) << "fht_float returned error for log_n=" << log_n;
    EXPECT_TRUE(vectors_equal(result, expected, 1e-4F))
        << "fht_float failed for log_n=" << log_n << " (size=" << kN << ")";
  }
}

TEST_F(FHTTest, FhtFloatInvalidSize) {
  // Test fht_float with unsupported sizes
  std::vector<float> buf(32);  // 2^5, not supported
  fill_random(buf, 99);

  int ret = alaya::simd::fht_float(buf.data(), 5);
  EXPECT_NE(ret, 0) << "fht_float should return error for unsupported log_n=5";

  ret = alaya::simd::fht_float(buf.data(), 12);
  EXPECT_NE(ret, 0) << "fht_float should return error for unsupported log_n=12";
}

TEST_F(FHTTest, FhtFloatInvolution) {
  // Test that applying fht_float twice gives n * original
  for (int log_n = 6; log_n <= 11; ++log_n) {
    const size_t kN = 1ULL << log_n;

    std::vector<float> original(kN);
    fill_random(original, log_n + 100);

    std::vector<float> transformed = original;
    alaya::simd::fht_float(transformed.data(), log_n);
    alaya::simd::fht_float(transformed.data(), log_n);

    for (size_t i = 0; i < kN; ++i) {
      EXPECT_NEAR(transformed[i], original[i] * static_cast<float>(kN), 1e-2F)
          << "Involution failed for log_n=" << log_n << " at index=" << i;
    }
  }
}
