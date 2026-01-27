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
#include "simd/distance_l2.hpp"

// ============================================================================
// L2 Float Tests
// ============================================================================

class L2SqrTest : public ::testing::Test {
 protected:
  static auto alloc_float(size_t n) -> std::vector<float> { return std::vector<float>(n); }

  static void fill_random(std::vector<float>& v, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto& x : v) {
      x = dist(rng);
    }
  }
};

TEST_F(L2SqrTest, SimdLevelDetection) {
  const char* level = alaya::simd::get_simd_level_name();
  std::cout << "Detected SIMD level: " << level << '\n';
  EXPECT_NE(level, nullptr);
}

TEST_F(L2SqrTest, GenericCorrectness) {
  const size_t kDim = 128;
  auto x = alloc_float(kDim);
  auto y = alloc_float(kDim);
  fill_random(x, 1);
  fill_random(y, 2);

  float expected = 0.0F;
  for (size_t i = 0; i < kDim; ++i) {
    float diff = x[i] - y[i];
    expected += diff * diff;
  }

  auto result = alaya::simd::l2_sqr_generic(x.data(), y.data(), kDim);
  EXPECT_NEAR(result, expected, 1e-5F);
}

TEST_F(L2SqrTest, SimdCorrectness) {
  const size_t kDim = 128;
  auto x = alloc_float(kDim);
  auto y = alloc_float(kDim);
  fill_random(x, 1);
  fill_random(y, 2);

  float expected = alaya::simd::l2_sqr_generic(x.data(), y.data(), kDim);
  float result = alaya::simd::l2_sqr(x.data(), y.data(), kDim);

  EXPECT_NEAR(result, expected, 1e-4F);
}

TEST_F(L2SqrTest, TailHandling) {
  std::vector<size_t> dims = {1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129};

  for (size_t dim : dims) {
    auto x = alloc_float(dim);
    auto y = alloc_float(dim);
    fill_random(x, dim);
    fill_random(y, dim + 100);

    float expected = alaya::simd::l2_sqr_generic(x.data(), y.data(), dim);
    float result = alaya::simd::l2_sqr(x.data(), y.data(), dim);

    EXPECT_NEAR(result, expected, 1e-4F) << "Failed for dim=" << dim;
  }
}

TEST_F(L2SqrTest, ZeroVector) {
  const size_t kDim = 64;
  std::vector<float> x(kDim, 0.0F);
  std::vector<float> y(kDim, 0.0F);

  float result = alaya::simd::l2_sqr(x.data(), y.data(), kDim);
  EXPECT_FLOAT_EQ(result, 0.0F);
}

TEST_F(L2SqrTest, IdenticalVectors) {
  const size_t kDim = 128;
  auto x = alloc_float(kDim);
  fill_random(x, 42);

  float result = alaya::simd::l2_sqr(x.data(), x.data(), kDim);
  EXPECT_FLOAT_EQ(result, 0.0F);
}

TEST_F(L2SqrTest, LargeDimension) {
  const size_t kDim = 1024;
  auto x = alloc_float(kDim);
  auto y = alloc_float(kDim);
  fill_random(x, 1);
  fill_random(y, 2);

  float expected = alaya::simd::l2_sqr_generic(x.data(), y.data(), kDim);
  float result = alaya::simd::l2_sqr(x.data(), y.data(), kDim);

  EXPECT_NEAR(result, expected, 1e-3F);
}

TEST_F(L2SqrTest, TemplateWrapper) {
  const size_t kDim = 64;
  auto x = alloc_float(kDim);
  auto y = alloc_float(kDim);
  fill_random(x, 1);
  fill_random(y, 2);

  float expected = alaya::simd::l2_sqr_generic(x.data(), y.data(), kDim);
  auto result = alaya::simd::l2_sqr<float, float>(x.data(), y.data(), kDim);

  EXPECT_NEAR(result, expected, 1e-4F);
}

// ============================================================================
// L2 SQ8 Tests
// ============================================================================

class L2SqrSQ8Test : public ::testing::Test {
 protected:
  static void fill_random_uint8(std::vector<uint8_t>& v, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& x : v) {
      x = static_cast<uint8_t>(dist(rng));
    }
  }

  // Generate min/max arrays where max > min
  static void fill_min_max(std::vector<float>& min_vals,
                           std::vector<float>& max_vals,
                           size_t dim,
                           unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-10.0F, 10.0F);
    min_vals.resize(dim);
    max_vals.resize(dim);
    for (size_t i = 0; i < dim; ++i) {
      float a = dist(rng);
      float b = dist(rng);
      min_vals[i] = std::min(a, b);
      max_vals[i] = std::max(a, b) + 0.1F;  // Ensure max > min
    }
  }

  // Reference implementation for validation
  static auto reference_l2_sqr_sq8(const uint8_t* x,
                                   const uint8_t* y,
                                   size_t dim,
                                   const float* min,
                                   const float* max) -> float {
    constexpr float kInv255 = 1.0F / 255.0F;
    float sum = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
      float scale = (max[i] - min[i]) * kInv255;
      float diff = static_cast<float>(x[i]) - static_cast<float>(y[i]);
      float scaled_diff = diff * scale;
      sum += scaled_diff * scaled_diff;
    }
    return sum;
  }
};

TEST_F(L2SqrSQ8Test, GenericCorrectness) {
  const size_t kDim = 128;
  std::vector<uint8_t> x(kDim);
  std::vector<uint8_t> y(kDim);
  std::vector<float> min_vals;
  std::vector<float> max_vals;

  fill_random_uint8(x, 1);
  fill_random_uint8(y, 2);
  fill_min_max(min_vals, max_vals, kDim, 3);

  float expected =
      reference_l2_sqr_sq8(x.data(), y.data(), kDim, min_vals.data(), max_vals.data());
  float result = alaya::simd::l2_sqr_sq8_generic(x.data(), y.data(), kDim, min_vals.data(),
                                                  max_vals.data());

  EXPECT_NEAR(result, expected, 1e-4F);
}

TEST_F(L2SqrSQ8Test, SimdCorrectness) {
  const size_t kDim = 128;
  std::vector<uint8_t> x(kDim);
  std::vector<uint8_t> y(kDim);
  std::vector<float> min_vals;
  std::vector<float> max_vals;

  fill_random_uint8(x, 1);
  fill_random_uint8(y, 2);
  fill_min_max(min_vals, max_vals, kDim, 3);

  float expected = alaya::simd::l2_sqr_sq8_generic(x.data(), y.data(), kDim, min_vals.data(),
                                                    max_vals.data());
  auto result = alaya::simd::l2_sqr_sq8<float, float>(x.data(), y.data(), kDim, min_vals.data(),
                                                        max_vals.data());

  EXPECT_NEAR(result, expected, 1e-3F);
}

TEST_F(L2SqrSQ8Test, TailHandling) {
  std::vector<size_t> dims = {1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129};

  for (size_t dim : dims) {
    std::vector<uint8_t> x(dim);
    std::vector<uint8_t> y(dim);
    std::vector<float> min_vals;
    std::vector<float> max_vals;

    fill_random_uint8(x, dim);
    fill_random_uint8(y, dim + 100);
    fill_min_max(min_vals, max_vals, dim, dim + 200);

    float expected = alaya::simd::l2_sqr_sq8_generic(x.data(), y.data(), dim, min_vals.data(),
                                                      max_vals.data());
    auto result = alaya::simd::l2_sqr_sq8<float, float>(x.data(), y.data(), dim, min_vals.data(),
                                                                            max_vals.data());

    EXPECT_NEAR(result, expected, 1e-3F) << "Failed for dim=" << dim;
  }
}

TEST_F(L2SqrSQ8Test, ZeroVector) {
  const size_t kDim = 64;
  std::vector<uint8_t> x(kDim, 0);
  std::vector<uint8_t> y(kDim, 0);
  std::vector<float> min_vals;
  std::vector<float> max_vals;

  fill_min_max(min_vals, max_vals, kDim, 42);

  auto result = alaya::simd::l2_sqr_sq8<float, float>(x.data(), y.data(), kDim, min_vals.data(),
                                                        max_vals.data());
  EXPECT_FLOAT_EQ(result, 0.0F);
}

TEST_F(L2SqrSQ8Test, IdenticalVectors) {
  const size_t kDim = 128;
  std::vector<uint8_t> x(kDim);
  std::vector<float> min_vals;
  std::vector<float> max_vals;

  fill_random_uint8(x, 42);
  fill_min_max(min_vals, max_vals, kDim, 43);

  auto result = alaya::simd::l2_sqr_sq8<float, float>(x.data(), x.data(), kDim, min_vals.data(),
                                                        max_vals.data());
  EXPECT_FLOAT_EQ(result, 0.0F);
}

TEST_F(L2SqrSQ8Test, LargeDimension) {
  const size_t kDim = 1024;
  std::vector<uint8_t> x(kDim);
  std::vector<uint8_t> y(kDim);
  std::vector<float> min_vals;
  std::vector<float> max_vals;

  fill_random_uint8(x, 1);
  fill_random_uint8(y, 2);
  fill_min_max(min_vals, max_vals, kDim, 3);

  float expected = alaya::simd::l2_sqr_sq8_generic(x.data(), y.data(), kDim, min_vals.data(),
                                                    max_vals.data());
  auto result = alaya::simd::l2_sqr_sq8<float, float>(x.data(), y.data(), kDim, min_vals.data(),
                                                        max_vals.data());

  EXPECT_NEAR(result, expected, 1e-2F);
}

TEST_F(L2SqrSQ8Test, QuantizationExtremes) {
  const size_t kDim = 64;
  std::vector<uint8_t> x(kDim, 0);    // All minimum
  std::vector<uint8_t> y(kDim, 255);  // All maximum
  std::vector<float> min_vals(kDim, -1.0F);
  std::vector<float> max_vals(kDim, 1.0F);

  // Expected: each dimension contributes ((0 - 255) * (2.0 / 255))^2 = 4.0
  float expected = static_cast<float>(kDim) * 4.0F;
  auto result = alaya::simd::l2_sqr_sq8<float, float>(x.data(), y.data(), kDim, min_vals.data(),
                                                        max_vals.data());

  EXPECT_NEAR(result, expected, 1e-3F);
}

#ifdef ALAYA_X86
TEST_F(L2SqrSQ8Test, AVX2Correctness) {
  const auto& features = alaya::simd::get_cpu_features();
  if (!features.avx2_ || !features.fma_) {
    GTEST_SKIP() << "AVX2 + FMA not available";
  }

  const size_t kDim = 128;
  std::vector<uint8_t> x(kDim);
  std::vector<uint8_t> y(kDim);
  std::vector<float> min_vals;
  std::vector<float> max_vals;

  fill_random_uint8(x, 1);
  fill_random_uint8(y, 2);
  fill_min_max(min_vals, max_vals, kDim, 3);

  float expected = alaya::simd::l2_sqr_sq8_generic(x.data(), y.data(), kDim, min_vals.data(),
                                                    max_vals.data());
  auto result =
      alaya::simd::l2_sqr_sq8_avx2(x.data(), y.data(), kDim, min_vals.data(), max_vals.data());

  EXPECT_NEAR(result, expected, 1e-3F);
}

TEST_F(L2SqrSQ8Test, AVX512Correctness) {
  const auto& features = alaya::simd::get_cpu_features();
  if (!features.avx512f_) {
    GTEST_SKIP() << "AVX-512 not available";
  }

  const size_t kDim = 128;
  std::vector<uint8_t> x(kDim);
  std::vector<uint8_t> y(kDim);
  std::vector<float> min_vals;
  std::vector<float> max_vals;

  fill_random_uint8(x, 1);
  fill_random_uint8(y, 2);
  fill_min_max(min_vals, max_vals, kDim, 3);

  float expected = alaya::simd::l2_sqr_sq8_generic(x.data(), y.data(), kDim, min_vals.data(),
                                                    max_vals.data());
  auto result =
      alaya::simd::l2_sqr_sq8_avx512(x.data(), y.data(), kDim, min_vals.data(), max_vals.data());

  EXPECT_NEAR(result, expected, 1e-3F);
}
#endif

// ============================================================================
// L2 SQ4 Tests
// ============================================================================

class L2SqrSQ4Test : public ::testing::Test {
 protected:
  // Pack two 4-bit values into one byte: low nibble = even index, high nibble = odd index
  static void pack_sq4(std::vector<uint8_t>& packed, const std::vector<uint8_t>& values) {
    size_t dim = values.size();
    size_t num_bytes = (dim + 1) / 2;
    packed.resize(num_bytes);
    for (size_t i = 0; i < dim; i += 2) {
      uint8_t lo = values[i] & 0x0F;
      uint8_t hi = (i + 1 < dim) ? (values[i + 1] & 0x0F) : 0;
      packed[i / 2] = lo | (hi << 4);
    }
  }

  static void fill_random_sq4(std::vector<uint8_t>& v, size_t dim, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 15);
    v.resize(dim);
    for (auto& x : v) {
      x = static_cast<uint8_t>(dist(rng));
    }
  }

  static void fill_min_max(std::vector<float>& min_vals,
                           std::vector<float>& max_vals,
                           size_t dim,
                           unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-10.0F, 10.0F);
    min_vals.resize(dim);
    max_vals.resize(dim);
    for (size_t i = 0; i < dim; ++i) {
      float a = dist(rng);
      float b = dist(rng);
      min_vals[i] = std::min(a, b);
      max_vals[i] = std::max(a, b) + 0.1F;
    }
  }

  // Reference implementation for validation
  static auto reference_l2_sqr_sq4(const uint8_t* x,
                                   const uint8_t* y,
                                   size_t dim,
                                   const float* min,
                                   const float* max
                                   ) -> float {
    constexpr float kInv15 = 1.0F / 15.0F;
    float sum = 0.0F;
    size_t byte_idx = 0;
    for (size_t i = 0; i < dim; i += 2, ++byte_idx) {
      uint8_t x_lo = x[byte_idx] & 0x0F;
      uint8_t y_lo = y[byte_idx] & 0x0F;
      float scale_lo = (max[i] - min[i]) * kInv15;
      float diff_lo = static_cast<float>(x_lo) - static_cast<float>(y_lo);
      sum += diff_lo * scale_lo * diff_lo * scale_lo;

      if (i + 1 < dim) {
        uint8_t x_hi = (x[byte_idx] >> 4) & 0x0F;
        uint8_t y_hi = (y[byte_idx] >> 4) & 0x0F;
        float scale_hi = (max[i + 1] - min[i + 1]) * kInv15;
        float diff_hi = static_cast<float>(x_hi) - static_cast<float>(y_hi);
        sum += diff_hi * scale_hi * diff_hi * scale_hi;
      }
    }
    return sum;
  }
};

TEST_F(L2SqrSQ4Test, GenericCorrectness) {
  const size_t kDim = 128;
  std::vector<uint8_t> x_vals;
  std::vector<uint8_t> y_vals;
  std::vector<uint8_t> x_packed;
  std::vector<uint8_t> y_packed;
  std::vector<float> min_vals;
  std::vector<float> max_vals;

  fill_random_sq4(x_vals, kDim, 1);
  fill_random_sq4(y_vals, kDim, 2);
  pack_sq4(x_packed, x_vals);
  pack_sq4(y_packed, y_vals);
  fill_min_max(min_vals, max_vals, kDim, 3);

  float expected =
      reference_l2_sqr_sq4(x_packed.data(), y_packed.data(), kDim, min_vals.data(), max_vals.data());
  float result = alaya::simd::l2_sqr_sq4_generic(x_packed.data(), y_packed.data(), kDim, min_vals.data(),
                                                  max_vals.data());

  EXPECT_NEAR(result, expected, 1e-3F);
}

TEST_F(L2SqrSQ4Test, SimdCorrectness) {
  const size_t kDim = 128;
  std::vector<uint8_t> x_vals;
  std::vector<uint8_t> y_vals;
  std::vector<uint8_t> x_packed;
  std::vector<uint8_t> y_packed;
  std::vector<float> min_vals;
  std::vector<float> max_vals;

  fill_random_sq4(x_vals, kDim, 1);
  fill_random_sq4(y_vals, kDim, 2);
  pack_sq4(x_packed, x_vals);
  pack_sq4(y_packed, y_vals);
  fill_min_max(min_vals, max_vals, kDim, 3);

  float expected = alaya::simd::l2_sqr_sq4_generic(x_packed.data(), y_packed.data(), kDim,
                                                    min_vals.data(), max_vals.data());
  auto result = alaya::simd::l2_sqr_sq4<float, float>(x_packed.data(), y_packed.data(), kDim,
                                                        min_vals.data(), max_vals.data());

  EXPECT_NEAR(result, expected, 1e-3F);
}

TEST_F(L2SqrSQ4Test, TailHandling) {
  std::vector<size_t> dims = {1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129};

  for (size_t dim : dims) {
    std::vector<uint8_t> x_vals;
    std::vector<uint8_t> y_vals;
    std::vector<uint8_t> x_packed;
    std::vector<uint8_t> y_packed;
    std::vector<float> min_vals;
    std::vector<float> max_vals;

    fill_random_sq4(x_vals, dim, dim);
    fill_random_sq4(y_vals, dim, dim + 100);
    pack_sq4(x_packed, x_vals);
    pack_sq4(y_packed, y_vals);
    fill_min_max(min_vals, max_vals, dim, dim + 200);

    float expected = alaya::simd::l2_sqr_sq4_generic(x_packed.data(), y_packed.data(), dim,
                                                      min_vals.data(), max_vals.data());
    auto result = alaya::simd::l2_sqr_sq4<float, float>(x_packed.data(), y_packed.data(), dim,
                                                          min_vals.data(), max_vals.data());

    EXPECT_NEAR(result, expected, 1e-3F) << "Failed for dim=" << dim;
  }
}

TEST_F(L2SqrSQ4Test, ZeroVector) {
  const size_t kDim = 64;
  size_t num_bytes = (kDim + 1) / 2;
  std::vector<uint8_t> x(num_bytes, 0);
  std::vector<uint8_t> y(num_bytes, 0);
  std::vector<float> min_vals;
  std::vector<float> max_vals;

  fill_min_max(min_vals, max_vals, kDim, 42);

  auto result = alaya::simd::l2_sqr_sq4<float, float>(x.data(), y.data(), kDim,
                                                        min_vals.data(), max_vals.data());
  EXPECT_FLOAT_EQ(result, 0.0F);
}

TEST_F(L2SqrSQ4Test, IdenticalVectors) {
  const size_t kDim = 128;
  std::vector<uint8_t> x_vals;
  std::vector<uint8_t> x_packed;
  std::vector<float> min_vals;
  std::vector<float> max_vals;

  fill_random_sq4(x_vals, kDim, 42);
  pack_sq4(x_packed, x_vals);
  fill_min_max(min_vals, max_vals, kDim, 43);

  auto result = alaya::simd::l2_sqr_sq4<float, float>(x_packed.data(), x_packed.data(), kDim,
                                                        min_vals.data(), max_vals.data());
  EXPECT_FLOAT_EQ(result, 0.0F);
}

TEST_F(L2SqrSQ4Test, LargeDimension) {
  const size_t kDim = 1024;
  std::vector<uint8_t> x_vals;
  std::vector<uint8_t> y_vals;
  std::vector<uint8_t> x_packed;
  std::vector<uint8_t> y_packed;
  std::vector<float> min_vals;
  std::vector<float> max_vals;

  fill_random_sq4(x_vals, kDim, 1);
  fill_random_sq4(y_vals, kDim, 2);
  pack_sq4(x_packed, x_vals);
  pack_sq4(y_packed, y_vals);
  fill_min_max(min_vals, max_vals, kDim, 3);

  float expected = alaya::simd::l2_sqr_sq4_generic(x_packed.data(), y_packed.data(), kDim,
                                                    min_vals.data(), max_vals.data());
  auto result = alaya::simd::l2_sqr_sq4<float, float>(x_packed.data(), y_packed.data(), kDim,
                                                        min_vals.data(), max_vals.data());

  EXPECT_NEAR(result, expected, 1e-2F);
}

TEST_F(L2SqrSQ4Test, QuantizationExtremes) {
  const size_t kDim = 64;
  size_t num_bytes = (kDim + 1) / 2;
  std::vector<uint8_t> x(num_bytes, 0x00);   // All 0 (low=0, high=0)
  std::vector<uint8_t> y(num_bytes, 0xFF);   // All 15 (low=15, high=15)
  std::vector<float> min_vals(kDim, -1.0F);
  std::vector<float> max_vals(kDim, 1.0F);

  // Expected: each dimension contributes ((0 - 15) * (2.0 / 15))^2 = 4.0
  float expected = static_cast<float>(kDim) * 4.0F;
  auto result = alaya::simd::l2_sqr_sq4<float, float>(x.data(), y.data(), kDim, min_vals.data(),
                                                        max_vals.data());

  EXPECT_NEAR(result, expected, 1e-3F);
}

#ifdef ALAYA_X86
TEST_F(L2SqrSQ4Test, AVX2Correctness) {
  const auto& features = alaya::simd::get_cpu_features();
  if (!features.avx2_ || !features.fma_) {
    GTEST_SKIP() << "AVX2 + FMA not available";
  }

  const size_t kDim = 128;
  std::vector<uint8_t> x_vals;
  std::vector<uint8_t> y_vals;
  std::vector<uint8_t> x_packed;
  std::vector<uint8_t> y_packed;
  std::vector<float> min_vals;
  std::vector<float> max_vals;

  fill_random_sq4(x_vals, kDim, 1);
  fill_random_sq4(y_vals, kDim, 2);
  pack_sq4(x_packed, x_vals);
  pack_sq4(y_packed, y_vals);
  fill_min_max(min_vals, max_vals, kDim, 3);

  float expected = alaya::simd::l2_sqr_sq4_generic(x_packed.data(), y_packed.data(), kDim,
                                                    min_vals.data(), max_vals.data());
  auto result = alaya::simd::l2_sqr_sq4_avx2(x_packed.data(), y_packed.data(), kDim, min_vals.data(),
                                               max_vals.data());

  EXPECT_NEAR(result, expected, 1e-3F);
}

TEST_F(L2SqrSQ4Test, AVX512Correctness) {
  const auto& features = alaya::simd::get_cpu_features();
  if (!features.avx512f_) {
    GTEST_SKIP() << "AVX-512 not available";
  }

  const size_t kDim = 128;
  std::vector<uint8_t> x_vals;
  std::vector<uint8_t> y_vals;
  std::vector<uint8_t> x_packed;
  std::vector<uint8_t> y_packed;
  std::vector<float> min_vals;
  std::vector<float> max_vals;

  fill_random_sq4(x_vals, kDim, 1);
  fill_random_sq4(y_vals, kDim, 2);
  pack_sq4(x_packed, x_vals);
  pack_sq4(y_packed, y_vals);
  fill_min_max(min_vals, max_vals, kDim, 3);

  float expected = alaya::simd::l2_sqr_sq4_generic(x_packed.data(), y_packed.data(), kDim,
                                                    min_vals.data(), max_vals.data());
  auto result = alaya::simd::l2_sqr_sq4_avx512(x_packed.data(), y_packed.data(), kDim,min_vals.data(), 
                                                 max_vals.data());

  EXPECT_NEAR(result, expected, 1e-3F);
}
#endif
