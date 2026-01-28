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
#include "simd/cpu_features.hpp"
#include "simd/distance_ip.hpp"

namespace {

class IpTest : public ::testing::Test {
 protected:
  std::mt19937 gen_{42};
  std::uniform_real_distribution<float> dist_{-1.0F, 1.0F};

  auto fill_random(std::vector<float>& v) -> void {
    for (auto& x : v) {
      x = dist_(gen_);
    }
  }

  static auto reference_ip(const float* x, const float* y, size_t dim) -> float {
    float sum = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
      sum += x[i] * y[i];
    }
    return -sum;
  }
};

TEST_F(IpTest, SimdLevelDetection) {
  const char* level = alaya::simd::get_simd_level_name();
  std::cout << "Detected SIMD level: " << level << '\n';
  EXPECT_NE(level, nullptr);
}

TEST_F(IpTest, GenericCorrectness) {
  constexpr size_t kDim = 128;
  std::vector<float> x(kDim);
  std::vector<float> y(kDim);
  fill_random(x);
  fill_random(y);

  float expected = reference_ip(x.data(), y.data(), kDim);
  float result = alaya::simd::ip_sqr_generic(x.data(), y.data(), kDim);

  EXPECT_NEAR(result, expected, 1e-5F);
}

TEST_F(IpTest, SimdCorrectness) {
  constexpr size_t kDim = 256;
  std::vector<float> x(kDim);
  std::vector<float> y(kDim);
  fill_random(x);
  fill_random(y);

  float expected = alaya::simd::ip_sqr_generic(x.data(), y.data(), kDim);
  auto result = alaya::simd::ip_sqr<float, float>(x.data(), y.data(), kDim);

  EXPECT_NEAR(result, expected, 1e-4F);
}

TEST_F(IpTest, TailHandling) {
  // Test various dimensions that don't align perfectly
  std::vector<size_t> dims = {1, 7, 15, 17, 31, 33, 63, 65, 127, 129};
  for (auto dim : dims) {
    std::vector<float> x(dim);
    std::vector<float> y(dim);
    fill_random(x);
    fill_random(y);

    float expected = alaya::simd::ip_sqr_generic(x.data(), y.data(), dim);
    auto result = alaya::simd::ip_sqr<float, float>(x.data(), y.data(), dim);

    EXPECT_NEAR(result, expected, 1e-4F) << "Failed for dim=" << dim;
  }
}

TEST_F(IpTest, ZeroVector) {
  constexpr size_t kDim = 64;
  std::vector<float> x(kDim, 0.0F);
  std::vector<float> y(kDim);
  fill_random(y);

  auto result = alaya::simd::ip_sqr<float, float>(x.data(), y.data(), kDim);
  EXPECT_NEAR(result, 0.0F, 1e-6F);  // IP of zero vector should be 0
}

TEST_F(IpTest, OrthogonalVectors) {
  // Two orthogonal vectors should have IP close to 0
  std::vector<float> x = {1.0F, 0.0F, 0.0F, 0.0F};
  std::vector<float> y = {0.0F, 1.0F, 0.0F, 0.0F};

  auto result = alaya::simd::ip_sqr<float, float>(x.data(), y.data(), 4);
  EXPECT_NEAR(result, 0.0F, 1e-6F);
}

TEST_F(IpTest, ParallelVectors) {
  // Parallel vectors should have maximum IP (negative since we return -IP)
  std::vector<float> x = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> y = {1.0F, 2.0F, 3.0F, 4.0F};

  float expected = -(1.0F + 4.0F + 9.0F + 16.0F);  // -30
  auto result = alaya::simd::ip_sqr<float, float>(x.data(), y.data(), 4);
  EXPECT_NEAR(result, expected, 1e-5F);
}

TEST_F(IpTest, LargeDimension) {
  constexpr size_t kDim = 1536;
  std::vector<float> x(kDim);
  std::vector<float> y(kDim);
  fill_random(x);
  fill_random(y);

  float expected = alaya::simd::ip_sqr_generic(x.data(), y.data(), kDim);
  auto result = alaya::simd::ip_sqr<float, float>(x.data(), y.data(), kDim);

  EXPECT_NEAR(result, expected, 1e-3F);
}

// SQ8 IP Tests
class IpSQ8Test : public ::testing::Test {
 protected:
  std::mt19937 gen_{42};
  std::uniform_int_distribution<int> dist_{0, 255};
  std::uniform_real_distribution<float> float_dist_{-10.0F, 10.0F};

  auto fill_random(std::vector<uint8_t>& v) -> void {
    for (auto& x : v) {
      x = static_cast<uint8_t>(dist_(gen_));
    }
  }

  auto fill_min_max(std::vector<float>& min_vals, std::vector<float>& max_vals) -> void {
    for (size_t i = 0; i < min_vals.size(); ++i) {
      float a = float_dist_(gen_);
      float b = float_dist_(gen_);
      min_vals[i] = std::min(a, b);
      max_vals[i] = std::max(a, b) + 0.1F;  // Ensure max > min
    }
  }

  static auto reference_ip_sqr_sq8(const uint8_t* x,
                               const uint8_t* y,
                               const float* min,
                               const float* max,
                               size_t dim) -> float {
    constexpr float kInv255 = 1.0F / 255.0F;
    float sum = 0.0F;
    for (size_t i = 0; i < dim; ++i) {
      float scale = (max[i] - min[i]) * kInv255;
      float x_val = min[i] + static_cast<float>(x[i]) * scale;
      float y_val = min[i] + static_cast<float>(y[i]) * scale;
      sum += x_val * y_val;
    }
    return -sum;
  }
};

TEST_F(IpSQ8Test, GenericCorrectness) {
  constexpr size_t kDim = 128;
  std::vector<uint8_t> x(kDim);
  std::vector<uint8_t> y(kDim);
  std::vector<float> min_vals(kDim);
  std::vector<float> max_vals(kDim);
  fill_random(x);
  fill_random(y);
  fill_min_max(min_vals, max_vals);

  float expected =
      reference_ip_sqr_sq8(x.data(), y.data(), min_vals.data(), max_vals.data(), kDim);
  float result = alaya::simd::ip_sqr_sq8_generic(x.data(), y.data(), kDim, min_vals.data(),
                                             max_vals.data());

  EXPECT_NEAR(result, expected, 1e-4F);
}

TEST_F(IpSQ8Test, SimdCorrectness) {
  constexpr size_t kDim = 256;
  std::vector<uint8_t> x(kDim);
  std::vector<uint8_t> y(kDim);
  std::vector<float> min_vals(kDim);
  std::vector<float> max_vals(kDim);
  fill_random(x);
  fill_random(y);
  fill_min_max(min_vals, max_vals);

  float expected = alaya::simd::ip_sqr_sq8_generic(x.data(), y.data(), kDim, min_vals.data(),
                                               max_vals.data());
  auto result = alaya::simd::ip_sqr_sq8<float, float>(x.data(), y.data(), kDim, min_vals.data(),
                                                  max_vals.data());

  EXPECT_NEAR(result, expected, 1e-3F);
}

TEST_F(IpSQ8Test, TailHandling) {
  std::vector<size_t> dims = {1, 7, 15, 17, 31, 33};
  for (auto dim : dims) {
    std::vector<uint8_t> x(dim);
    std::vector<uint8_t> y(dim);
    std::vector<float> min_vals(dim);
    std::vector<float> max_vals(dim);
    fill_random(x);
    fill_random(y);
    fill_min_max(min_vals, max_vals);

    float expected = alaya::simd::ip_sqr_sq8_generic(x.data(), y.data(), dim, min_vals.data(),
                                                 max_vals.data());
    auto result = alaya::simd::ip_sqr_sq8<float, float>(x.data(), y.data(), dim, min_vals.data(),
                                                    max_vals.data());

    EXPECT_NEAR(result, expected, 1e-3F) << "Failed for dim=" << dim;
  }
}

TEST_F(IpSQ8Test, ZeroVector) {
  constexpr size_t kDim = 64;
  std::vector<uint8_t> x(kDim, 0);
  std::vector<uint8_t> y(kDim);
  std::vector<float> min_vals(kDim, 0.0F);
  std::vector<float> max_vals(kDim, 1.0F);
  fill_random(y);

  auto result = alaya::simd::ip_sqr_sq8<float, float>(x.data(), y.data(), kDim, min_vals.data(),
                                                  max_vals.data());
  // When min=0, x[i]=0 means x_val=0, so IP should be 0
  EXPECT_NEAR(result, 0.0F, 1e-5F);
}

TEST_F(IpSQ8Test, IdenticalVectors) {
  constexpr size_t kDim = 64;
  std::vector<uint8_t> x(kDim);
  std::vector<float> min_vals(kDim);
  std::vector<float> max_vals(kDim);
  fill_random(x);
  fill_min_max(min_vals, max_vals);

  auto result = alaya::simd::ip_sqr_sq8<float, float>(x.data(), x.data(), kDim, min_vals.data(),
                                                  max_vals.data());
  // IP of identical vectors should be negative (since we return -IP)
  EXPECT_LT(result, 0.0F);
}

#ifdef ALAYA_ARCH_X86
TEST_F(IpSQ8Test, AVX2Correctness) {
  const auto& features = alaya::simd::get_cpu_features();
  if (!features.avx2_ || !features.fma_) {
    GTEST_SKIP() << "AVX2 + FMA not available";
  }

  constexpr size_t kDim = 128;
  std::vector<uint8_t> x(kDim);
  std::vector<uint8_t> y(kDim);
  std::vector<float> min_vals(kDim);
  std::vector<float> max_vals(kDim);
  fill_random(x);
  fill_random(y);
  fill_min_max(min_vals, max_vals);

  float expected = alaya::simd::ip_sqr_sq8_generic(x.data(), y.data(), kDim, min_vals.data(),
                                               max_vals.data());
  auto result =
      alaya::simd::ip_sqr_sq8_avx2(x.data(), y.data(), kDim, min_vals.data(), max_vals.data());

  EXPECT_NEAR(result, expected, 1e-3F);
}

TEST_F(IpSQ8Test, AVX512Correctness) {
  const auto& features = alaya::simd::get_cpu_features();
  if (!features.avx512f_) {
    GTEST_SKIP() << "AVX-512 not available";
  }

  constexpr size_t kDim = 128;
  std::vector<uint8_t> x(kDim);
  std::vector<uint8_t> y(kDim);
  std::vector<float> min_vals(kDim);
  std::vector<float> max_vals(kDim);
  fill_random(x);
  fill_random(y);
  fill_min_max(min_vals, max_vals);

  float expected = alaya::simd::ip_sqr_sq8_generic(x.data(),  y.data(),kDim, min_vals.data(),
                                               max_vals.data());
  auto result =
      alaya::simd::ip_sqr_sq8_avx512(x.data(), y.data(), kDim, min_vals.data(), max_vals.data());

  EXPECT_NEAR(result, expected, 1e-3F);
}
#endif

// SQ4 IP Tests
class IpSQ4Test : public ::testing::Test {
 protected:
  std::mt19937 gen_{42};
  std::uniform_int_distribution<int> dist_{0, 15};
  std::uniform_real_distribution<float> float_dist_{-10.0F, 10.0F};

  auto pack_sq4(std::vector<uint8_t>& packed, size_t dim) -> void {
    size_t num_bytes = (dim + 1) / 2;
    packed.resize(num_bytes);
    for (size_t i = 0; i < num_bytes; ++i) {
      auto lo = static_cast<uint8_t>(dist_(gen_));
      auto hi = static_cast<uint8_t>(dist_(gen_));
      packed[i] = (hi << 4) | lo;
    }
  }

  auto fill_min_max(std::vector<float>& min_vals, std::vector<float>& max_vals) -> void {
    for (size_t i = 0; i < min_vals.size(); ++i) {
      float a = float_dist_(gen_);
      float b = float_dist_(gen_);
      min_vals[i] = std::min(a, b);
      max_vals[i] = std::max(a, b) + 0.1F;
    }
  }

  static auto reference_ip_sqr_sq4(const uint8_t* x,
                               const uint8_t* y,
                               const float* min,
                               const float* max,
                               size_t dim) -> float {
    constexpr float kInv15 = 1.0F / 15.0F;
    float sum = 0.0F;
    size_t byte_idx = 0;
    for (size_t i = 0; i < dim; i += 2, ++byte_idx) {
      uint8_t x_lo = x[byte_idx] & 0x0F;
      uint8_t y_lo = y[byte_idx] & 0x0F;
      float scale_lo = (max[i] - min[i]) * kInv15;
      float x_val_lo = min[i] + static_cast<float>(x_lo) * scale_lo;
      float y_val_lo = min[i] + static_cast<float>(y_lo) * scale_lo;
      sum += x_val_lo * y_val_lo;

      if (i + 1 < dim) {
        uint8_t x_hi = (x[byte_idx] >> 4) & 0x0F;
        uint8_t y_hi = (y[byte_idx] >> 4) & 0x0F;
        float scale_hi = (max[i + 1] - min[i + 1]) * kInv15;
        float x_val_hi = min[i + 1] + static_cast<float>(x_hi) * scale_hi;
        float y_val_hi = min[i + 1] + static_cast<float>(y_hi) * scale_hi;
        sum += x_val_hi * y_val_hi;
      }
    }
    return -sum;
  }
};

TEST_F(IpSQ4Test, GenericCorrectness) {
  constexpr size_t kDim = 128;
  std::vector<uint8_t> x_packed;
  std::vector<uint8_t> y_packed;
  std::vector<float> min_vals(kDim);
  std::vector<float> max_vals(kDim);
  pack_sq4(x_packed, kDim);
  pack_sq4(y_packed, kDim);
  fill_min_max(min_vals, max_vals);

  float expected =
      reference_ip_sqr_sq4(x_packed.data(), y_packed.data(), min_vals.data(), max_vals.data(), kDim);
  float result = alaya::simd::ip_sqr_sq4_generic(x_packed.data(), y_packed.data(), kDim, min_vals.data(),
                                             max_vals.data());

  EXPECT_NEAR(result, expected, 1e-3F);
}

TEST_F(IpSQ4Test, SimdCorrectness) {
  constexpr size_t kDim = 256;
  std::vector<uint8_t> x_packed;
  std::vector<uint8_t> y_packed;
  std::vector<float> min_vals(kDim);
  std::vector<float> max_vals(kDim);
  pack_sq4(x_packed, kDim);
  pack_sq4(y_packed, kDim);
  fill_min_max(min_vals, max_vals);

  float expected = alaya::simd::ip_sqr_sq4_generic(x_packed.data(), y_packed.data(),
                                               kDim, min_vals.data(), max_vals.data());
  auto result = alaya::simd::ip_sqr_sq4<float, float>(x_packed.data(), y_packed.data(),
                                                  kDim, min_vals.data(), max_vals.data());

  EXPECT_NEAR(result, expected, 1e-2F);
}

TEST_F(IpSQ4Test, TailHandling) {
  std::vector<size_t> dims = {2, 6, 14, 18, 30, 34};
  for (auto dim : dims) {
    std::vector<uint8_t> x_packed;
    std::vector<uint8_t> y_packed;
    std::vector<float> min_vals(dim);
    std::vector<float> max_vals(dim);
    pack_sq4(x_packed, dim);
    pack_sq4(y_packed, dim);
    fill_min_max(min_vals, max_vals);

    float expected = alaya::simd::ip_sqr_sq4_generic(x_packed.data(), y_packed.data(),
                                                 dim, min_vals.data(), max_vals.data());
    auto result = alaya::simd::ip_sqr_sq4<float, float>(x_packed.data(), y_packed.data(),
                                                    dim, min_vals.data(), max_vals.data());

    EXPECT_NEAR(result, expected, 1e-2F) << "Failed for dim=" << dim;
  }
}

TEST_F(IpSQ4Test, ZeroVector) {
  constexpr size_t kDim = 64;
  size_t num_bytes = (kDim + 1) / 2;
  std::vector<uint8_t> x(num_bytes, 0);
  std::vector<uint8_t> y(num_bytes);
  std::vector<float> min_vals(kDim, 0.0F);
  std::vector<float> max_vals(kDim, 1.0F);
  pack_sq4(y, kDim);

  auto result = alaya::simd::ip_sqr_sq4<float, float>(x.data(), y.data(), kDim, min_vals.data(),
                                                  max_vals.data());
  EXPECT_NEAR(result, 0.0F, 1e-5F);
}

TEST_F(IpSQ4Test, IdenticalVectors) {
  constexpr size_t kDim = 64;
  std::vector<uint8_t> x_packed;
  std::vector<float> min_vals(kDim);
  std::vector<float> max_vals(kDim);
  pack_sq4(x_packed, kDim);
  fill_min_max(min_vals, max_vals);

  auto result = alaya::simd::ip_sqr_sq4<float, float>(x_packed.data(), x_packed.data(),
                                                  kDim, min_vals.data(), max_vals.data());
  EXPECT_LT(result, 0.0F);
}

#ifdef ALAYA_ARCH_X86
TEST_F(IpSQ4Test, AVX2Correctness) {
  const auto& features = alaya::simd::get_cpu_features();
  if (!features.avx2_ || !features.fma_) {
    GTEST_SKIP() << "AVX2 + FMA not available";
  }

  constexpr size_t kDim = 128;
  std::vector<uint8_t> x_packed;
  std::vector<uint8_t> y_packed;
  std::vector<float> min_vals(kDim);
  std::vector<float> max_vals(kDim);
  pack_sq4(x_packed, kDim);
  pack_sq4(y_packed, kDim);
  fill_min_max(min_vals, max_vals);

  float expected = alaya::simd::ip_sqr_sq4_generic(x_packed.data(), y_packed.data(),
                                               kDim, min_vals.data(), max_vals.data());
  auto result = alaya::simd::ip_sqr_sq4_avx2(x_packed.data(), y_packed.data(), kDim, min_vals.data(),
                                         max_vals.data());

  EXPECT_NEAR(result, expected, 1e-2F);
}

TEST_F(IpSQ4Test, AVX512Correctness) {
  const auto& features = alaya::simd::get_cpu_features();
  if (!features.avx512f_) {
    GTEST_SKIP() << "AVX-512 not available";
  }

  constexpr size_t kDim = 128;
  std::vector<uint8_t> x_packed;
  std::vector<uint8_t> y_packed;
  std::vector<float> min_vals(kDim);
  std::vector<float> max_vals(kDim);
  pack_sq4(x_packed, kDim);
  pack_sq4(y_packed, kDim);
  fill_min_max(min_vals, max_vals);

  float expected = alaya::simd::ip_sqr_sq4_generic(x_packed.data(), y_packed.data(), kDim,
                                               min_vals.data(), max_vals.data());
  auto result = alaya::simd::ip_sqr_sq4_avx512(x_packed.data(), y_packed.data(), kDim, min_vals.data(),
                                           max_vals.data());

  EXPECT_NEAR(result, expected, 1e-2F);
}
#endif

}  // namespace
