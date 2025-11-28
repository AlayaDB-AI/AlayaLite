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

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include <utils/rabitq_utils/fht_avx.hpp>

auto naive_hadamard(const std::vector<float> &input) -> std::vector<float> {
  size_t n = input.size();
  if ((n & (n - 1)) != 0) {
    throw std::invalid_argument("Size must be power of two");
  }
  std::vector<float> result = input;
  for (size_t len = 1; len < n; len <<= 1) {
    for (size_t i = 0; i < n; i += 2 * len) {
      for (size_t j = 0; j < len; ++j) {
        float a = result[i + j];
        float b = result[i + j + len];
        result[i + j] = a + b;
        result[i + j + len] = a - b;
      }
    }
  }
  return result;
}

auto approx_equal(const float *a, const float *b, size_t n, float eps = 1e-5F) -> bool {
  for (size_t i = 0; i < n; ++i) {
    if (std::abs(a[i] - b[i]) > eps) {
      return false;
    }
  }
  return true;
}

// NOLINTBEGIN
struct HadamardCase {
  int logn;
  void (*func)(float *);
  size_t size() const { return 1ULL << logn; }
};
// NOLINTEND

class HadamardTest : public ::testing::TestWithParam<HadamardCase> {};

TEST_P(HadamardTest, Correctness) {
  auto param = GetParam();
  size_t N = param.size();  // NOLINT
  std::vector<float> input(N);
  std::vector<float> expected(N);
  std::vector<float> actual(N);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0F, 1.0F);
  for (size_t i = 0; i < N; ++i) {
    input[i] = dis(gen);
  }

  expected = naive_hadamard(input);
  std::copy(input.begin(), input.end(), actual.begin());  // NOLINT
  param.func(actual.data());
#if defined(__AVX512F__)
  EXPECT_TRUE(approx_equal(expected.data(), actual.data(), N));
#endif
}

INSTANTIATE_TEST_SUITE_P(HadamardFunctions, HadamardTest,
                         ::testing::Values(HadamardCase{6, alaya::helper_float_6},
                                           HadamardCase{7, alaya::helper_float_7},
                                           HadamardCase{8, alaya::helper_float_8},
                                           HadamardCase{9, alaya::helper_float_9},
                                           HadamardCase{10, alaya::helper_float_10},
                                           HadamardCase{11, alaya::helper_float_11}));
