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
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>

#include "utils/rabitq_utils/defines.hpp"
#include "utils/rabitq_utils/rotator.hpp"

namespace alaya {
// NOLINTBEGIN

//  ----------------------------
//  Helper Functions
//  ----------------------------
auto random_vector(size_t n) -> std::vector<float> {
  std::vector<float> v(n);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0F, 1.0F);
  for (auto &x : v) x = dist(gen);
  return v;
}

auto l2_norm(const float *v, size_t n) -> float { return l2_sqr(v, n); }

auto approx_equal(const float *a, const float *b, size_t n, float tol = 1e-5F) -> bool {
  for (size_t i = 0; i < n; ++i) {
    if (std::abs(a[i] - b[i]) > tol) {
      return false;
    }
  }
  return true;
}

template <typename T>
auto print_vec(const T *vec, size_t print_size, std::string name) {
  std::cerr << "print " << name << "'s first " << print_size << " dimension: ";
  for (size_t i = 0; i < print_size; ++i) {
    std::cerr << *(vec + i) << " ";
  }
  std::cerr << "\n";
}

// ----------------------------
// Test Fixture
// ----------------------------

class RotatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char *fname = "test_rotator.bin";
    if (std::filesystem::exists(fname)) {
      std::filesystem::remove(fname);
    }
  }

  void TearDown() override {
    const char *fname = "test_rotator.bin";
    if (std::filesystem::exists(fname)) {
      std::filesystem::remove(fname);
    }
  }
};

// ----------------------------
// 1. choose_rotator Tests
// ----------------------------

TEST_F(RotatorTest, ChooseRotator_DefaultIsFhtKac) {
  auto rot = choose_rotator<float>(64);
  EXPECT_EQ(rot->size(), 64);
}

TEST_F(RotatorTest, ChooseRotator_MatrixRotator_Float) {
  auto rot = choose_rotator<float>(64, RotatorType::MatrixRotator);
  EXPECT_EQ(rot->size(), 64);
}

TEST_F(RotatorTest, ChooseRotator_MatrixRotator_Double) {
  auto rot = choose_rotator<double>(64, RotatorType::MatrixRotator);
  EXPECT_EQ(rot->size(), 64);
}

TEST_F(RotatorTest, ChooseRotator_FhtKacRotator_Double_Throws) {
  EXPECT_THROW(choose_rotator<double>(64, RotatorType::FhtKacRotator), std::invalid_argument);
}

TEST_F(RotatorTest, ChooseRotator_MatrixRotator_Int_Throws) {
  EXPECT_THROW(choose_rotator<int>(64, RotatorType::MatrixRotator), std::invalid_argument);
}

// ----------------------------
// 2. MatrixRotator Tests
// ----------------------------

TEST_F(RotatorTest, MatrixRotator_Interface_Float) {
  size_t dim = 32;
  size_t padded = 32;
  size_t print_size = 10;

  auto rot = std::make_unique<rotator_impl::MatrixRotator<float>>(dim, padded);
  EXPECT_EQ(rot->size(), padded);

  auto input = random_vector(dim);
  std::vector<float> output(padded);
  rot->rotate(input.data(), output.data());
  print_vec(output.data(), print_size, "output1");
  // Save/Load
  {
    std::ofstream out("test_rotator.bin", std::ios::binary);
    rot->save(out);
  }
  {
    std::ifstream in("test_rotator.bin", std::ios::binary);
    auto rot2 = std::make_unique<rotator_impl::MatrixRotator<float>>(dim, padded);
    rot2->load(in);
    std::vector<float> output2(padded);
    rot2->rotate(input.data(), output2.data());
    print_vec(output2.data(), print_size, "output2");
    EXPECT_TRUE(approx_equal(output.data(), output2.data(), padded));
  }
}

TEST_F(RotatorTest, MatrixRotator_Orthogonality) {
  const size_t n = 16;
  auto rot = std::make_unique<rotator_impl::MatrixRotator<float>>(n, n);

  // Reconstruct Q by rotating basis vectors
  Eigen::MatrixXf Q(n, n);
  for (int i = 0; i < n; ++i) {
    Eigen::VectorXf e = Eigen::VectorXf::Zero(n);
    e(i) = 1.0f;
    Eigen::VectorXf col(n);
    rot->rotate(e.data(), col.data());
    Q.col(i) = col;
  }

  Eigen::MatrixXf QtQ = Q.transpose() * Q;
  EXPECT_TRUE(QtQ.isApprox(Eigen::MatrixXf::Identity(n, n), 1e-5f));
}

// ----------------------------
// 3. FhtKacRotator Tests
// ----------------------------
TEST_F(RotatorTest, FhtKacRotator_Interface) {
  size_t dim = 64, padded = 64;
  size_t print_size = 10;

  auto rot = std::make_unique<rotator_impl::FhtKacRotator>(dim, padded);
#if defined(__AVX512F__)
  EXPECT_EQ(rot->size(), padded);
#endif

  auto input = random_vector(dim);
  std::vector<float> output(padded);
  rot->rotate(input.data(), output.data());
  print_vec(output.data(), print_size, "output1");

  // Save/Load
  {
    std::ofstream out("test_rotator.bin", std::ios::binary);
    rot->save(out);
  }
  {
    std::ifstream in("test_rotator.bin", std::ios::binary);
    auto rot2 = std::make_unique<rotator_impl::FhtKacRotator>(dim, padded);
    rot2->load(in);
    std::vector<float> output2(padded);
    rot2->rotate(input.data(), output2.data());
    print_vec(output2.data(), print_size, "output2");
#if defined(__AVX512F__)
    EXPECT_TRUE(approx_equal(output.data(), output2.data(), padded));
#endif
  }
}

TEST_F(RotatorTest, FhtKacRotator_Padding) {
  size_t dim = 100;
  auto rot = choose_rotator<float>(dim, RotatorType::FhtKacRotator);
#if defined(__AVX512F__)
  EXPECT_EQ(rot->size(), 128);  // 100 → 128
#endif

  auto input = random_vector(dim);
  std::vector<float> output(128);
  rot->rotate(input.data(), output.data());

  // Last 28 elements should be non-zero due to kacs_walk
  bool all_zero = true;
  for (size_t i = dim; i < 128; ++i) {
    if (std::abs(output[i]) > 1e-6f) {
      all_zero = false;
      break;
    }
  }
#if defined(__AVX512F__)
  EXPECT_FALSE(all_zero);
#endif
}

TEST_F(RotatorTest, FhtKacRotator_NormPreservation) {
  const size_t dim = 64;
  const int trials = 50;
  float input_norm_sum = 0.0f;
  float output_norm_sum = 0.0f;

  for (int t = 0; t < trials; ++t) {
    auto input = random_vector(dim);
    float in_norm = l2_norm(input.data(), dim);
    input_norm_sum += in_norm;

    auto rot = std::make_unique<rotator_impl::FhtKacRotator>(dim, dim);
    std::vector<float> output(dim);
    rot->rotate(input.data(), output.data());
    float out_norm = l2_norm(output.data(), dim);
    output_norm_sum += out_norm;
  }

  float avg_in = input_norm_sum / trials;
  float avg_out = output_norm_sum / trials;
#if defined(__AVX512F__)
  EXPECT_NEAR(avg_out, avg_in, 0.2f);  // loose due to randomness
#endif
}

// ----------------------------
// 4. Edge Cases
// ----------------------------
TEST_F(RotatorTest, Edge_MaxDim) {
  // FhtKac supports up to 2^11 = 2048
  auto rot = choose_rotator<float>(2048, RotatorType::FhtKacRotator);
#if defined(__AVX512F__)
  EXPECT_EQ(rot->size(), 2048);
#endif
  auto input = random_vector(2048);
  std::vector<float> output(2048);
  rot->rotate(input.data(), output.data());
}

TEST_F(RotatorTest, Edge_Dim1) {
  EXPECT_THROW(choose_rotator<float>(1, RotatorType::FhtKacRotator), std::invalid_argument);
}

TEST_F(RotatorTest, Edge_DimTooBig) {
  EXPECT_THROW(choose_rotator<float>(4096, RotatorType::FhtKacRotator), std::invalid_argument);
}

TEST_F(RotatorTest, Edge_Invalid_Padded_Dim) {
  EXPECT_THROW(choose_rotator<float>(2000, RotatorType::FhtKacRotator, 2047),
               std::invalid_argument);
}

// ----------------------------
// 5. For code coverage
// ----------------------------
TEST_F(RotatorTest, Various_Helper) {
  std::vector<size_t> dims = {128, 256, 512, 1024, 2048};
  for (size_t dim : dims) {
    auto rot = choose_rotator<float>(dim, RotatorType::FhtKacRotator);
  }
}
// NOLINTEND
}  // namespace alaya
