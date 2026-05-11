// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "space/quant/sq8.hpp"
#include <gtest/gtest.h>

namespace alaya {
using IDType = uint32_t;

class SQ8QuantizerTest : public ::testing::Test {
 protected:
  void SetUp() override { quantizer_ = SQ8Quantizer<float>(dim_); }

  uint32_t dim_ = 4;
  SQ8Quantizer<float> quantizer_;
};

TEST_F(SQ8QuantizerTest, Constructor) {
  EXPECT_EQ(quantizer_.dim_, dim_);
  EXPECT_EQ(quantizer_.min_vector_.size(), dim_);
  EXPECT_EQ(quantizer_.max_vector_.size(), dim_);
}

TEST_F(SQ8QuantizerTest, Fit) {
  float data[] = {1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5};
  quantizer_.fit(data, 2);

  EXPECT_FLOAT_EQ(quantizer_.min_vector_[0], 0.5);
  EXPECT_FLOAT_EQ(quantizer_.max_vector_[0], 1.0);
  EXPECT_FLOAT_EQ(quantizer_.min_vector_[1], 1.5);
  EXPECT_FLOAT_EQ(quantizer_.max_vector_[1], 2.0);
  EXPECT_FLOAT_EQ(quantizer_.min_vector_[2], 2.5);
  EXPECT_FLOAT_EQ(quantizer_.max_vector_[2], 3.0);
  EXPECT_FLOAT_EQ(quantizer_.min_vector_[3], 3.5);
  EXPECT_FLOAT_EQ(quantizer_.max_vector_[3], 4.0);
}

TEST_F(SQ8QuantizerTest, Quantize) {
  float min_val = 0.0;
  float max_val = 10.0;
  EXPECT_EQ(quantizer_.quantize(0.0, min_val, max_val), 0);
  EXPECT_EQ(quantizer_.quantize(10.0, min_val, max_val), 255);
  EXPECT_EQ(quantizer_.quantize(5.0, min_val, max_val), 127);
}

TEST_F(SQ8QuantizerTest, Encode) {
  float raw_data[] = {0.0, 5.0, 10.0, 7.5};
  uint8_t encoded_data[4] = {0};

  quantizer_.min_vector_ = {0.0, 0.0, 0.0, 0.0};
  quantizer_.max_vector_ = {10.0, 10.0, 10.0, 10.0};

  quantizer_.encode(raw_data, encoded_data);
  EXPECT_EQ(encoded_data[0], 0);
  EXPECT_EQ(encoded_data[1], 127);
  EXPECT_EQ(encoded_data[2], 255);
  EXPECT_EQ(encoded_data[3], 191);
}
}  // namespace alaya
