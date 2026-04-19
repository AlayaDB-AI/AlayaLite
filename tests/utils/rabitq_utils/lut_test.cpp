// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "utils/rabitq_utils/lut.hpp"

namespace alaya {

TEST(LutTest, SimpleExample) {
  size_t padded_dim = 64;
  std::vector<float> rotated_query(padded_dim, 1.0F);
  Lut<float> lookup_table;
  lookup_table = Lut<float>(rotated_query.data(), padded_dim);
  float delta = 4.0F / 255.0F;  // vr = 4, 4/(2^8-1) = 4/255

  EXPECT_EQ(lookup_table.delta(), delta);
  EXPECT_EQ(lookup_table.sum_vl(), 0);  // vl = 0

  EXPECT_EQ(*(lookup_table.lut() + 0), std::round(0 / delta));
  EXPECT_EQ(*(lookup_table.lut() + 1), std::round(1.0F / delta));
  EXPECT_EQ(*(lookup_table.lut() + 3), std::round(2.0F / delta));
  EXPECT_EQ(*(lookup_table.lut() + 7), std::round(3.0F / delta));
  EXPECT_EQ(*(lookup_table.lut() + 15), std::round(4.0F / delta));
}

TEST(LutTest, InvalidDataType) {
  size_t padded_dim = 64;
  std::vector<int> rotated_query(padded_dim, 1);
  EXPECT_THROW(auto lookup_table = Lut(rotated_query.data(), padded_dim), std::invalid_argument);
}

TEST(LutTest, ScalarFastScanAccumulatesUnsignedLookupValues) {
  std::array<uint8_t, fastscan::kBatchSize> codes{};
  std::array<uint8_t, fastscan::kBatchSize> lookup_table{};
  std::array<uint16_t, fastscan::kBatchSize> result{};

  lookup_table[0] = 200;
  lookup_table[16] = 250;

  fastscan::detail::accumulate_scalar(codes.data(), lookup_table.data(), result.data(), 8);

  for (const auto value : result) {
    EXPECT_EQ(value, 450);
  }
}

}  // namespace alaya
