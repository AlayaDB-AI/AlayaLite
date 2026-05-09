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
#include "utils/rabitq_utils/search_utils/buffer.hpp"

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

#if !defined(__AVX512F__)
TEST(LutTest, FastScanFallbackEstimatesDistances) {
  alignas(64) std::array<uint16_t, fastscan::kBatchSize> nth_segments{};
  alignas(64) std::array<float, fastscan::kBatchSize> f_add{};
  alignas(64) std::array<float, fastscan::kBatchSize> f_rescale{};
  alignas(64) std::array<float, fastscan::kBatchSize> result{};

  for (size_t i = 0; i < fastscan::kBatchSize; ++i) {
    nth_segments[i] = 2;
    f_add[i] = static_cast<float>(i);
    f_rescale[i] = 2.0F;
  }

  fastscan::estimate_distances(nth_segments.data(),
                               f_add.data(),
                               f_rescale.data(),
                               3.0F,
                               0.5F,
                               1.0F,
                               result.data());

  for (size_t i = 0; i < fastscan::kBatchSize; ++i) {
    EXPECT_FLOAT_EQ(result[i], static_cast<float>(i) + 7.0F);
  }
}

TEST(LutTest, FastScanFallbackAccumulatesAndEstimatesDistances) {
  alignas(64) std::array<uint8_t, fastscan::kBatchSize> codes{};
  alignas(64) std::array<uint8_t, fastscan::kBatchSize> lookup_table{};
  alignas(64) std::array<float, fastscan::kBatchSize> f_add{};
  alignas(64) std::array<float, fastscan::kBatchSize> f_rescale{};
  alignas(64) std::array<float, fastscan::kBatchSize> result{};

  lookup_table[0] = 4;
  lookup_table[16] = 6;
  for (size_t i = 0; i < fastscan::kBatchSize; ++i) {
    f_rescale[i] = 0.5F;
  }

  fastscan::accumulate_and_estimate_distances(codes.data(),
                                              lookup_table.data(),
                                              f_add.data(),
                                              f_rescale.data(),
                                              1.0F,
                                              2.0F,
                                              3.0F,
                                              result.data(),
                                              8);

  for (const auto value : result) {
    EXPECT_FLOAT_EQ(value, 12.5F);
  }
}
#endif

TEST(SearchBufferTest, HandlesEmptyResizeAndSortedInsertion) {
  SearchBuffer<float> buffer;

  EXPECT_FALSE(buffer.insert(1, 1.0F));
  EXPECT_EQ(buffer.top_dist(), std::numeric_limits<float>::max());
  EXPECT_FALSE(buffer.has_next());

  buffer.resize(2);
  EXPECT_TRUE(buffer.insert(2, 2.0F));
  EXPECT_TRUE(buffer.insert(1, 1.0F));
  EXPECT_FALSE(buffer.insert(3, 3.0F));
  EXPECT_TRUE(buffer.is_full());
  EXPECT_FLOAT_EQ(buffer.top_dist(), 2.0F);

  std::array<uint32_t, 2> results{};
  buffer.copy_results_to(results.data(), results.size());
  EXPECT_EQ(results[0], 1U);
  EXPECT_EQ(results[1], 2U);

  EXPECT_TRUE(buffer.has_next());
  EXPECT_EQ(buffer.next_id(), 1U);
  EXPECT_EQ(buffer.pop(), 1U);
  EXPECT_TRUE(buffer.has_next());
  EXPECT_EQ(buffer.pop(), 2U);
  EXPECT_FALSE(buffer.has_next());

  buffer.clear();
  EXPECT_EQ(buffer.size(), 0U);
  EXPECT_EQ(buffer.top_dist(), std::numeric_limits<float>::max());
}

}  // namespace alaya
