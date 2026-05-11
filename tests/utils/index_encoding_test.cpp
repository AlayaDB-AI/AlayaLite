// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "utils/index_encoding.hpp"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <string>
#include <vector>

namespace alaya::index_encoding {

TEST(IndexEncodingTest, Int64EncodingRoundTripsAndSortsLexicographically) {
  const std::array<int64_t, 5> values = {
      std::numeric_limits<int64_t>::min(),
      -7,
      0,
      42,
      std::numeric_limits<int64_t>::max(),
  };

  std::vector<std::string> encoded_values;
  encoded_values.reserve(values.size());
  for (auto value : values) {
    auto encoded = encode_int64(value);
    encoded_values.push_back(encoded);
    EXPECT_EQ(decode_int64(encoded), value);
  }

  EXPECT_TRUE(std::is_sorted(encoded_values.begin(), encoded_values.end()));
}

TEST(IndexEncodingTest, DoubleEncodingRoundTripsAndSortsLexicographically) {
  const std::array<double, 5> values = {-10.5, -0.25, 0.0, 2.5, 42.75};

  std::vector<std::string> encoded_values;
  encoded_values.reserve(values.size());
  for (auto value : values) {
    auto encoded = encode_double(value);
    encoded_values.push_back(encoded);
    EXPECT_DOUBLE_EQ(decode_double(encoded), value);
  }

  EXPECT_TRUE(std::is_sorted(encoded_values.begin(), encoded_values.end()));
}

TEST(IndexEncodingTest, EncodeValueSupportsAllMetadataTypes) {
  EXPECT_EQ(encode_value(std::string("alpha")), "s_alpha");
  EXPECT_EQ(encode_value(true), "b_1");
  EXPECT_EQ(encode_value(false), "b_0");
  EXPECT_EQ(encode_value(int64_t(-7)), "i_" + encode_int64(-7));
  EXPECT_EQ(encode_value(3.5), "d_" + encode_double(3.5));
}

TEST(IndexEncodingTest, FieldKeyHelpersProduceExpectedFormats) {
  const auto encoded_value = encode_value(int64_t(99));
  const auto key = make_field_index_key<uint32_t>("score", encoded_value, 42);

  EXPECT_EQ(key, "f_score_" + encoded_value + "_42");
  EXPECT_EQ(make_field_index_prefix("score", encoded_value), "f_score_" + encoded_value + "_");
  EXPECT_EQ(make_field_prefix("score"), "f_score_");
  EXPECT_EQ(extract_id_from_key<uint32_t>(key), 42U);
}

TEST(IndexEncodingTest, ExtractIdReturnsZeroWhenKeyHasNoDelimiter) {
  EXPECT_EQ(extract_id_from_key<uint32_t>("nosuffix"), 0U);
  EXPECT_THROW(extract_id_from_key<uint32_t>("invalid_key_without_numeric_suffix"),
               std::invalid_argument);
}

}  // namespace alaya::index_encoding
