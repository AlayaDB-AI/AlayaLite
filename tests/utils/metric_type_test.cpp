// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "utils/metric_type.hpp"
#include <gtest/gtest.h>
#include <string_view>

namespace alaya {

// Test operator[]
TEST(MetricTypeTest, NormalCase) {
  EXPECT_EQ(kMetricMap["L2"], MetricType::L2);
  EXPECT_EQ(kMetricMap["IP"], MetricType::IP);
  EXPECT_EQ(kMetricMap["COS"], MetricType::COS);
}

// Test string view copy
TEST(MetricTypeTest, StringViewCopyBehavior) {
  std::string test_key = "L2";
  MetricType metric = kMetricMap[test_key];
  EXPECT_EQ(metric, MetricType::L2);
  EXPECT_EQ(test_key, "L2");
}

}  // namespace alaya
