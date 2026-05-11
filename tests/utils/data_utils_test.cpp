// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "utils/data_utils.hpp"
#include <gtest/gtest.h>
#include "simd/distance_ip.hpp"

namespace alaya {

TEST(NormalizationTest, simple) {
  std::vector<float> x = {1.0F, 2.0F, 3.0F};
  std::vector<float> y = {3.0F, 4.0F, 3.0F};

  auto actual = alaya::cos_dist(x.data(), y.data(), x.size());

  alaya::normalize(x.data(), x.size());
  alaya::normalize(y.data(), y.size());
  auto dist = simd::ip_sqr<float, float>(x.data(), y.data(), x.size());

  EXPECT_FLOAT_EQ(actual, dist);
}

}  // namespace alaya
