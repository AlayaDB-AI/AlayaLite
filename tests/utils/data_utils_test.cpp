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

#include "utils/data_utils.hpp"
#include <gtest/gtest.h>
#include "space/distance/dist_ip.hpp"

namespace alaya {

TEST(NormalizationTest, simple) {
  std::vector<float> x = {1.0F, 2.0F, 3.0F};
  std::vector<float> y = {3.0F, 4.0F, 3.0F};

  auto actual = alaya::cos_dist(x.data(), y.data(), x.size());

  alaya::normalize(x.data(), x.size());
  alaya::normalize(y.data(), y.size());
  auto dist = alaya::ip_sqr<float, float>(x.data(), y.data(), x.size());

  EXPECT_FLOAT_EQ(actual, dist);
}

}  // namespace alaya
