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
#include <limits>
#include <vector>

#include "utils/evaluate.hpp"

namespace alaya {

TEST(FindExactGTTest, BasicFunctionality) {
  std::vector<float> queries = {1.0, 2.0, 3.0};
  std::vector<float> data = {3.0, 2.0, 1.0, 4.0, 5.0, 6.0};
  uint32_t dim = 3;
  uint32_t topk = 2;
  auto result = find_exact_gt(queries, data, dim, topk);
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], 0);  // Closest point
  EXPECT_EQ(result[1], 1);  // Second closest
}

TEST(FindExactGTTest, EmptyData) {
  std::vector<float> queries = {1.0, 2.0, 3.0};
  std::vector<float> data;
  uint32_t dim = 3;
  uint32_t topk = 1;
  auto result = find_exact_gt(queries, data, dim, topk);
  EXPECT_TRUE(result.empty());
}

TEST(FindExactGTTest, EmptyQueries) {
  std::vector<float> queries;
  std::vector<float> data = {3.0, 2.0, 1.0};
  uint32_t dim = 3;
  uint32_t topk = 1;
  auto result = find_exact_gt(queries, data, dim, topk);
  EXPECT_TRUE(result.empty());
}

TEST(FindExactGTTest, LargeDataset) {
  std::vector<float> queries(300, 1.0);
  std::vector<float> data(3000, 2.0);
  uint32_t dim = 3;
  uint32_t topk = 5;
  auto result = find_exact_gt(queries, data, dim, topk);
  ASSERT_EQ(result.size(), queries.size() / dim * topk);
}

TEST(CalcRecallTest, PerfectMatch) {
  std::vector<uint32_t> res = {0, 1, 2, 3};
  std::vector<uint32_t> gt = {0, 1, 2, 3};
  uint32_t topk = 1;
  float recall = calc_recall(res.data(), gt.data(), 4, 1, topk);
  EXPECT_FLOAT_EQ(recall, 1.0);
}

TEST(CalcRecallTest, PartialMatch) {
  std::vector<uint32_t> res = {0, 1, 2, 3};
  std::vector<uint32_t> gt = {1, 2, 3, 4};
  uint32_t topk = 1;
  float recall = calc_recall(res.data(), gt.data(), 4, 1, topk);
  EXPECT_FLOAT_EQ(recall, 0);
}

TEST(CalcRecallTest, NoMatch) {
  std::vector<uint32_t> res = {5, 6, 7, 8};
  std::vector<uint32_t> gt = {1, 2, 3, 4};
  uint32_t topk = 1;
  float recall = calc_recall(res.data(), gt.data(), 4, 1, topk);
  EXPECT_FLOAT_EQ(recall, 0.0);
}

// Tests for calc_recall with vector<vector<IDType>> overload
TEST(CalcRecallVectorTest, PerfectMatch) {
  std::vector<std::vector<uint32_t>> res = {{0, 1}, {2, 3}};
  std::vector<uint32_t> gt = {0, 1, 2, 3};  // gt_dim = 2
  uint32_t topk = 2;
  float recall = calc_recall(res, gt.data(), 2, 2, topk);
  EXPECT_FLOAT_EQ(recall, 1.0);
}

TEST(CalcRecallVectorTest, PartialMatch) {
  std::vector<std::vector<uint32_t>> res = {{0, 5}, {2, 6}};  // 0 and 2 match
  std::vector<uint32_t> gt = {0, 1, 2, 3};  // gt_dim = 2
  uint32_t topk = 2;
  float recall = calc_recall(res, gt.data(), 2, 2, topk);
  EXPECT_FLOAT_EQ(recall, 0.5);  // 2 out of 4 match
}

TEST(CalcRecallVectorTest, NoMatch) {
  std::vector<std::vector<uint32_t>> res = {{10, 11}, {12, 13}};
  std::vector<uint32_t> gt = {0, 1, 2, 3};
  uint32_t topk = 2;
  float recall = calc_recall(res, gt.data(), 2, 2, topk);
  EXPECT_FLOAT_EQ(recall, 0.0);
}

TEST(CalcRecallVectorTest, SingleQuery) {
  std::vector<std::vector<uint32_t>> res = {{0, 1, 2}};
  std::vector<uint32_t> gt = {0, 1, 2, 3, 4};  // gt_dim = 5
  uint32_t topk = 3;
  float recall = calc_recall(res, gt.data(), 1, 5, topk);
  EXPECT_FLOAT_EQ(recall, 1.0);  // All 3 results are in top-5 GT
}

TEST(CalcRecallVectorTest, LargerGtDim) {
  // Test when gt_dim > topk (common case)
  std::vector<std::vector<uint32_t>> res = {{0}, {5}};  // topk = 1
  std::vector<uint32_t> gt = {0, 1, 2, 5, 6, 7};  // gt_dim = 3, query_num = 2
  uint32_t topk = 1;
  float recall = calc_recall(res, gt.data(), 2, 3, topk);
  EXPECT_FLOAT_EQ(recall, 1.0);  // Both 0 and 5 are in their respective GT
}

}  // namespace alaya
