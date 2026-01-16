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
#include <bitset>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <thread>
#include <unordered_set>
#include <vector>
#include "executor/jobs/graph_search_job.hpp"
#include "executor/jobs/graph_update_job.hpp"
#include "executor/jobs/job_context.hpp"
#include "executor/scheduler.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "space/distance/dist_l2.hpp"
#include "space/raw_space.hpp"
#include "space/sq4_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/io_utils.hpp"
#include "utils/log.hpp"
#include "utils/timer.hpp"

namespace alaya {

class UpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path data_dir = std::filesystem::current_path().parent_path() / "data";
    dataset_ = std::make_unique<SIFTTestData>(data_dir.string());
    dataset_->ensure_dataset();

    data_ = dataset_->get_data();
    queries_ = dataset_->get_queries();
    answers_ = dataset_->get_answers();

    points_num_ = dataset_->get_data_num();
    query_num_ = dataset_->get_query_num();

    dim_ = dataset_->get_dim();
    gt_col_ = dataset_->get_ans_dim();
  }

  void TearDown() override {}

  std::unique_ptr<TestDatasetBase> dataset_;
  std::vector<float> data_;
  std::vector<float> queries_;
  std::vector<uint32_t> answers_;
  uint32_t dim_;
  uint32_t points_num_;
  uint32_t query_num_;
  uint32_t gt_col_;

  std::unordered_set<uint32_t> point_set_;  ///< The set of points that has been inserted.
};

TEST_F(UpdateTest, HalfInsertTest) {
  uint32_t topk = 10;
  uint32_t half_size = data_.size() / dim_ / 2;

  LOG_DEBUG("the data size is {}", data_.size());
  auto space = std::make_shared<alaya::RawSpace<>>(points_num_, dim_, MetricType::L2);

  // Use the first half of the data to build the graph.
  space->fit(data_.data(), half_size);

  auto build_start = std::chrono::steady_clock::now();

  alaya::HNSWBuilder<alaya::RawSpace<>> hnsw = alaya::HNSWBuilder<alaya::RawSpace<>>(space);
  std::shared_ptr<alaya::Graph<>> hnsw_graph = hnsw.build_graph();

  auto build_end = std::chrono::steady_clock::now();
  auto build_time = static_cast<std::chrono::duration<double>>(build_end - build_start).count();
  LOG_INFO("The time of building hnsw is {}s.", build_time);

  std::vector<float> half_data(half_size * dim_);
  half_data.insert(half_data.begin(), data_.begin(), data_.begin() + half_size * dim_);

  auto half_gt = find_exact_gt<>(queries_, half_data, dim_, topk);

  auto search_job = std::make_shared<alaya::GraphSearchJob<alaya::RawSpace<>>>(space, hnsw_graph);
  std::vector<uint32_t> ids(query_num_ * topk);
  for (uint32_t i = 0; i < query_num_; i++) {
    auto cur_query = queries_.data() + i * dim_;
    search_job->search_solo(cur_query, topk, ids.data() + i * topk, 30);
  }

  auto recall = calc_recall(ids, half_gt, topk);
  ASSERT_GT(recall, 0.9);

  auto update_job = std::make_shared<alaya::GraphUpdateJob<RawSpace<>>>(search_job);

  for (uint32_t i = half_size; i < points_num_; i++) {
    auto cur_data = data_.data() + i * dim_;
    update_job->insert_and_update(cur_data, 50);
  }

  for (uint32_t i = 0; i < query_num_; i++) {
    auto cur_query = queries_.data() + i * dim_;
    search_job->search_solo(cur_query, topk, ids.data() + i * topk, 50);
  }

  auto full_gt = find_exact_gt(queries_, data_, dim_, topk);
  auto full_recall = calc_recall(ids, full_gt, topk);
  ASSERT_GT(full_recall, 0.9);

  for (uint32_t i = half_size; i < points_num_; i++) {
    update_job->remove(i);
  }
  for (uint32_t i = 0; i < query_num_; i++) {
    auto cur_query = queries_.data() + i * dim_;
    search_job->search_solo_updated(cur_query, topk, ids.data() + i * topk, 50);
  }
  auto recall_after_delete = calc_recall(ids, full_gt, topk);
  LOG_INFO("The recall after delete is {}", recall_after_delete);

  auto gt_after_delete =
      find_exact_gt<>(queries_, data_, dim_, topk, &update_job->job_context_->removed_vertices_);

  auto recall_after_delete_gt = calc_recall(ids, gt_after_delete, topk);
  LOG_INFO("The recall after delete gt is {}", recall_after_delete_gt);
}

}  // namespace alaya
