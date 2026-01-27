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
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <unordered_set>
#include <vector>
#include "executor/jobs/graph_search_job.hpp"
#include "executor/jobs/graph_update_job.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "space/raw_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/log.hpp"

namespace alaya {

class UpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path data_dir = std::filesystem::current_path().parent_path() / "data";
    ds_ = load_dataset(sift_small(data_dir));
  }

  void TearDown() override {}

  Dataset ds_;
  std::unordered_set<uint32_t> point_set_;  ///< The set of points that has been inserted.
};

TEST_F(UpdateTest, HalfInsertTest) {
  uint32_t topk = 10;
  uint32_t half_size = ds_.data_.size() / ds_.dim_ / 2;

  LOG_DEBUG("the data size is {}", ds_.data_.size());
  auto space = std::make_shared<alaya::RawSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);

  // Use the first half of the data to build the graph.
  space->fit(ds_.data_.data(), half_size);

  auto build_start = std::chrono::steady_clock::now();

  alaya::HNSWBuilder<alaya::RawSpace<>> hnsw = alaya::HNSWBuilder<alaya::RawSpace<>>(space);
  std::shared_ptr<alaya::Graph<>> hnsw_graph = hnsw.build_graph();

  auto build_end = std::chrono::steady_clock::now();
  auto build_time = static_cast<std::chrono::duration<double>>(build_end - build_start).count();
  LOG_INFO("The time of building hnsw is {}s.", build_time);

  std::vector<float> half_data(half_size * ds_.dim_);
  half_data.insert(half_data.begin(), ds_.data_.begin(), ds_.data_.begin() + half_size * ds_.dim_);

  auto half_gt = find_exact_gt<>(ds_.queries_, half_data, ds_.dim_, topk);

  auto search_job = std::make_shared<alaya::GraphSearchJob<alaya::RawSpace<>>>(space, hnsw_graph);
  std::vector<uint32_t> ids(ds_.query_num_ * topk);
  size_t ef_1 = 30;
  std::vector<uint32_t> ef_results(ef_1);
  for (uint32_t i = 0; i < ds_.query_num_; i++) {
    auto cur_query = ds_.queries_.data() + i * ds_.dim_;
    search_job->search_solo(cur_query, ef_results.data(), ef_1);
    std::copy(ef_results.begin(), ef_results.begin() + topk, ids.data() + i * topk);
  }

  auto recall = calc_recall(ids.data(), half_gt.data(), ds_.query_num_, topk, topk);
  ASSERT_GT(recall, 0.9);

  auto update_job = std::make_shared<alaya::GraphUpdateJob<RawSpace<>>>(search_job);

  for (uint32_t i = half_size; i < ds_.data_num_; i++) {
    auto cur_data = ds_.data_.data() + i * ds_.dim_;
    update_job->insert_and_update(cur_data, 50);
  }

  size_t ef_2 = 50;
  std::vector<uint32_t> ef_results_2(ef_2);
  for (uint32_t i = 0; i < ds_.query_num_; i++) {
    auto cur_query = ds_.queries_.data() + i * ds_.dim_;
    search_job->search_solo(cur_query, ef_results_2.data(), ef_2);
    std::copy(ef_results_2.begin(), ef_results_2.begin() + topk, ids.data() + i * topk);
  }

  auto full_gt = find_exact_gt(ds_.queries_, ds_.data_, ds_.dim_, topk);
  auto full_recall = calc_recall(ids.data(), full_gt.data(), ds_.query_num_, topk, topk);
  ASSERT_GT(full_recall, 0.9);

  for (uint32_t i = half_size; i < ds_.data_num_; i++) {
    update_job->remove(i);
  }
  size_t ef_3 = 50;
  std::vector<uint32_t> ef_results_3(ef_3);
  for (uint32_t i = 0; i < ds_.query_num_; i++) {
    auto cur_query = ds_.queries_.data() + i * ds_.dim_;
    search_job->search_solo_updated(cur_query, ef_results_3.data(), ef_3);
    std::copy(ef_results_3.begin(), ef_results_3.begin() + topk, ids.data() + i * topk);
  }
  auto recall_after_delete = calc_recall(ids.data(), full_gt.data(), ds_.query_num_, topk, topk);
  LOG_INFO("The recall after delete is {}", recall_after_delete);

  auto gt_after_delete =
      find_exact_gt<>(ds_.queries_, ds_.data_, ds_.dim_, topk,
                      &update_job->job_context_->removed_vertices_);

  auto recall_after_delete_gt = calc_recall(ids.data(), gt_after_delete.data(), ds_.query_num_, topk, topk);
  LOG_INFO("The recall after delete gt is {}", recall_after_delete_gt);
}

}  // namespace alaya
