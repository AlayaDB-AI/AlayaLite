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
#include <string>
#include <thread>
#include <vector>
#include "executor/jobs/graph_search_job.hpp"
#include "executor/scheduler.hpp"
#include "fmt/format.h"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "space/raw_space.hpp"
#include "space/sq4_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/io_utils.hpp"
#include "utils/log.hpp"
#include "utils/timer.hpp"

namespace alaya {

class SearchTest : public ::testing::Test {
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

  uint32_t max_thread_num_ = std::thread::hardware_concurrency();
  std::unique_ptr<TestDatasetBase> dataset_;
  std::vector<float> data_;
  std::vector<float> queries_;
  std::vector<uint32_t> answers_;
  uint32_t dim_;
  uint32_t points_num_;
  uint32_t query_num_;
  uint32_t gt_col_;
};

TEST_F(SearchTest, FullGraphTest) {
  const size_t kM = 64;
  std::string index_type = "HNSW";

  std::filesystem::path index_file =
      fmt::format("{}_M{}.{}", dataset_->get_dataset_name(), kM, index_type);
  LOG_INFO("the data size is {}, point number is: {}", data_.size(), points_num_);
  std::shared_ptr<alaya::RawSpace<>> space =
      std::make_shared<alaya::RawSpace<>>(points_num_, dim_, MetricType::L2);
  LOG_INFO("Initialize space successfully!");
  space->fit(data_.data(), points_num_);

  LOG_INFO("Fit space successfully!");
  alaya::Graph<uint32_t> load_graph = alaya::Graph<uint32_t>(points_num_, kM);
  if (!std::filesystem::exists(index_file)) {
    auto build_start = std::chrono::steady_clock::now();

    alaya::HNSWBuilder<alaya::RawSpace<>> hnsw = alaya::HNSWBuilder<alaya::RawSpace<>>(space);
    LOG_INFO("Initialize the hnsw builder successfully!");
    auto hnsw_graph = hnsw.build_graph(max_thread_num_);

    auto build_end = std::chrono::steady_clock::now();
    auto build_time = static_cast<std::chrono::duration<double>>(build_end - build_start).count();
    LOG_INFO("The time of building hnsw is {}s, saving it to {}", build_time, index_file.string());

    std::string_view path = index_file.native();
    hnsw_graph->save(path);
  }
  LOG_INFO("Begin Loading the graph from file: {}", index_file.string());
  std::string_view path = index_file.native();
  load_graph.load(path);

  std::vector<uint32_t> inpoint_num(points_num_);
  std::vector<uint32_t> outpoint_num(points_num_);

  for (uint32_t i = 0; i < points_num_; i++) {
    for (uint32_t j = 0; j < load_graph.max_nbrs_; j++) {
      auto id = load_graph.at(i, j);
      if (id == alaya::Graph<uint32_t>::kEmptyId) {
        break;
      }
      outpoint_num[i]++;
      inpoint_num[id]++;
    }
  }

  uint64_t zero_outpoint_cnt = 0;
  uint64_t zero_inpoint_cnt = 0;

  // Check if edge exists on each node
  for (uint32_t i = 0; i < points_num_; i++) {
    if (outpoint_num[i] != 0) {
      zero_outpoint_cnt++;
    }
    if (inpoint_num[i] != 0) {
      zero_inpoint_cnt++;
    }
  }
  LOG_INFO("no_zero_inpoint = {} , no_zero_oupoint = {}", zero_inpoint_cnt, zero_outpoint_cnt);
  EXPECT_EQ(zero_inpoint_cnt, points_num_);
  EXPECT_EQ(zero_outpoint_cnt, points_num_);
}

TEST_F(SearchTest, SearchHNSWTest) {
  const size_t kM = 64;
  size_t topk = 10;
  size_t ef = 100;
  std::string index_type = "HNSW";

  std::filesystem::path index_file =
      fmt::format("{}_M{}.{}", dataset_->get_dataset_name(), kM, index_type);
  std::shared_ptr<alaya::RawSpace<>> space =
      std::make_shared<alaya::RawSpace<>>(points_num_, dim_, MetricType::L2);
  space->fit(data_.data(), points_num_);

  auto load_graph = std::make_shared<alaya::Graph<>>(points_num_, kM);
  if (!std::filesystem::exists(index_file)) {
    auto build_start = std::chrono::steady_clock::now();

    alaya::HNSWBuilder<alaya::RawSpace<>> hnsw = alaya::HNSWBuilder<alaya::RawSpace<>>(space);
    auto hnsw_graph = hnsw.build_graph(max_thread_num_);

    auto build_end = std::chrono::steady_clock::now();
    auto build_time = static_cast<std::chrono::duration<double>>(build_end - build_start).count();
    LOG_INFO("The time of building hnsw is {}s.", build_time);

    std::string_view path = index_file.native();
    hnsw_graph->save(path);
  }
  std::string_view path = index_file.native();
  load_graph->load(path);

  auto search_job = std::make_unique<alaya::GraphSearchJob<alaya::RawSpace<>>>(space, load_graph);

  LOG_INFO("Create task generator successfully");

  using IDType = uint32_t;

  Timer timer{};
  std::vector<std::vector<IDType>> res_pool(query_num_, std::vector<IDType>(topk));
  const size_t kSearchThreadNum = 16;
  std::vector<std::thread> tasks(kSearchThreadNum);

  auto search_knn = [&](uint32_t i) {
    for (; i < query_num_; i += kSearchThreadNum) {
      std::vector<uint32_t> ids(topk);
      auto cur_query = queries_.data() + i * dim_;
      search_job->search_solo(cur_query, topk, ids.data(), ef);

      auto id_set = std::set(ids.begin(), ids.end());

      if (id_set.size() < topk) {
        fmt::println("i id: {}", i);
        fmt::println("ids size: {}", id_set.size());
      }
      res_pool[i] = ids;
    }
  };

  for (size_t i = 0; i < kSearchThreadNum; i++) {
    tasks[i] = std::thread(search_knn, i);
  }

  for (size_t i = 0; i < kSearchThreadNum; i++) {
    if (tasks[i].joinable()) {
      tasks[i].join();
    }
  }

  LOG_INFO("total time: {} s.", timer.elapsed() / 1000000.0);

  // Computing recall;
  size_t cnt = 0;
  for (uint32_t i = 0; i < query_num_; i++) {
    for (size_t j = 0; j < topk; j++) {
      for (size_t k = 0; k < topk; k++) {
        if (res_pool[i][j] == answers_[i * gt_col_ + k]) {
          cnt++;
          break;
        }
      }
    }
  }

  float recall = cnt * 1.0 / query_num_ / topk;
  LOG_INFO("recall is {}.", recall);
  EXPECT_GE(recall, 0.5);
}

TEST_F(SearchTest, SearchHNSWTestSQSpace) {
  const size_t kM = 64;
  std::string index_type = "HNSW";

  std::filesystem::path index_file =
      fmt::format("{}_M{}_SQ.{}", dataset_->get_dataset_name(), kM, index_type);

  auto load_graph = std::make_shared<alaya::Graph<>>(points_num_, kM);
  if (!std::filesystem::exists(index_file)) {
    std::shared_ptr<alaya::RawSpace<>> build_graph_space =
        std::make_shared<alaya::RawSpace<>>(points_num_, dim_, MetricType::L2);
    build_graph_space->fit(data_.data(), points_num_);
    auto build_start = std::chrono::steady_clock::now();

    alaya::HNSWBuilder<alaya::RawSpace<>> hnsw =
        alaya::HNSWBuilder<alaya::RawSpace<>>(build_graph_space);
    auto hnsw_graph = hnsw.build_graph(max_thread_num_);

    auto build_end = std::chrono::steady_clock::now();
    auto build_time = static_cast<std::chrono::duration<double>>(build_end - build_start).count();
    LOG_INFO("The time of building hnsw is {}s.", build_time);

    std::string_view path = index_file.native();
    hnsw_graph->save(path);
  }
}
}  // namespace alaya
