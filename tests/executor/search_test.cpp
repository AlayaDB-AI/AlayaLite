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
#include <string>
#include <thread>
#include <vector>
#include "executor/jobs/graph_search_job.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "space/raw_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/log.hpp"
#include "utils/timer.hpp"

namespace alaya {

class SearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path data_dir = std::filesystem::current_path().parent_path() / "data";
    ds_ = load_dataset(sift_small(data_dir));
  }

  void TearDown() override {}

  uint32_t max_thread_num_ = std::thread::hardware_concurrency();
  Dataset ds_;
};

TEST_F(SearchTest, FullGraphTest) {
  const size_t kM = 64;
  std::string index_type = "HNSW";

  std::filesystem::path index_file = fmt::format("{}_M{}.{}", ds_.name_, kM, index_type);
  LOG_INFO("the data size is {}, point number is: {}", ds_.data_.size(), ds_.data_num_);
  std::shared_ptr<alaya::RawSpace<>> space =
      std::make_shared<alaya::RawSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);
  LOG_INFO("Initialize space successfully!");
  space->fit(ds_.data_.data(), ds_.data_num_);

  LOG_INFO("Fit space successfully!");
  alaya::Graph<uint32_t> load_graph = alaya::Graph<uint32_t>(ds_.data_num_, kM);
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

  std::vector<uint32_t> inpoint_num(ds_.data_num_);
  std::vector<uint32_t> outpoint_num(ds_.data_num_);

  for (uint32_t i = 0; i < ds_.data_num_; i++) {
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
  for (uint32_t i = 0; i < ds_.data_num_; i++) {
    if (outpoint_num[i] != 0) {
      zero_outpoint_cnt++;
    }
    if (inpoint_num[i] != 0) {
      zero_inpoint_cnt++;
    }
  }
  LOG_INFO("no_zero_inpoint = {} , no_zero_oupoint = {}", zero_inpoint_cnt, zero_outpoint_cnt);
  EXPECT_EQ(zero_inpoint_cnt, ds_.data_num_);
  EXPECT_EQ(zero_outpoint_cnt, ds_.data_num_);
}

TEST_F(SearchTest, SearchHNSWTest) {
  const size_t kM = 64;
  size_t topk = 10;
  size_t ef = 100;
  std::string index_type = "HNSW";

  std::filesystem::path index_file = fmt::format("{}_M{}.{}", ds_.name_, kM, index_type);
  std::shared_ptr<alaya::RawSpace<>> space =
      std::make_shared<alaya::RawSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);
  space->fit(ds_.data_.data(), ds_.data_num_);

  auto load_graph = std::make_shared<alaya::Graph<>>(ds_.data_num_, kM);
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
  std::vector<std::vector<IDType>> res_pool(ds_.query_num_, std::vector<IDType>(topk));
  const size_t kSearchThreadNum = 16;
  std::vector<std::thread> tasks(kSearchThreadNum);

  auto search_knn = [&](uint32_t i) {
    for (; i < ds_.query_num_; i += kSearchThreadNum) {
      std::vector<uint32_t> ids(topk);
      auto cur_query = ds_.queries_.data() + i * ds_.dim_;
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
  for (uint32_t i = 0; i < ds_.query_num_; i++) {
    for (size_t j = 0; j < topk; j++) {
      for (size_t k = 0; k < topk; k++) {
        if (res_pool[i][j] == ds_.ground_truth_[i * ds_.gt_dim_ + k]) {
          cnt++;
          break;
        }
      }
    }
  }

  float recall = cnt * 1.0 / ds_.query_num_ / topk;
  LOG_INFO("recall is {}.", recall);
  EXPECT_GE(recall, 0.5);
}

TEST_F(SearchTest, SearchHNSWTestSQSpace) {
  const size_t kM = 64;
  std::string index_type = "HNSW";

  std::filesystem::path index_file = fmt::format("{}_M{}_SQ.{}", ds_.name_, kM, index_type);

  auto load_graph = std::make_shared<alaya::Graph<>>(ds_.data_num_, kM);
  if (!std::filesystem::exists(index_file)) {
    std::shared_ptr<alaya::RawSpace<>> build_graph_space =
        std::make_shared<alaya::RawSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);
    build_graph_space->fit(ds_.data_.data(), ds_.data_num_);
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
