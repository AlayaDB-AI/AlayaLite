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

#if defined(__linex__)
#include <gtest/gtest.h>
#include <atomic>
#include <filesystem>
#include <memory>
#include <unordered_set>
#include <vector>
#include "coro/mutex.hpp"
#include "coro/task.hpp"
#include "executor/jobs/graph_search_job.hpp"
#include "executor/jobs/graph_update_job.hpp"
#include "executor/scheduler.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "space/raw_space.hpp"
#include "utils/io_utils.hpp"
#include "utils/log.hpp"

namespace alaya {

using coro::mutex;
using coro::task;

class HNSWCoroutineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!std::filesystem::exists(dir_name_)) {
      GTEST_SKIP() << "Test dataset not found";
    }

    uint32_t ans_dim;
    load_fvecs(dir_name_ / "siftsmall_base.fvecs", data_, points_num_, dim_);
    load_fvecs(dir_name_ / "siftsmall_query.fvecs", queries_, query_num_, dim_);
    load_ivecs(dir_name_ / "siftsmall_groundtruth.ivecs", answers_, ans_num_, ans_dim);

    build_hnsw_index();
    init_scheduler();
  }

  void TearDown() override {
    scheduler_->join();  // Ensure all tasks are processed
  }

  std::filesystem::path dir_name_ = "/home/zijian/zijian/AlayaLite/build/bin/siftsmall";
  std::shared_ptr<RawSpace<>> space_;
  std::shared_ptr<Graph<>> graph_;
  std::unique_ptr<Scheduler> scheduler_;
  std::vector<float> data_, queries_;
  std::vector<uint32_t> answers_;
  uint32_t points_num_, dim_, query_num_, ans_num_;

 private:
  void build_hnsw_index() {
    space_ = std::make_shared<RawSpace<>>(points_num_, dim_, MetricType::L2);
    space_->fit(data_.data(), points_num_);

    HNSWBuilder<RawSpace<>> builder(space_);
    graph_ = builder.build_graph(64);
  }

  void init_scheduler() {
    std::vector<CpuID> cpus{0, 1, 2, 3};
    scheduler_ = std::make_unique<Scheduler>(cpus);
    scheduler_->begin();
  }
};

TEST_F(HNSWCoroutineTest, CoroutineSearch) {
  constexpr uint32_t k_ = 10;
  constexpr uint32_t kEf = 100;
  std::atomic<uint32_t> completed_queries{0};
  std::vector<std::vector<uint32_t>> results(query_num_, std::vector<uint32_t>(k_));
  mutex result_mutex;

  auto search_task = [&](uint32_t query_id) -> task<> {
    auto search_job = std::make_shared<GraphSearchJob<RawSpace<>>>(space_, graph_);
    auto query = queries_.data() + query_id * dim_;
    std::vector<uint32_t> ids(k_);

    LOG_INFO("Starting search task {}", query_id);
    co_await scheduler_->schedule();

    LOG_INFO("Resumed search task {}", query_id);

    search_job->search_solo(query, k_, ids.data(), kEf);
    LOG_INFO("Search completed for task {}", query_id);

    {
      auto lock = co_await result_mutex.lock();
      results[query_id] = ids;
      completed_queries.fetch_add(1);
    }
    LOG_INFO("Updated results for task {}", query_id);

    co_return;
  };

  std::vector<std::shared_ptr<task<>>> tasks;
  tasks.reserve(query_num_);
  for (uint32_t i = 0; i < query_num_; ++i) {
    auto t = std::make_shared<task<>>(search_task(i));
    tasks.push_back(t);
    scheduler_->schedule(t->handle());
  }

  scheduler_->join();  // Waiting for all tasks to complete

  EXPECT_EQ(completed_queries.load(), query_num_);
}

// ConcurrentUpdates
TEST_F(HNSWCoroutineTest, ConcurrentUpdates) {
  auto search_job = std::make_shared<GraphSearchJob<RawSpace<>>>(space_, graph_);
  auto update_job = std::make_shared<GraphUpdateJob<RawSpace<>>>(search_job);
  mutex graph_mutex;
  std::atomic<uint32_t> completed_ops{0};

  auto update_task = [&](uint32_t node_id) -> task<> {
    co_await scheduler_->schedule();
    {
      auto lock = co_await graph_mutex.lock();
      update_job->remove(node_id);
    }
    completed_ops.fetch_add(1);
  };

  std::vector<task<>> tasks;
  constexpr uint32_t kNumUpdates = 10;
  tasks.reserve(kNumUpdates);
  for (uint32_t i = 0; i < kNumUpdates; ++i) {
    tasks.emplace_back(update_task(i % points_num_));
  }

  scheduler_->join();

  uint32_t valid_nodes = 0;
  for (uint32_t i = 0; i < points_num_; ++i) {
    if (!update_job->job_context_->removed_vertices_.contains(i)) {
      ++valid_nodes;
    }
  }
  EXPECT_EQ(valid_nodes, points_num_ - kNumUpdates);
}

}  // namespace alaya

#endif
