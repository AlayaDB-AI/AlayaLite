// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#if defined(__linux__)
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <memory>
#include <thread>
#include <vector>
#include "coro/task.hpp"
#include "executor/jobs/graph_search_job.hpp"
#include "executor/jobs/graph_update_job.hpp"
#include "executor/scheduler.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "space/raw_space.hpp"
#include "utils/dataset_utils.hpp"

namespace alaya {

using coro::task;

namespace {

struct WorkerLocalYield {
  bool await_ready() const noexcept { return false; }

  template <typename Promise>
  void await_suspend(std::coroutine_handle<Promise>) const noexcept {}

  void await_resume() const noexcept {}
};

}  // namespace

class HNSWCoroutineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto ds = load_dataset(sift_tiny(resolve_data_dir()));
    data_ = std::move(ds.data_);
    queries_ = std::move(ds.queries_);
    points_num_ = ds.data_num_;
    dim_ = ds.dim_;
    query_num_ = std::min<uint32_t>(ds.query_num_, 4);

    build_hnsw_index();
    init_scheduler();
  }

  void TearDown() override {
    if (scheduler_) {
      scheduler_->join();  // Ensure all tasks are processed
    }
  }

  std::shared_ptr<RawSpace<>> space_;
  std::shared_ptr<Graph<>> graph_;
  std::unique_ptr<Scheduler> scheduler_;
  std::vector<float> data_, queries_;
  uint32_t points_num_, dim_, query_num_;

 private:
  void build_hnsw_index() {
    space_ = std::make_shared<RawSpace<>>(points_num_, dim_, MetricType::L2);
    space_->fit(data_.data(), points_num_);

    HNSWBuilder<RawSpace<>> builder(space_);
    graph_ = builder.build_graph(64);
  }

  void init_scheduler() {
    std::vector<CpuID> cpus;
    auto cpu_count = std::max(1U, std::min(2U, std::thread::hardware_concurrency()));
    for (CpuID cpu = 0; cpu < cpu_count; ++cpu) {
      cpus.push_back(cpu);
    }
    scheduler_ = std::make_unique<Scheduler>(cpus);
  }
};

TEST_F(HNSWCoroutineTest, CoroutineSearch) {
  constexpr uint32_t k_ = 10;
  constexpr uint32_t kEf = 100;
  std::atomic<uint32_t> completed_queries{0};
  std::vector<std::vector<uint32_t>> results(query_num_, std::vector<uint32_t>(k_));
  std::vector<std::vector<uint32_t>> expected_results(query_num_, std::vector<uint32_t>(k_));
  std::mutex result_mutex;

  auto baseline_search = std::make_shared<GraphSearchJob<RawSpace<>>>(space_, graph_);
  for (uint32_t i = 0; i < query_num_; ++i) {
    auto *query = queries_.data() + i * dim_;
    baseline_search->search_solo(query, expected_results[i].data(), k_, kEf);
  }

  auto search_task = [&](uint32_t query_id) -> task<> {
    auto search_job = std::make_shared<GraphSearchJob<RawSpace<>>>(space_, graph_);
    auto query = queries_.data() + query_id * dim_;
    std::vector<uint32_t> ids(k_);
    co_await WorkerLocalYield{};
    search_job->search_solo(query, ids.data(), k_, kEf);
    co_await WorkerLocalYield{};

    {
      std::scoped_lock lock(result_mutex);
      results[query_id] = ids;
      completed_queries.fetch_add(1);
    }

    co_return;
  };

  std::vector<std::shared_ptr<task<>>> tasks;
  tasks.reserve(query_num_);
  for (uint32_t i = 0; i < query_num_; ++i) {
    auto t = std::make_shared<task<>>(search_task(i));
    tasks.push_back(t);
    scheduler_->schedule(t->handle());
  }

  scheduler_->begin();
  scheduler_->join();  // Waiting for all tasks to complete

  EXPECT_EQ(completed_queries.load(), query_num_);
  EXPECT_EQ(results, expected_results);
}

// ConcurrentUpdates
TEST_F(HNSWCoroutineTest, ConcurrentUpdates) {
  auto search_job = std::make_shared<GraphSearchJob<RawSpace<>>>(space_, graph_);
  auto update_job = std::make_shared<GraphUpdateJob<RawSpace<>>>(search_job);
  std::mutex graph_mutex;
  std::atomic<uint32_t> completed_ops{0};

  auto update_task = [&](uint32_t node_id) -> task<> {
    co_await WorkerLocalYield{};
    {
      std::scoped_lock lock(graph_mutex);
      update_job->remove(node_id);
    }
    completed_ops.fetch_add(1);
  };

  std::vector<std::shared_ptr<task<>>> tasks;
  constexpr uint32_t kNumUpdates = 4;
  tasks.reserve(kNumUpdates);
  for (uint32_t i = 0; i < kNumUpdates; ++i) {
    auto t = std::make_shared<task<>>(update_task(i % points_num_));
    tasks.push_back(t);
    scheduler_->schedule(t->handle());
  }

  scheduler_->begin();
  scheduler_->join();

  EXPECT_EQ(completed_ops.load(), kNumUpdates);
  EXPECT_EQ(update_job->job_context_->removed_vertices_.size(), kNumUpdates);
  for (uint32_t node_id = 0; node_id < kNumUpdates; ++node_id) {
    EXPECT_TRUE(update_job->job_context_->removed_vertices_.contains(node_id));
  }

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
