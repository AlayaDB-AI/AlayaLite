// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "executor/worker.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <memory>
#include <thread>

namespace alaya {

class WorkerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    total_tasks_ = std::make_shared<std::atomic<size_t>>(0);
    finished_tasks_ = std::make_shared<std::atomic<size_t>>(0);
    task_queue_ = std::make_shared<TaskQueue>();

    worker_ = std::make_shared<Worker>(1, 0, task_queue_.get(), total_tasks_.get(),
                                       finished_tasks_.get());
  }

  std::shared_ptr<TaskQueue> task_queue_;
  std::shared_ptr<std::atomic<size_t>> total_tasks_;
  std::shared_ptr<std::atomic<size_t>> finished_tasks_;
  std::shared_ptr<Worker> worker_;

  auto create_mock_task(std::atomic<int> &counter) {
    struct Task {
      std::atomic<int> &counter_;
      void operator()() { counter_++; }
    };
    return std::coroutine_handle<>::from_address(new Task{counter});
  }
};

// Validate basic properties
TEST_F(WorkerTest, Initialization) {
  EXPECT_EQ(worker_->id(), 1);
  EXPECT_EQ(worker_->cpu_id(), 0);
}

}  // namespace alaya
