// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "executor/task_queue.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace alaya {
TEST(TaskQueueTest, ThreadSafePushPop) {
  alaya::TaskQueue queue;
  constexpr int kN = 1000;
  std::vector<std::thread> threads;

  threads.reserve(2);

  // Concurrent push
  for (int i = 0; i < 2; ++i) {
    threads.emplace_back([&] {
      for (int j = 0; j < kN; ++j) {
        queue.push(std::noop_coroutine());
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  // Concurrent pop
  std::atomic<int> popped{0};
  threads.clear();
  threads.reserve(4);

  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&] {
      std::coroutine_handle<> h;
      while (popped < 2 * kN) {
        if (queue.pop(h)) {
          popped++;
        }
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }
  EXPECT_EQ(popped.load(), 2 * kN);
}
}  // namespace alaya
