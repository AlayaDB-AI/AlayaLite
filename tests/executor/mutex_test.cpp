// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#if defined(__linux__)
#include <gtest/gtest.h>
#include <coro/mutex.hpp>
#include <coro/sync_wait.hpp>
#include <coro/task.hpp>
#include <coro/when_all.hpp>

namespace alaya {
TEST(mutexTest, ConcurrentAccess) {
  coro::mutex mutex;
  int counter = 0;
  constexpr int kN = 100;

  auto task = [&](coro::mutex &m) -> coro::task<> {
    for (int i = 0; i < kN; ++i) {
      auto lock = co_await m.lock();
      counter++;
    }
  };

  auto run = [&]() -> coro::task<> { co_await when_all(task(mutex), task(mutex)); };

  sync_wait(run());

  EXPECT_EQ(counter, 2 * kN);
}
}  // namespace alaya
#endif
