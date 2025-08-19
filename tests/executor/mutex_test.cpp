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