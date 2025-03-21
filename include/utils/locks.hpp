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

#pragma once

#include <atomic>
#include <cassert>
class SpinLock {
 public:
  void lock() {
    while (flag_.test_and_set(std::memory_order_acquire)) {
      // 自旋等待锁释放
    }
  }

  auto try_lock() -> bool { return !flag_.test_and_set(std::memory_order_acquire); }

  void unlock() { flag_.clear(std::memory_order_release); }

 private:
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

class SpinLockGuard {
 public:
  explicit SpinLockGuard(SpinLock &lock) : lock_(lock) { lock_.lock(); }

  ~SpinLockGuard() { lock_.unlock(); }

  // 禁用拷贝和赋值操作
  SpinLockGuard(const SpinLockGuard &) = delete;
  auto operator=(const SpinLockGuard &) -> SpinLockGuard & = delete;

 private:
  SpinLock &lock_;
};

class SharedLock {
 public:
  SharedLock() : state_(0) {}

  SharedLock(const SharedLock &) = delete;
  SharedLock(SharedLock &&) = delete;
  auto operator=(const SharedLock &) -> SharedLock & = delete;
  auto operator=(SharedLock &&) -> SharedLock & = delete;
  // acquire a shared lock
  void lock_shared() {
    int expected;
    do {
      expected = state_.load();
      // Wait if the current lock is exclusive lock (-1)
      while (expected == -1) {
        expected = state_.load();
      }
    } while (!state_.compare_exchange_weak(expected, expected + 1));
  }

  // release a shared lock
  void unlock_shared() { state_.fetch_sub(1); }

  // acquire a exclusive lock
  void lock() {
    int expected = 0;
    // Wait until there is no other shared lock and exclusive lock
    while (!state_.compare_exchange_weak(expected, -1)) {
      expected = 0;
    }
  }

  void reset() { state_.store(0); }

  // release an exclusive lock
  void unlock() { state_.store(0); }

  // degrade an exclusive lock to a shared lock
  void degrade_lock() {
    int expected = -1;
    bool ret = state_.compare_exchange_weak(expected, 1);
    assert(ret);
  }

  // upgrade a shared lock to an exclusive lock
  void upgrade_lock() {
    int expected = 1;
    bool ret = state_.compare_exchange_weak(expected, -1);
    assert(ret);
  }

  auto get_state() -> int { return state_.load(); }

  auto no_lock() -> bool { return state_.load() == 0; }

 private:
  std::atomic<int> state_;  // num > 0: shared, -1: exclusive, 0: no
};
