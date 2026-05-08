/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

/**
 * @file concurrent_queue.hpp
 * @brief Thread-safe queue for managing shared resources (e.g., thread-local buffers).
 *
 * Used in disk-based search to manage a pool of ThreadData structures,
 * allowing multiple search threads to safely acquire and release scratch buffers.
 */

#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <unordered_set>

template <typename T>
class ConcurrentQueue {
  using chrono_us_t = std::chrono::microseconds;
  using mutex_locker = std::unique_lock<std::mutex>;

  std::queue<T> queue_;
  std::mutex mutex_;
  std::mutex push_mut_;
  std::mutex pop_mut_;
  std::condition_variable push_cv_;
  std::condition_variable pop_cv_;
  T null_T_;

 public:
  ConcurrentQueue() {}

  explicit ConcurrentQueue(T nullT) { this->null_T_ = nullT; }

  ~ConcurrentQueue() {
    this->push_cv_.notify_all();
    this->pop_cv_.notify_all();
  }

  // queue stats
  /**
   * @brief Returns the current size of the concurrent queue
   * @return The number of elements currently in the queue
   * @thread-safe This method is thread-safe
   */
  uint64_t size() {
    mutex_locker lk(this->mutex_);
    uint64_t ret = queue_.size();
    lk.unlock();
    return ret;
  }

  bool empty() { return (this->size() == 0); }

  // PUSH BACK
  void push(T &new_val) {
    mutex_locker lk(this->mutex_);
    this->queue_.push(new_val);
    lk.unlock();
  }

  template <class Iterator>
  void insert(Iterator iter_begin, Iterator iter_end) {
    mutex_locker lk(this->mutex_);
    for (Iterator it = iter_begin; it != iter_end; it++) {
      this->queue_.push(*it);
    }
    lk.unlock();
  }

  // POP FRONT
  T pop() {
    mutex_locker lk(this->mutex_);
    if (this->queue_.empty()) {
      lk.unlock();
      return this->null_T_;
    } else {
      T ret = this->queue_.front();
      this->queue_.pop();
      // diskann::cout << "thread_id: " << std::this_thread::get_id() << ",
      // ctx: "
      // << ret.ctx << "\n";
      lk.unlock();
      return ret;
    }
  }

  // register for notifications
  void wait_for_push_notify(chrono_us_t wait_time = chrono_us_t{10}) {
    mutex_locker lk(this->push_mut_);
    this->push_cv_.wait_for(lk, wait_time);
    lk.unlock();
  }

  void wait_for_pop_notify(chrono_us_t wait_time = chrono_us_t{10}) {
    mutex_locker lk(this->pop_mut_);
    this->pop_cv_.wait_for(lk, wait_time);
    lk.unlock();
  }

  // just notify functions
  void push_notify_one() { this->push_cv_.notify_one(); }
  void push_notify_all() { this->push_cv_.notify_all(); }
  void pop_notify_one() { this->pop_cv_.notify_one(); }
  void pop_notify_all() { this->pop_cv_.notify_all(); }
};
