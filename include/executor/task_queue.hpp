// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <atomic>
#include <coroutine>
#include <tuple>
#include "concurrentqueue.h"  // NOLINT
namespace alaya {

/**
 * @brief A thread-safe queue for managing coroutines tasks.
 *
 * This class provides a queue to hold coroutines and allows pushing and popping tasks
 * in a thread-safe manner using `ConcurrentQueue` from the `moodycamel` library.
 * It also tracks the number of tasks in the queue with an atomic counter.
 */
class TaskQueue {
 public:
  TaskQueue() = default;
  ~TaskQueue() = default;

  /**
   * @brief Pushes a coroutine task onto the queue.
   *
   * This function increments the task counter and enqueues the provided coroutine
   * handle into the queue.
   *
   * @param item The coroutine handle representing the task to be enqueued.
   */
  void push(std::coroutine_handle<> item) {
    task_counter_.fetch_add(1, std::memory_order_relaxed);

    queue_.enqueue(item);
  }

  /**
   * @brief Pops a task (coroutine) from the queue and returns it.
   *
   * This function attempts to dequeue a coroutine handle from the queue and if successful,
   * decrements the task counter. It returns a boolean indicating whether a task was dequeued.
   *
   * @param item Reference to the coroutine handle that will hold the dequeued task.
   * @return `true` if a task was successfully dequeued, `false` otherwise.
   */
  auto pop(std::coroutine_handle<> &item) -> bool {
    auto ret = queue_.try_dequeue(item);
    if (ret) {
      task_counter_.fetch_sub(1, std::memory_order_relaxed);
    }
    return ret;
  }

 private:
  std::atomic_size_t task_counter_{0};  ///< tracks the number of tasks in the queue.
  moodycamel::ConcurrentQueue<std::coroutine_handle<>>
      queue_;  ///< A concurrent queue that holds coroutine handles.
};
}  // namespace alaya
