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
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <tuple>
#include <vector>

#include "../utils/locks.hpp"
#include "../utils/log.hpp"
#include "../utils/types.hpp"
#include "task_queue.hpp"

namespace alaya {

class Scheduler;
class Worker : public std::enable_shared_from_this<Worker> {
  friend class Scheduler;

 public:
  Worker() = default;  // Designed for running on current cpu

  Worker(WorkerID worker_id,
         CpuID cpu_id,
         TaskQueue *task_queue,
         std::atomic<size_t> *total_task_cnt,
         std::atomic<size_t> *total_finish_cnt,
         uint32_t local_task_cnt = 4)
      : id_(worker_id),
        cpu_id_(cpu_id),
        task_queue_(task_queue),
        local_task_cnt_(local_task_cnt),
        local_tasks_(std::vector<std::coroutine_handle<>>(local_task_cnt)),
        total_task_cnt_(total_task_cnt),
        total_finish_cnt_(total_finish_cnt) {}

  /**
   * @brief Retrieves the unique identifier of the current Worker.
   *
   * @return WorkerID The unique identifier of the Worker.
   */
  auto id() const -> WorkerID { return id_; }

  /**
   * @brief Retrieves the CPU identifier associated with the current Worker.
   *
   * @return CpuID The CPU identifier associated with the Worker (which CPU it will bind to).
   */
  auto cpu_id() const -> CpuID { return cpu_id_; }

  /**
   * @brief Starts the Worker thread.
   *
   * This method creates a new thread and executes the `run` method of the Worker.
   */
  void start() {
    using run_type = void (Worker::*)();
    thread_ = std::thread(static_cast<run_type>(&Worker::run), this);
  }

  /**
   * @brief Joins the Worker thread.
   *
   * This method sets the Worker as inactive and waits for the thread to finish execution.
   * It is invoked by the Scheduler to ensure that all Workers have completed their tasks.
   * The quit logic is implemented in the `run` method.
   */
  void join() {
    active_ = false;
    thread_.join();
  }

  auto operator=(const Worker &) -> Worker & = delete;
  auto operator=(Worker &&) -> Worker & = delete;
  Worker(const Worker &) = delete;
  Worker(Worker &&) = delete;
  ~Worker() = default;

 protected:
  /**
   * @brief Executes tasks (coroutines) in a round-robin manner until all tasks are completed.
   *
   * This function repeatedly selects and resumes tasks from the task queue, handling tasks
   * on a round-robin basis across the available local tasks. It continues processing until
   * all tasks have been completed.
   *
   * @note The function relies on a `local_task_cnt_` counter to determine how many tasks
   * are available for processing. The `task_queue_` is used to pop tasks, and tasks are resumed
   * when selected. The function terminates when all tasks are completed, which is monitored
   * using the `total_finish_cnt_` counter.
   */
  void run() {
    // set_affinity();

    uint32_t navigator = 0;

    while (true) {
      uint32_t idx = navigator++ % local_task_cnt_;
      auto &handle = local_tasks_[idx];

      if (handle == nullptr) {
        auto success = task_queue_->pop(handle);
        if (!success) {
          if (total_finish_cnt_->load() == total_task_cnt_->load()) {
            break;
          }
          continue;
        }
      }
      handle.resume();
      if (handle.done()) {
        handle = nullptr;
        total_finish_cnt_->fetch_add(1);
      } else {
      }
    }
  }

  /**
   * @brief Executes tasks on the current CPU until the task queue is empty.
   *
   * This function processes tasks by popping and resuming tasks from the task queue until
   * the queue is empty. It operates in a straightforward manner, processing one task at a time
   * on the current CPU. The function terminates once there are no more tasks in the queue.
   *
   * @note The function operates only on the current CPU, it can be treated as an optimization when
   * only one core is used.
   */
  void run_on_current_cpu() {
    while (true) {
      std::coroutine_handle<> handle;
      auto success = task_queue_->pop(handle);
      if (!success) {
        break;
      }
      handle.resume();
    }
  }

 private:
  /** @brief Sets the CPU affinity for the current thread.
   *
   * This function sets the CPU affinity for the current thread to the specified CPU ID.
   * By setting the CPU affinity, the thread is bound to run on a particular CPU core,
   * which can improve performance in certain parallel workloads.
   *
   * @note If the `pthread_setaffinity_np` call fails, an error message is logged. It can cause a
   * huge performance degradation when exeucing in an docker environment.
   *
   */
  void set_affinity() {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id_, &cpuset);
    auto return_code = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (return_code != 0) {
      LOG_ERROR("Error calling pthread_setaffinity_np");
    }
#endif
  }

  WorkerID id_{0};   ///< Worker identifier
  CpuID cpu_id_{0};  ///< CPU identifier, represents the CPU on which the worker operates.

  bool active_{true};  ///< Flag indicating whether the worker is active.

  std::thread thread_;  ///< The thread associated with the worker.

  TaskQueue *task_queue_;  ///< Pointer to the task queue. All workers share the same task queue.

  uint32_t local_task_cnt_;  ////< The count of local tasks assigned to the worker.

  std::vector<std::coroutine_handle<>>
      local_tasks_;  ///< The mini-batch of tasks assigned to the worker. Each task is
                     ///< represented by a coroutine handle.

  std::atomic<size_t> *total_task_cnt_;  ///< Pointer to an atomic variable that tracks the
                                         ///< total number of tasks across all workers.
  std::atomic<size_t>
      *total_finish_cnt_;  ///< Pointer to an atomic variable that tracks the number of tasks that
                           ///< have been completed across all workers.
};

}  // namespace alaya
