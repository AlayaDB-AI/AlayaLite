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
#if (not defined (__NO_COROUTINE__))
#include <atomic>
#include <coroutine>
#include <memory>
#include <vector>

#include "../utils/locks.hpp"
#include "../utils/types.hpp"
#include "task_queue.hpp"
#include "worker.hpp"
namespace alaya {

/**
 * @brief The Scheduler class is responsible for managing task execution across multiple
 *        worker threads, and coordinating their execution in a coroutine-based manner.
 *
 * The Scheduler manages the lifecycle of worker threads, the distribution of tasks to these
 * workers, and provides a coroutine-based mechanism to resume tasks to a Scheduler-level queue.
 * The scheduling is pull-based, i.e., workers pull tasks from the queue and the detailed logic is
 * defined in the Worker class.
 */
class Scheduler {
  friend class Worker;

 public:
  /**
   * @brief An operation class that represents an `awaitable' for scheduling and resuming
   * tasks.
   *
   * Operations is used after `co_await'. Then, the current coroutine is suspended, and the
   * scheduler will place its coroutine handle.
   */
  class Operation {
    friend class Scheduler;
    /**
     * @brief Constructs an Operation object for scheduling a task.
     *
     * @param scheduler The scheduler instance that will manage the operation.
     * @note Only work thread can create operations when a task is being scheduled.
     */
    explicit Operation(Scheduler &scheduler) noexcept : scheduler_(scheduler) {}

   public:
    /**
     * @brief Checks whether the operation is ready to be resumed.
     *
     * Since tasks are always paused, this function always returns false to allow the executing
     * thread to be switched.
     *
     * @return Always returns false to ensure suspension.
     */
    bool await_ready() const noexcept {  // NOLINT
      return false;
    }

    /**
     * @brief Suspends the current coroutine and schedules it to be resumed later by a worker
     * thread.
     *
     * This function stores the awaiting coroutine and hands control back to the scheduler. The
     * coroutine will be resumed by the worker thread once the task is ready to continue.
     *
     * @param awaiting_coroutine The coroutine that is waiting to be resumed.
     * @return Always returns true to indicate that the coroutine is suspended.
     */
    template <typename Promise>
    auto await_suspend(std::coroutine_handle<Promise> awaiting_coroutine) noexcept -> bool {
      scheduler_.resume(awaiting_coroutine);
      return true;
    }

    /**
     * @brief A no-op function called when the coroutine resumes. It does not perform any actions.
     */
    void await_resume() noexcept {};

   private:
    Scheduler &scheduler_;  // NOLINT  ///< The scheduler instance managing this operation.

    std::coroutine_handle<> awaiting_coroutine_{
        nullptr};  ///< The coroutine that is awaiting to be resumed.
  };

  Scheduler() = delete;
  Scheduler(const Scheduler &) = delete;
  Scheduler(Scheduler &&) = delete;
  auto operator=(const Scheduler &) -> Scheduler & = delete;
  auto operator=(Scheduler &&) -> Scheduler & = delete;

  /**
   * @brief Constructs the Scheduler instance and initializes the task queue.
   *
   * @param cpus A vector of CPU IDs that the scheduler will utilize to distribute tasks across
   * workers.
   */
  explicit Scheduler(std::vector<CpuID> &cpus) : cpus_(cpus) {
    task_queue_ = std::make_unique<TaskQueue>();
  }

  /**
   * @brief Destructor for the Scheduler class. Ensures that all worker threads are joined before
   *        shutting down the scheduler.
   */
  ~Scheduler() {
    bool expected = false;
    bool ret = shutdown_.compare_exchange_strong(expected, true, std::memory_order::release);
    if (ret) {
      for (auto &worker : workers_) {
        worker->join();
      }
    }
  }

  /**
   * @brief Begins the lifecycle of the scheduler, initializing worker threads.
   *
   * This function registers the scheduler as the global instance, starts worker threads for each
   * available CPU, and begins the task scheduling process. The worker threads will process tasks
   * as they are scheduled.
   */
  void begin() {
    for (CpuID i = 0; i < cpus_.size(); i++) {
      workers_.emplace_back(std::make_unique<Worker>(
          i, cpus_.at(i), task_queue_.get(), &this->total_task_count_, &this->total_finish_count_));
    }
    for (auto &worker : workers_) {
      worker->start();
    }
  }

  /**
   * @brief Joins all worker threads and shuts down the scheduler.
   *
   * This function ensures that all worker threads finish executing before the scheduler shuts down.
   */
  void join() {
    bool expected = false;
    bool ret = shutdown_.compare_exchange_strong(expected, true, std::memory_order::release);
    if (ret) {
      for (auto &worker : workers_) {
        worker->join();
      }
    }
  }

  /**
   * @brief Schedules a new task by creating an Operation object.
   *
   * This method returns an `Operation' object that can be used to suspend and resume a task on the
   * worker thread. It is `co_await'-ed in the worker.
   *
   * @return An Operation object representing the scheduled task.
   */
  auto schedule() -> Operation { return Operation{*this}; }

  /**
   * @brief Schedules a coroutine handle to be executed by a worker thread.
   *
   * This method enqueues the given coroutine handle to the task queue, incrementing the total task
   * count. The worker threads will later dequeue and execute the task.
   *
   * @param handle The coroutine handle representing the task to be scheduled.
   */
  void schedule(std::coroutine_handle<> handle) {
    assert(handle != nullptr);
    SpinLockGuard guard(enqueue_lock_);
    total_task_count_.fetch_add(1);
    task_queue_->push(handle);
  }

  /**
   * @brief Resumes a suspended task by pushing its coroutine handle to the task queue.
   *
   * This method is used to resume tasks that were previously suspended and have been scheduled for
   * execution. The handle is pushed to the task queue for worker threads to process.
   *
   * @param handle The coroutine handle representing the task to be resumed.
   */
  void resume(std::coroutine_handle<> handle) {
    assert(handle != nullptr);
    SpinLockGuard guard(enqueue_lock_);
    task_queue_->push(handle);
  }

 private:
  std::vector<CpuID> cpus_;  ///< List of CPU IDs on which worker threads will run.

  std::atomic<std::size_t>
      total_task_count_;  ///< Atomic counter tracking the total number of tasks scheduled.
  std::atomic<std::size_t>
      total_finish_count_;  ///< Atomic counter tracking the total number of completed tasks.

  std::unique_ptr<TaskQueue>
      task_queue_;  ///< The shared task queue that holds tasks for workers to process.

  std::vector<std::unique_ptr<Worker>>
      workers_;  ///< List of workers managing task execution across CPUs.

  SpinLock enqueue_lock_;  ///< Lock used to synchronize task enqueuing to the task queue.

  std::atomic_bool shutdown_{false};  ///< Flag indicating whether the scheduler is shutting down.
};

}  // namespace alaya

#endif
