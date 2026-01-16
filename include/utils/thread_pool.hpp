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
#include <condition_variable>  //NOLINT [build/c++11]
#include <functional>
#include <future>  //NOLINT [build/c++11]
#include <memory>
#include <mutex>  //NOLINT [build/c++11]
#include <queue>
#include <thread>  //NOLINT [build/c++11]
#include <utility>
#include <vector>
namespace alaya {

class ThreadPool {
 public:
  /**
   * @brief Constructs a ThreadPool with a specified number of worker threads.
   *
   * This constructor initializes the thread pool by creating a specified number of threads
   * that will wait for tasks to be assigned. The threads will execute tasks concurrently,
   * allowing for parallel execution of functions. This class can serve as an alternative to
   * OpenMP for managing parallel workloads in C++.
   *
   * @param num_threads The number of threads to create in the pool.
   */
  explicit ThreadPool(size_t num_threads) {
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this]() -> void {
        for (;;) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex_);
            // Wait until a task is available or the pool is stopped
            this->condition_.wait(lock, [this]() -> bool {
              return this->stop_ || !this->tasks_.empty();
            });
            if (this->stop_ && this->tasks_.empty()) {
              return;  // Exit the thread if the pool is stopped and there are no tasks
            }
            task = std::move(this->tasks_.front());
            this->tasks_.pop();
          }
          task();  // Execute the task
          {
            std::lock_guard<std::mutex> lock(this->queue_mutex_);
            tasks_completed_.fetch_add(1);
            condition_tasks_completed_.notify_one();
          }
        }
      });
    }
  }

  /**
   * @brief Enqueues a task to be executed by the thread pool.
   *
   * This function allows users to submit a callable (function, lambda, etc.) along with its
   * arguments to the thread pool. The task will be executed by one of the worker threads
   * when it becomes available. The function returns a std::future that can be used to retrieve
   * the result of the task once it is completed.
   *
   * @tparam F The type of the callable object.
   * @tparam Args The types of the arguments passed to the callable.
   * @param f The callable to be executed.
   * @param args The arguments to be passed to the callable.
   * @return A std::future representing the result of the task.
   */
  template <class F, class... Args>
  auto enqueue(F &&f, Args &&...args) -> std::future<std::invoke_result_t<F, Args...>> {
    using return_type = std::invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<return_type> res = task->get_future();
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (stop_) {
        throw std::runtime_error("enqueue on stopped ThreadPool");
      }
      tasks_.emplace([task]() -> auto {
        (*task)();
      });  // Store the task in the queue
    }
    condition_.notify_one();  // Notify one waiting thread
    return res;               // Return the future for the task result
  }

  /**
   * @brief Blocks the calling thread until all specified tasks in the thread pool have completed.
   *
   * This method uses a condition variable to synchronize the calling thread with the worker threads
   * in the thread pool. It waits until the number of completed tasks matches the specified total
   * number of tasks (`task_num`) that have been enqueued. This ensures that the calling thread
   * does not proceed until all expected tasks have finished executing.
   *
   * The mutex (`queue_mutex_`) is used to protect access to the shared variable `tasks_completed_`
   * to ensure thread safety and prevent race conditions. The condition variable
   * (`condition_tasks_completed_`) is notified whenever a task completes, allowing the waiting
   * thread to check the completion condition.
   *
   * @param task_num The total number of tasks that were enqueued and for which the completion
   *                  status is being checked. This parameter allows the method to determine when
   *                  all expected tasks have been completed.
   */
  void wait_until_all_tasks_completed(size_t task_num) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    // Wait until the number of completed tasks matches the total number of tasks
    condition_tasks_completed_.wait(lock, [this, task_num]() -> bool {
      return tasks_completed_.load() == task_num;
    });
  }

  void reset_task() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    tasks_completed_.exchange(0);
  }

  /**
   * @brief Destructor that joins all worker threads.
   *
   * This destructor ensures that all worker threads are properly joined before the ThreadPool
   * object is destroyed. It sets the stop flag, notifies all waiting threads, and waits for
   * each thread to finish executing any remaining tasks.
   */
  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;  // Stop accepting new tasks
    }
    condition_.notify_all();  // Wake up all threads to exit
    for (std::thread &worker : workers_) {
      worker.join();  // Wait for each worker to finish
    }
  }

 private:
  std::vector<std::thread> workers_;         // Vector of worker threads
  std::queue<std::function<void()>> tasks_;  // Queue of tasks to be executed
  std::mutex queue_mutex_;                   // Mutex for synchronizing access to the task queue
  std::condition_variable condition_;        // Condition variable for notifying threads
  bool stop_ = false;                        // Flag to indicate if the thread pool is stopping

  std::condition_variable condition_tasks_completed_;
  std::atomic<uint32_t> tasks_completed_ = 0;
};
}  // namespace alaya
