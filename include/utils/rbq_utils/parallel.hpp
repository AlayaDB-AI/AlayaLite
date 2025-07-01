#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

template <class Function>
inline void parallel_for(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr last_exception = nullptr;
        std::mutex last_except_mutex;

        threads.reserve(numThreads);
        for (size_t thread_id = 0; thread_id < numThreads; ++thread_id) {
            threads.push_back(std::thread([&, thread_id] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, thread_id);
                    } catch (...) {
                        std::unique_lock<std::mutex> last_except_lock(last_except_mutex);
                        last_exception = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        if (last_exception) {
            std::rethrow_exception(last_exception);
        }
    }
}