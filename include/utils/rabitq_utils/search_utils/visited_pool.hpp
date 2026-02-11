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

#pragma once

#include <cstddef>
#include <deque>
#include <mutex>

#include "utils/rabitq_utils/search_utils/hashset.hpp"

namespace alaya {

/**
 * @brief Thread-safe pool of HashBasedBooleanSet objects for reuse across searches.
 * Avoids repeated allocation/deallocation of visited sets in high-throughput search scenarios.
 */
class HashSetPool {
 private:
  std::deque<HashBasedBooleanSet *> pool_;
  std::mutex poolguard_;
  size_t numelements_;

 public:
  HashSetPool(size_t init_pool_size, size_t max_elements) : numelements_(max_elements / 10) {
    for (size_t i = 0; i < init_pool_size; ++i) {
      pool_.push_front(new HashBasedBooleanSet(numelements_));
    }
  }

  ~HashSetPool() {
    while (!pool_.empty()) {
      auto *ptr = pool_.front();
      pool_.pop_front();
      delete ptr;
    }
  }

  HashSetPool(const HashSetPool &) = delete;
  auto operator=(const HashSetPool &) -> HashSetPool & = delete;
  HashSetPool(HashSetPool &&) = delete;
  auto operator=(HashSetPool &&) -> HashSetPool & = delete;

  auto acquire() -> HashBasedBooleanSet * {
    HashBasedBooleanSet *res = nullptr;
    {
      std::unique_lock<std::mutex> lock(poolguard_);
      if (!pool_.empty()) {
        res = pool_.front();
        pool_.pop_front();
      } else {
        res = new HashBasedBooleanSet(numelements_);
      }
    }
    res->clear();
    return res;
  }

  void release(HashBasedBooleanSet *vl) {
    std::unique_lock<std::mutex> lock(poolguard_);
    pool_.push_front(vl);
  }
};

}  // namespace alaya
