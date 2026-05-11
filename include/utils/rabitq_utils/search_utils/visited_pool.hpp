// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <deque>
#include <mutex>

#include "utils/query_utils.hpp"
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

/**
 * @brief Thread-safe pool of EpochVisitedSet objects for reuse across searches.
 * Uses epoch tags so acquired objects can be reset in O(1) logical time.
 */
template <typename TagType = uint32_t>
class EpochVisitedPool {
 private:
  std::deque<EpochVisitedSet<TagType> *> pool_;
  std::mutex poolguard_;
  size_t numelements_;

 public:
  EpochVisitedPool(size_t init_pool_size, size_t max_elements) : numelements_(max_elements) {
    for (size_t i = 0; i < init_pool_size; ++i) {
      pool_.push_front(new EpochVisitedSet<TagType>(numelements_));
    }
  }

  ~EpochVisitedPool() {
    while (!pool_.empty()) {
      auto *ptr = pool_.front();
      pool_.pop_front();
      delete ptr;
    }
  }

  EpochVisitedPool(const EpochVisitedPool &) = delete;
  auto operator=(const EpochVisitedPool &) -> EpochVisitedPool & = delete;
  EpochVisitedPool(EpochVisitedPool &&) = delete;
  auto operator=(EpochVisitedPool &&) -> EpochVisitedPool & = delete;

  auto acquire() -> EpochVisitedSet<TagType> * {
    EpochVisitedSet<TagType> *res = nullptr;
    {
      std::unique_lock<std::mutex> lock(poolguard_);
      if (!pool_.empty()) {
        res = pool_.front();
        pool_.pop_front();
      } else {
        res = new EpochVisitedSet<TagType>(numelements_);
      }
    }
    res->clear();
    return res;
  }

  void release(EpochVisitedSet<TagType> *vl) {
    std::unique_lock<std::mutex> lock(poolguard_);
    pool_.push_front(vl);
  }
};

}  // namespace alaya
