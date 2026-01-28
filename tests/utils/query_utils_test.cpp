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

#include "utils/query_utils.hpp"
#include "utils/thread_pool.hpp"
#include <gtest/gtest.h>

namespace alaya {

class LinearPoolTest : public ::testing::Test {
 protected:
  void SetUp() override { pool_ = new LinearPool<float, int>(10, 5); }

  void TearDown() override { delete pool_; }

  LinearPool<float, int> *pool_;
};

TEST_F(LinearPoolTest, InsertBoundaryTest) {
  pool_->insert(1, 2.5);
  pool_->insert(2, 1.5);
  pool_->insert(3, 3.0);
  pool_->insert(4, 4.0);
  pool_->insert(5, 5.0);

  EXPECT_FALSE(pool_->insert(6, 6.0));
  EXPECT_EQ(pool_->size(), 5);
}

TEST_F(LinearPoolTest, PopTest) {
  pool_->insert(1, 2.5);
  pool_->insert(2, 1.5);
  pool_->insert(3, 3.0);
  EXPECT_EQ(pool_->top(), 2);

  EXPECT_EQ(pool_->pop(), 2);
  EXPECT_EQ(pool_->pop(), 1);
  EXPECT_EQ(pool_->pop(), 3);
}

// Test for multiple insertions and pops
TEST_F(LinearPoolTest, MultipleInsertAndPopTest) {
  pool_->insert(1, 2.5);
  pool_->insert(2, 1.5);
  pool_->insert(3, 3.0);
  pool_->insert(4, 0.5);
  pool_->insert(5, 4.0);

  EXPECT_EQ(pool_->size(), 5);  // Check the current size

  // Pop elements and check
  EXPECT_EQ(pool_->pop(), 4);  // ID with the smallest distance

  pool_->insert(6, 2.0);  // Insert a new element

  // Pop all elements and check the order
  EXPECT_EQ(pool_->pop(), 2);
  EXPECT_EQ(pool_->pop(), 6);
  EXPECT_EQ(pool_->pop(), 1);
  EXPECT_EQ(pool_->pop(), 3);
  EXPECT_EQ(pool_->pop(), 5);
  EXPECT_EQ(pool_->has_next(), false);  // Finally should be empty
}

TEST_F(LinearPoolTest, BoundaryConditionsTest) {
  // Fill the pool
  pool_->insert(1, 2.5);
  pool_->insert(2, 1.5);
  pool_->insert(3, 3.0);
  pool_->insert(4, 0.5);
  pool_->insert(5, 4.0);

  // Try to insert an element exceeding capacity
  EXPECT_FALSE(pool_->insert(6, 5.0));  // Should return false
  EXPECT_EQ(pool_->size(), 5);          // Size should still be 5

  // Try to insert a negative value
  EXPECT_TRUE(pool_->insert(7, -1.0));  // Should successfully insert
  EXPECT_EQ(pool_->size(), 5);          // Size should increase
}

// Performance test
TEST_F(LinearPoolTest, PerformanceTest) {
  const int kNumElements = 10000;
  for (int i = 0; i < kNumElements; ++i) {
    pool_->insert(i, static_cast<float>(kNumElements - i));  // Insert elements
  }
  EXPECT_EQ(pool_->size(), std::min(kNumElements, 5));  // Check size
}

// Concurrent test
TEST_F(LinearPoolTest, ConcurrentInsertTest) {
  const int kNumThreads = 10;
  const int kInsertsPerThread = 100;

  // Lambda function for inserting elements
  auto insert_function = [this](int thread_id) -> void {
    for (int i = 0; i < kInsertsPerThread; ++i) {
      pool_->insert(thread_id * kInsertsPerThread + i, static_cast<float>(i));
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(insert_function, i);  // Create threads for insertion
  }

  for (auto &thread : threads) {
    thread.join();  // Wait for all threads to finish
  }

  // Check final size
  EXPECT_LE(pool_->size(), 5);  // Size should be less than or equal to capacity
}

}  // namespace alaya
