// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "utils/query_utils.hpp"
#include "utils/rabitq_utils/search_utils/hashset.hpp"
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

TEST(EpochVisitedSetTest, ClearResetsLogicalState) {
  EpochVisitedSet<> visited(8);

  visited.set(2);
  visited.set(5);
  EXPECT_TRUE(visited.get(2));
  EXPECT_TRUE(visited.get(5));

  visited.clear();

  EXPECT_FALSE(visited.get(2));
  EXPECT_FALSE(visited.get(5));

  visited.set(3);
  EXPECT_TRUE(visited.get(3));
  EXPECT_FALSE(visited.get(2));
}

TEST(EpochVisitedSetTest, ResizeClearsExistingState) {
  EpochVisitedSet<> visited(4);

  visited.set(1);
  visited.resize(10);

  EXPECT_EQ(visited.size(), 10U);
  for (uint32_t i = 0; i < 10; ++i) {
    EXPECT_FALSE(visited.get(i));
  }
}

TEST(HashBasedBooleanSetTest, HandlesCollisionsAndClear) {
  HashBasedBooleanSet visited;
  visited.initialize(8);

  visited.set(1);
  visited.set(9);
  visited.set(17);

  EXPECT_TRUE(visited.get(1));
  EXPECT_TRUE(visited.get(9));
  EXPECT_TRUE(visited.get(17));
  EXPECT_FALSE(visited.get(2));

  visited.clear();

  EXPECT_FALSE(visited.get(1));
  EXPECT_FALSE(visited.get(9));
  EXPECT_FALSE(visited.get(17));
}

}  // namespace alaya
