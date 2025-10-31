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

#include <gtest/gtest.h>
#include <cstring>  // for memcmp
#include <fstream>
#include <vector>

#include "storage/static_storage.hpp"

namespace alaya {
// NOLINTBEGIN
//  Helper to create a temporary file name
auto GetTempFileName() -> std::string {
  std::string name = std::tmpnam(nullptr);
  if (name.empty()) name = "temp_static_storage_test.bin";
  return name;
}

TEST(StaticStorageTest, BasicConstructionAndAccess) {
  const size_t item_cnt = 10;
  const size_t data_chunk_size = 5;
  const size_t total_size = item_cnt * data_chunk_size;

  alaya::StaticStorage<> storage(std::vector<size_t>{item_cnt, data_chunk_size});

  // Check size and bytes
  EXPECT_EQ(storage.size(), total_size);
  EXPECT_EQ(storage.bytes(), total_size * sizeof(char));

  // Initialize data
  for (size_t i = 0; i < total_size; ++i) {
    storage[i] = static_cast<char>(i % 256);
  }

  // Verify data via at() and operator[]
  for (size_t i = 0; i < total_size; ++i) {
    EXPECT_EQ(storage.at(i), static_cast<char>(i % 256));
    EXPECT_EQ(storage[i], static_cast<char>(i % 256));
  }
}

TEST(StaticStorageTest, MoveConstructor) {
  const size_t item_cnt = 3;
  const size_t data_chunk_size = 4;
  const size_t total_size = item_cnt * data_chunk_size;

  alaya::StaticStorage<> original(std::vector<size_t>{item_cnt, data_chunk_size});
  for (size_t i = 0; i < total_size; ++i) {
    original[i] = static_cast<char>(i + 10);
  }

  // Move construct
  alaya::StaticStorage<> moved = std::move(original);

  // Original should be in valid but unspecified state (pointer null)
  EXPECT_EQ(original.data(), nullptr);

  // Moved should have the data
  EXPECT_NE(moved.data(), nullptr);
  for (size_t i = 0; i < total_size; ++i) {
    EXPECT_EQ(moved[i], static_cast<char>(i + 10));
  }
}

TEST(StaticStorageTest, MoveAssignment) {
  const size_t item_cnt = 2;
  const size_t data_chunk_size = 3;
  const size_t total_size = item_cnt * data_chunk_size;

  alaya::StaticStorage<> a(std::vector<size_t>{item_cnt, data_chunk_size});
  for (size_t i = 0; i < total_size; ++i) {
    a[i] = static_cast<char>(i * 2);
  }

  alaya::StaticStorage<> b;  // default constructed (null)

  b = std::move(a);

  EXPECT_EQ(a.data(), nullptr);  // a is now empty
  for (size_t i = 0; i < total_size; ++i) {
    EXPECT_EQ(b[i], static_cast<char>(i * 2));
  }
}

TEST(StaticStorageTest, SaveAndLoad) {
  const size_t item_cnt = 4;
  const size_t data_chunk_size = 6;
  const size_t total_size = item_cnt * data_chunk_size;

  alaya::StaticStorage<> storage(std::vector<size_t>{item_cnt, data_chunk_size});
  for (size_t i = 0; i < total_size; ++i) {
    storage[i] = static_cast<char>(0xFF - i);
  }

  std::string filename = GetTempFileName();
  {
    std::ofstream out(filename, std::ios::binary);
    storage.save(out);
  }

  alaya::StaticStorage<> loaded(std::vector<size_t>{item_cnt, data_chunk_size});
  {
    std::ifstream in(filename, std::ios::binary);
    loaded.load(in);
  }

  // Compare raw memory
  EXPECT_EQ(0, std::memcmp(storage.data(), loaded.data(), storage.bytes()));

  // Clean up
  std::remove(filename.c_str());
}

TEST(StaticStorageTest, AllocatorThrowsOnExcessiveSize) {
  using T = float;  // sizeof = 4
  using Alloc = AlignedAllocator<T, 64>;
  using Storage = StaticStorage<T, std::vector<size_t>, Alloc>;

  size_t max_n = std::numeric_limits<size_t>::max() / sizeof(T);
  std::vector<size_t> dims = {max_n + 1};

  EXPECT_THROW(Storage s(dims), std::bad_array_new_length);
}

TEST(StaticStorageTest, MoveAssignmentCallsDestroy) {
  using Storage = StaticStorage<char>;

  // Create 'a' with valid allocation
  std::vector<size_t> dims1 = {10, 8};  // 80 bytes
  Storage a(dims1);

  // Verify a has non-null pointer
  ASSERT_NE(a.data(), nullptr);

  // Create 'b' with another allocation
  std::vector<size_t> dims2 = {5, 16};  // 80 bytes
  Storage b(dims2);
  ASSERT_NE(b.data(), nullptr);

  // Move assign b to a → this triggers destroy() on a's old memory
  a = std::move(b);

  // After move:
  // - a should now hold b's data
  // - b's pointer should be null
  EXPECT_NE(a.data(), nullptr);
  EXPECT_EQ(b.data(), nullptr);
}

// NOLINTEND
}  // namespace alaya
