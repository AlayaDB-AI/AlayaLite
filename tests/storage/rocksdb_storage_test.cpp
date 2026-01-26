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
#include <rocksdb/db.h>
#include <filesystem>
#include <string>
#include <vector>

#include "storage/rocksdb_storage.hpp"

namespace alaya {
// NOLINTBEGIN
namespace fs = std::filesystem;

// A simple trivially copyable struct for testing
struct TestData {
  int value;
  float score;
};

static_assert(std::is_trivially_copyable_v<TestData>, "TestData must be trivially copyable");

class RocksDBStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string test_name = std::string(test_info->test_suite_name()) + "_" + test_info->name();

    // Replace any problematic characters (e.g., '/', spaces, etc.) if needed
    std::replace(test_name.begin(), test_name.end(), '/', '_');
    std::replace(test_name.begin(), test_name.end(), ' ', '_');

    temp_dir_ = fs::temp_directory_path() / ("rocksdb_test_" + test_name);
    fs::create_directories(temp_dir_);
    config_.db_path_ = temp_dir_.string();
  }

  void TearDown() override {
    // Clean up the temporary database directory
    if (fs::exists(temp_dir_)) {
      fs::remove_all(temp_dir_);
    }
  }

  RocksDBConfig config_;
  fs::path temp_dir_;
};

// ============================================================================
// Basic Operations Tests
// ============================================================================

TEST_F(RocksDBStorageTest, BasicInsertAndGet) {
  RocksDBStorage<TestData> storage(config_);

  TestData data{42, 3.14f};
  auto id = storage.insert(data);
  EXPECT_NE(id, static_cast<uint32_t>(-1));
  EXPECT_EQ(storage.count(), 1U);
  EXPECT_EQ(storage.next_id(), 1U);

  auto retrieved = storage[id];
  EXPECT_EQ(retrieved.value, 42);
  EXPECT_FLOAT_EQ(retrieved.score, 3.14f);
}

TEST_F(RocksDBStorageTest, InvalidIDReturnsDefault) {
  RocksDBStorage<TestData> storage(config_);
  auto invalid_data = storage[999];
  EXPECT_EQ(invalid_data.value, 0);
  EXPECT_FLOAT_EQ(invalid_data.score, 0.0f);
}

TEST_F(RocksDBStorageTest, IsValidWorksCorrectly) {
  RocksDBStorage<TestData> storage(config_);
  TestData data{100, 2.5f};
  auto id = storage.insert(data);
  EXPECT_TRUE(storage.is_valid(id));
  EXPECT_FALSE(storage.is_valid(id + 1));
  EXPECT_FALSE(storage.is_valid(999));
}

TEST_F(RocksDBStorageTest, UpdateOperations) {
  RocksDBStorage<TestData> storage(config_);

  // Test valid update
  TestData data{10, 1.0f};
  auto id = storage.insert(data);
  EXPECT_EQ(storage[id].value, 10);

  TestData updated{20, 2.0f};
  auto result_id = storage.update(id, updated);
  EXPECT_EQ(result_id, id);
  auto retrieved = storage[id];
  EXPECT_EQ(retrieved.value, 20);
  EXPECT_FLOAT_EQ(retrieved.score, 2.0f);

  // Test invalid update
  TestData invalid_data{30, 3.0f};
  auto invalid_result = storage.update(999, invalid_data);
  EXPECT_EQ(invalid_result, static_cast<uint32_t>(-1));
}

TEST_F(RocksDBStorageTest, RemoveOperations) {
  RocksDBStorage<TestData> storage(config_);

  // Test removing invalid ID
  auto invalid_removed = storage.remove(999);
  EXPECT_EQ(invalid_removed, static_cast<uint32_t>(-1));

  // Test removing valid entries
  auto id1 = storage.insert({10, 1.0f});
  auto id2 = storage.insert({20, 2.0f});
  auto id3 = storage.insert({30, 3.0f});
  EXPECT_EQ(storage.count(), 3U);

  // Remove them one by one
  auto result1 = storage.remove(id1);
  EXPECT_EQ(result1, id1);
  EXPECT_EQ(storage.count(), 2U);
  EXPECT_FALSE(storage.is_valid(id1));

  auto result2 = storage.remove(id2);
  EXPECT_EQ(result2, id2);
  EXPECT_EQ(storage.count(), 1U);
  EXPECT_FALSE(storage.is_valid(id2));

  auto result3 = storage.remove(id3);
  EXPECT_EQ(result3, id3);
  EXPECT_EQ(storage.count(), 0U);
  EXPECT_FALSE(storage.is_valid(id3));
}

// ============================================================================
// Batch Operations Tests
// ============================================================================

TEST_F(RocksDBStorageTest, BatchInsert) {
  RocksDBStorage<TestData> storage(config_);
  std::vector<TestData> inputs = {{1, 1.1f}, {2, 2.2f}, {3, 3.3f}};

  bool success = storage.batch_insert(inputs.begin(), inputs.end());

  EXPECT_TRUE(success);
  EXPECT_EQ(storage.count(), 3U);
  EXPECT_EQ(storage.next_id(), 3U);

  // Verify data by ID (IDs start from 0)
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto data = storage[i];
    EXPECT_EQ(data.value, inputs[i].value);
    EXPECT_FLOAT_EQ(data.score, inputs[i].score);
  }
}

TEST_F(RocksDBStorageTest, EmptyBatchInsert) {
  RocksDBStorage<TestData> storage(config_);
  std::vector<TestData> empty_inputs;

  bool success = storage.batch_insert(empty_inputs.begin(), empty_inputs.end());

  EXPECT_TRUE(success);  // Empty batch is still successful
  EXPECT_EQ(storage.count(), 0U);
}

// ============================================================================
// Persistence Tests
// ============================================================================

TEST_F(RocksDBStorageTest, PersistenceAcrossInstances) {
  // Test both metadata and compressed data persistence
  {
    RocksDBStorage<TestData> storage1(config_);

    // Insert various data patterns
    storage1.insert({100, 10.0f});
    storage1.insert({200, 20.0f});

    // Add compressible data
    for (int i = 0; i < 50; ++i) {
      storage1.insert({i, static_cast<float>(i) * 1.5f});
    }

    storage1.flush();
    // Destructor will save metadata and close DB
  }

  {
    // Reopen the same DB - verify both metadata and data integrity
    RocksDBStorage<TestData> storage2(config_);
    EXPECT_EQ(storage2.count(), 52U);
    EXPECT_EQ(storage2.next_id(), 52U);

    // Verify original data
    auto data0 = storage2[0];
    auto data1 = storage2[1];
    EXPECT_EQ(data0.value, 100);
    EXPECT_EQ(data1.value, 200);

    // Verify compressed data integrity
    for (uint32_t i = 2; i < 52; ++i) {
      auto data = storage2[i];
      EXPECT_EQ(data.value, static_cast<int>(i - 2));
      EXPECT_FLOAT_EQ(data.score, static_cast<float>(i - 2) * 1.5f);
    }
  }
}

TEST_F(RocksDBStorageTest, SaveCheckpointAndRestore) {
  fs::path checkpoint_path = temp_dir_ / "checkpoint";

  {
    RocksDBStorage<TestData> storage(config_);
    storage.insert({111, 1.11f});
    storage.insert({222, 2.22f});
    storage.save(checkpoint_path.string());
  }

  // Verify that the checkpoint was created
  EXPECT_TRUE(fs::exists(checkpoint_path / "CURRENT"));

  // Restore from checkpoint to a new directory
  fs::path restored_path = temp_dir_ / "restored";
  fs::copy(checkpoint_path, restored_path, fs::copy_options::recursive);

  // Open the restored DB
  RocksDBConfig restore_config;
  restore_config.db_path_ = restored_path.string();
  RocksDBStorage<TestData> restored_storage(restore_config);

  EXPECT_EQ(restored_storage.count(), 2U);
  EXPECT_EQ(restored_storage.next_id(), 2U);

  auto d0 = restored_storage[0];
  auto d1 = restored_storage[1];
  EXPECT_EQ(d0.value, 111);
  EXPECT_EQ(d1.value, 222);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(RocksDBStorageTest, DatabaseOpenFailure) {
  // Test opening database with invalid configuration
  RocksDBConfig bad_config;
  bad_config.db_path_ = "/invalid/nonexistent/path/to/db";
  bad_config.create_if_missing_ = false;

  // Should throw exception when database cannot be opened
  EXPECT_THROW({ RocksDBStorage<TestData> storage(bad_config); }, std::runtime_error);
}

TEST_F(RocksDBStorageTest, SaveCheckpointToInvalidPath) {
  RocksDBStorage<TestData> storage(config_);
  storage.insert({333, 3.33f});

  // Try to save checkpoint to an invalid path (parent directory doesn't exist)
  fs::path invalid_checkpoint_path = temp_dir_ / "nonexistent_parent/checkpoint";

  // This should log an error but not throw
  EXPECT_NO_THROW(storage.save(invalid_checkpoint_path.string()));

  // Verify the invalid checkpoint was not created
  EXPECT_FALSE(fs::exists(invalid_checkpoint_path));
}

TEST_F(RocksDBStorageTest, DataCorruptionThrowsException) {
  // Insert data normally
  {
    RocksDBStorage<TestData> storage(config_);
    storage.insert({123, 4.56f});
  }

  // Corrupt the data by writing wrong size using raw RocksDB API
  {
    rocksdb::DB *db;
    rocksdb::Options options;
    rocksdb::Status status = rocksdb::DB::Open(options, config_.db_path_, &db);
    ASSERT_TRUE(status.ok());

    // Write corrupted data with wrong size (only 4 bytes instead of sizeof(TestData))
    std::string corrupted_data = "XXXX";
    status = db->Put(rocksdb::WriteOptions(), "data_0", corrupted_data);
    ASSERT_TRUE(status.ok());

    // Close before delete to release lock properly
    status = db->Close();
    ASSERT_TRUE(status.ok());
    delete db;
  }

  // Try to read corrupted data - should throw exception
  {
    RocksDBStorage<TestData> storage(config_);
    EXPECT_THROW({ [[maybe_unused]] auto data = storage[0]; }, std::runtime_error);
  }
}

// ============================================================================
// Configuration and Accessors Tests
// ============================================================================

TEST_F(RocksDBStorageTest, ConfigAndAccessors) {
  RocksDBStorage<TestData> storage(config_);

  // Test get_db_path method
  EXPECT_EQ(storage.get_db_path(), config_.db_path_);

  // Test config() method
  const auto &retrieved_config = storage.config();
  EXPECT_EQ(retrieved_config.db_path_, config_.db_path_);
  EXPECT_EQ(retrieved_config.write_buffer_size_, config_.write_buffer_size_);
  EXPECT_EQ(retrieved_config.max_write_buffer_number_, config_.max_write_buffer_number_);

  // Test get_statistics (just verify it doesn't crash)
  storage.insert({777, 7.77f});
  EXPECT_NO_THROW(storage.get_statistics());
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(RocksDBStorageTest, MoveSemantics) {
  // Test move constructor
  {
    RocksDBStorage<TestData> storage1(config_);
    storage1.insert({999, 9.99f});

    RocksDBStorage<TestData> storage2(std::move(storage1));

    EXPECT_EQ(storage2.count(), 1U);
    auto data = storage2[0];
    EXPECT_EQ(data.value, 999);
    EXPECT_FLOAT_EQ(data.score, 9.99f);
  }

  // Test move assignment
  {
    RocksDBConfig config1 = config_;
    config1.db_path_ = (temp_dir_ / "db1").string();
    fs::create_directories(config1.db_path_);

    RocksDBConfig config2 = config_;
    config2.db_path_ = (temp_dir_ / "db2").string();
    fs::create_directories(config2.db_path_);

    RocksDBStorage<TestData> storage1(config1);
    storage1.insert({111, 1.11f});

    RocksDBStorage<TestData> storage2(config2);
    storage2.insert({222, 2.22f});

    storage2 = std::move(storage1);

    EXPECT_EQ(storage2.count(), 1U);
    auto data = storage2[0];
    EXPECT_EQ(data.value, 111);
    EXPECT_FLOAT_EQ(data.score, 1.11f);
  }
}

// ============================================================================
// Compression Tests
// ============================================================================

TEST_F(RocksDBStorageTest, CompressionDataIntegrity) {
  RocksDBStorage<TestData> storage(config_);

  // Insert data with patterns that compress well (repetitive values)
  std::vector<TestData> compressible_data;
  for (int i = 0; i < 500; ++i) {
    // Repetitive pattern - should compress well with LZ4/ZSTD
    compressible_data.push_back({i % 10, static_cast<float>(i % 10) * 1.1f});
  }

  // Batch insert compressible data
  bool success = storage.batch_insert(compressible_data.begin(), compressible_data.end());
  EXPECT_TRUE(success);
  EXPECT_EQ(storage.count(), 500U);

  // Force flush to disk (data will be compressed)
  storage.flush();

  // Verify data integrity after compression/decompression (IDs start from 0)
  for (size_t i = 0; i < compressible_data.size(); ++i) {
    auto data = storage[i];
    EXPECT_EQ(data.value, static_cast<int>(i % 10));
    EXPECT_FLOAT_EQ(data.score, static_cast<float>(i % 10) * 1.1f);
  }
}

TEST_F(RocksDBStorageTest, CompressionWithCompaction) {
  RocksDBStorage<TestData> storage(config_);

  // Insert large amount of data to trigger compaction
  // This will move data to bottommost level where ZSTD compression is used
  // Also tests flush and large batch operations
  for (int batch = 0; batch < 5; ++batch) {
    std::vector<TestData> batch_data;
    for (int i = 0; i < 200; ++i) {
      int value = batch * 200 + i;
      batch_data.push_back({value, static_cast<float>(value) * 0.5f});
    }
    storage.batch_insert(batch_data.begin(), batch_data.end());
    storage.flush();
  }

  EXPECT_EQ(storage.count(), 1000U);

  // Trigger compaction (data will be recompressed with ZSTD for bottommost level)
  storage.compact();

  // Verify all data is still intact after compaction and recompression
  for (uint32_t id = 0; id < 1000; ++id) {
    auto data = storage[id];
    EXPECT_EQ(data.value, static_cast<int>(id));
    EXPECT_FLOAT_EQ(data.score, static_cast<float>(id) * 0.5f);
  }
}

TEST_F(RocksDBStorageTest, MixedCompressibleData) {
  RocksDBStorage<TestData> storage(config_);

  // Insert mix of compressible and incompressible data
  std::vector<TestData> mixed_data;

  // Compressible: repetitive patterns
  for (int i = 0; i < 100; ++i) {
    mixed_data.push_back({42, 3.14f});  // Same value repeated
  }

  // Less compressible: unique values
  for (int i = 0; i < 100; ++i) {
    mixed_data.push_back({i * 7919, static_cast<float>(i) * 3.14159f});
  }

  bool success = storage.batch_insert(mixed_data.begin(), mixed_data.end());
  EXPECT_TRUE(success);
  EXPECT_EQ(storage.count(), 200U);

  storage.flush();
  storage.compact();

  // Verify all data integrity regardless of compressibility (IDs start from 0)
  for (size_t i = 0; i < 100; ++i) {
    auto data = storage[i];
    EXPECT_EQ(data.value, 42);
    EXPECT_FLOAT_EQ(data.score, 3.14f);
  }

  for (size_t i = 100; i < 200; ++i) {
    auto data = storage[i];
    int expected_value = static_cast<int>(i - 100) * 7919;
    float expected_score = static_cast<float>(i - 100) * 3.14159f;
    EXPECT_EQ(data.value, expected_value);
    EXPECT_FLOAT_EQ(data.score, expected_score);
  }
}

// NOLINTEND
}  // namespace alaya
