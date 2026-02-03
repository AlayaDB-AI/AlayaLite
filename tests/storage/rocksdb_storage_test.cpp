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
#include "utils/scalar_data.hpp"

namespace alaya {
// NOLINTBEGIN
namespace fs = std::filesystem;

class RocksDBStorageTest : public ::testing::Test {
 protected:
    void SetUp() override {
        const ::testing::TestInfo *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string test_name = std::string(test_info->test_suite_name()) + "_" + test_info->name();

        std::replace(test_name.begin(), test_name.end(), '/', '_');
        std::replace(test_name.begin(), test_name.end(), ' ', '_');

        temp_dir_ = fs::temp_directory_path() / ("rocksdb_test_" + test_name);
        fs::create_directories(temp_dir_);
        config_.db_path_ = temp_dir_.string();
    }

    void TearDown() override {
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
    RocksDBStorage<> storage(config_);

    ScalarData data{"id_001", "document content", {{"category", "tech"}, {"score", int64_t(95)}}};
    bool success = storage.insert(0, data);
    EXPECT_TRUE(success);
    EXPECT_EQ(storage.count(), 1U);

    auto retrieved = storage[0];
    EXPECT_EQ(retrieved.item_id, "id_001");
    EXPECT_EQ(retrieved.document, "document content");
    EXPECT_EQ(std::get<std::string>(retrieved.metadata.at("category")), "tech");
    EXPECT_EQ(std::get<int64_t>(retrieved.metadata.at("score")), 95);
}

TEST_F(RocksDBStorageTest, InvalidIDReturnsEmpty) {
    RocksDBStorage<> storage(config_);
    auto invalid_data = storage[999];
    EXPECT_TRUE(invalid_data.item_id.empty());
    EXPECT_TRUE(invalid_data.document.empty());
    EXPECT_TRUE(invalid_data.metadata.empty());
}

TEST_F(RocksDBStorageTest, IsValidWorksCorrectly) {
    RocksDBStorage<> storage(config_);
    ScalarData data{"test_id", "test doc", {}};
    storage.insert(0, data);
    EXPECT_TRUE(storage.is_valid(0));
    EXPECT_FALSE(storage.is_valid(1));
    EXPECT_FALSE(storage.is_valid(999));
}

TEST_F(RocksDBStorageTest, UpdateOperations) {
    RocksDBStorage<> storage(config_);

    ScalarData data1{"id_001", "original", {{"version", int64_t(1)}}};
    storage.insert(0, data1);
    EXPECT_EQ(storage[0].document, "original");

    ScalarData data2{"id_001", "updated", {{"version", int64_t(2)}}};
    bool success = storage.update(0, data2);
    EXPECT_TRUE(success);

    auto retrieved = storage[0];
    EXPECT_EQ(retrieved.document, "updated");
    EXPECT_EQ(std::get<int64_t>(retrieved.metadata.at("version")), 2);
}

TEST_F(RocksDBStorageTest, RemoveOperations) {
    RocksDBStorage<> storage(config_);

    bool invalid_removed = storage.remove(999);
    EXPECT_FALSE(invalid_removed);

    ScalarData data1{"id_001", "doc1", {}};
    ScalarData data2{"id_002", "doc2", {}};
    ScalarData data3{"id_003", "doc3", {}};

    storage.insert(0, data1);
    storage.insert(1, data2);
    storage.insert(2, data3);

    EXPECT_EQ(storage.count(), 3U);

    bool result0 = storage.remove(0);
    EXPECT_TRUE(result0);
    EXPECT_EQ(storage.count(), 2U);
    EXPECT_FALSE(storage.is_valid(0));

    bool result1 = storage.remove(1);
    EXPECT_TRUE(result1);
    EXPECT_EQ(storage.count(), 1U);
    EXPECT_FALSE(storage.is_valid(1));

    bool result2 = storage.remove(2);
    EXPECT_TRUE(result2);
    EXPECT_EQ(storage.count(), 0U);
    EXPECT_FALSE(storage.is_valid(2));
}

// ============================================================================
// Batch Operations Tests
// ============================================================================

TEST_F(RocksDBStorageTest, BatchInsert) {
    RocksDBStorage<> storage(config_);

    std::vector<ScalarData> inputs = {
        {"id_001", "doc1", {{"score", int64_t(100)}}},
        {"id_002", "doc2", {{"score", int64_t(200)}}},
        {"id_003", "doc3", {{"score", int64_t(300)}}},
    };

    bool success = storage.batch_insert(0, inputs.begin(), inputs.end());
    EXPECT_TRUE(success);
    EXPECT_EQ(storage.count(), 3U);

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto data = storage[i];
        EXPECT_EQ(data.item_id, inputs[i].item_id);
        EXPECT_EQ(data.document, inputs[i].document);
    }
}

TEST_F(RocksDBStorageTest, EmptyBatchInsert) {
    RocksDBStorage<> storage(config_);
    std::vector<ScalarData> empty_inputs;

    bool success = storage.batch_insert(0, empty_inputs.begin(), empty_inputs.end());
    EXPECT_TRUE(success);
    EXPECT_EQ(storage.count(), 0U);
}

TEST_F(RocksDBStorageTest, BatchInsertWithOffset) {
    RocksDBStorage<> storage(config_);

    std::vector<ScalarData> batch1 = {
        {"id_001", "doc1", {}},
        {"id_002", "doc2", {}},
    };

    std::vector<ScalarData> batch2 = {
        {"id_003", "doc3", {}},
        {"id_004", "doc4", {}},
    };

    storage.batch_insert(0, batch1.begin(), batch1.end());
    storage.batch_insert(2, batch2.begin(), batch2.end());

    EXPECT_EQ(storage.count(), 4U);
    EXPECT_EQ(storage[0].item_id, "id_001");
    EXPECT_EQ(storage[1].item_id, "id_002");
    EXPECT_EQ(storage[2].item_id, "id_003");
    EXPECT_EQ(storage[3].item_id, "id_004");
}

// ============================================================================
// Metadata Tests
// ============================================================================

TEST_F(RocksDBStorageTest, ComplexMetadata) {
    RocksDBStorage<> storage(config_);

    MetadataMap meta;
    meta["int_field"] = int64_t(42);
    meta["float_field"] = 3.14;
    meta["string_field"] = std::string("hello");
    meta["bool_field"] = true;

    ScalarData data{"item_1", "content", meta};
    storage.insert(0, data);

    auto retrieved = storage[0];
    EXPECT_EQ(std::get<int64_t>(retrieved.metadata.at("int_field")), 42);
    EXPECT_FLOAT_EQ(std::get<double>(retrieved.metadata.at("float_field")), 3.14);
    EXPECT_EQ(std::get<std::string>(retrieved.metadata.at("string_field")), "hello");
    EXPECT_EQ(std::get<bool>(retrieved.metadata.at("bool_field")), true);
}

TEST_F(RocksDBStorageTest, FindByItemId) {
    RocksDBStorage<> storage(config_);

    storage.insert(0, ScalarData{"user_001", "doc1", {}});
    storage.insert(1, ScalarData{"user_002", "doc2", {}});
    storage.insert(2, ScalarData{"user_003", "doc3", {}});

    auto id1 = storage.find_by_item_id("user_001");
    EXPECT_TRUE(id1.has_value());
    EXPECT_EQ(id1.value(), 0U);

    auto id2 = storage.find_by_item_id("user_002");
    EXPECT_TRUE(id2.has_value());
    EXPECT_EQ(id2.value(), 1U);

    auto id_notfound = storage.find_by_item_id("nonexistent");
    EXPECT_FALSE(id_notfound.has_value());
}

// ============================================================================
// Batch Get Tests
// ============================================================================

TEST_F(RocksDBStorageTest, BatchGet) {
    RocksDBStorage<> storage(config_);

    std::vector<ScalarData> inputs = {
        {"id_001", "doc1", {}},
        {"id_002", "doc2", {}},
        {"id_003", "doc3", {}},
    };

    storage.batch_insert(0, inputs.begin(), inputs.end());

    std::vector<uint32_t> ids = {0, 1, 2};
    auto results = storage.batch_get(ids);

    EXPECT_EQ(results.size(), 3U);
    EXPECT_EQ(results[0].item_id, "id_001");
    EXPECT_EQ(results[1].item_id, "id_002");
    EXPECT_EQ(results[2].item_id, "id_003");
}

// ============================================================================
// Persistence Tests
// ============================================================================

TEST_F(RocksDBStorageTest, PersistenceAcrossInstances) {
    {
        RocksDBStorage<> storage(config_);
        storage.insert(0, ScalarData{"id_001", "doc1", {{"key", std::string("value1")}}});
        storage.insert(1, ScalarData{"id_002", "doc2", {{"key", std::string("value2")}}});
        storage.flush();
    }

    {
        RocksDBStorage<> storage(config_);
        EXPECT_EQ(storage.count(), 2U);
        EXPECT_EQ(storage[0].item_id, "id_001");
        EXPECT_EQ(storage[1].item_id, "id_002");
    }
}

TEST_F(RocksDBStorageTest, SaveCheckpointAndRestore) {
    fs::path checkpoint_path = temp_dir_ / "checkpoint";

    {
        RocksDBStorage<> storage(config_);
        storage.insert(0, ScalarData{"item_1", "data1", {}});
        storage.insert(1, ScalarData{"item_2", "data2", {}});
        storage.save(checkpoint_path.string());
    }

    EXPECT_TRUE(fs::exists(checkpoint_path / "CURRENT"));

    fs::path restored_path = temp_dir_ / "restored";
    fs::copy(checkpoint_path, restored_path, fs::copy_options::recursive);

    RocksDBConfig restore_config;
    restore_config.db_path_ = restored_path.string();
    RocksDBStorage<> restored_storage(restore_config);

    EXPECT_EQ(restored_storage.count(), 2U);
    EXPECT_EQ(restored_storage[0].item_id, "item_1");
    EXPECT_EQ(restored_storage[1].item_id, "item_2");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(RocksDBStorageTest, DatabaseOpenFailure) {
    RocksDBConfig bad_config;
    bad_config.db_path_ = "/invalid/nonexistent/path/to/db";
    bad_config.create_if_missing_ = false;

    EXPECT_THROW({ RocksDBStorage<> storage(bad_config); }, std::runtime_error);
}

TEST_F(RocksDBStorageTest, SaveCheckpointToInvalidPath) {
    RocksDBStorage<> storage(config_);
    storage.insert(0, ScalarData{"test", "data", {}});

    fs::path invalid_checkpoint_path = temp_dir_ / "nonexistent_parent/checkpoint";

    EXPECT_NO_THROW(storage.save(invalid_checkpoint_path.string()));
    EXPECT_FALSE(fs::exists(invalid_checkpoint_path));
}

// ============================================================================
// Configuration and Accessors Tests
// ============================================================================

TEST_F(RocksDBStorageTest, ConfigAndAccessors) {
    RocksDBStorage<> storage(config_);

    EXPECT_EQ(storage.get_db_path(), config_.db_path_);

    const auto &retrieved_config = storage.config();
    EXPECT_EQ(retrieved_config.db_path_, config_.db_path_);
    EXPECT_EQ(retrieved_config.write_buffer_size_, config_.write_buffer_size_);

    storage.insert(0, ScalarData{"test", "data", {}});
    EXPECT_NO_THROW(storage.get_statistics());
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(RocksDBStorageTest, MoveSemantics) {
    {
        RocksDBStorage<> storage1(config_);
        storage1.insert(0, ScalarData{"test", "data", {}});

        RocksDBStorage<> storage2(std::move(storage1));
        EXPECT_EQ(storage2.count(), 1U);
        EXPECT_EQ(storage2[0].item_id, "test");
    }

    {
        RocksDBConfig config1 = config_;
        config1.db_path_ = (temp_dir_ / "db1").string();
        fs::create_directories(config1.db_path_);

        RocksDBConfig config2 = config_;
        config2.db_path_ = (temp_dir_ / "db2").string();
        fs::create_directories(config2.db_path_);

        RocksDBStorage<> storage1(config1);
        storage1.insert(0, ScalarData{"item1", "doc1", {}});

        RocksDBStorage<> storage2(config2);
        storage2.insert(0, ScalarData{"item2", "doc2", {}});

        storage2 = std::move(storage1);
        EXPECT_EQ(storage2.count(), 1U);
        EXPECT_EQ(storage2[0].item_id, "item1");
    }
}

// NOLINTEND
}  // namespace alaya
