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

#include "space/sq4_space.hpp"
#include <gtest/gtest.h>
#include <sys/types.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include "utils/log.hpp"

namespace alaya {

class SQ4SpaceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dim_ = 4;
    capacity_ = 10;
    metric_ = MetricType::L2;
    space_ = std::make_shared<SQ4Space<>>(capacity_, dim_, metric_);
    file_name_ = "test_sq4_space.bin";
    if (std::filesystem::exists(file_name_)) {
      std::filesystem::remove(file_name_);
    }
  }

  void TearDown() override {
    if (std::filesystem::exists(file_name_)) {
      std::filesystem::remove(file_name_);
    }
  }

  std::shared_ptr<SQ4Space<>> space_;
  std::string file_name_;
  size_t dim_;
  uint32_t capacity_;
  MetricType metric_;
};

TEST_F(SQ4SpaceTest, Initialization) {
  EXPECT_EQ(space_->get_dim(), 4);
  EXPECT_EQ(space_->get_data_num(), 0);
  EXPECT_EQ(space_->get_data_size(), 2);
}

TEST_F(SQ4SpaceTest, FitData) {
  float data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  space_->fit(data, 2);
  EXPECT_EQ(space_->get_data_num(), 2);
}

TEST_F(SQ4SpaceTest, InsertAndRemove) {
  float vec[4] = {1.0, 2.0, 3.0, 4.0};
  uint32_t id = space_->insert(vec);
  EXPECT_GE(id, 0);
  EXPECT_EQ(space_->get_data_num(), 1);

  space_->remove(id);
  EXPECT_EQ(space_->get_data_num(), 1);  // remove() only marks it as deleted
}

TEST_F(SQ4SpaceTest, GetDistance) {
  float data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  space_->fit(data, 2);
  float dist = space_->get_distance(0, 1);
  EXPECT_FLOAT_EQ(dist, 64);
}

TEST_F(SQ4SpaceTest, SaveAndLoad) {
  float data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  space_->fit(reinterpret_cast<float *>(data), 2);
  std::string_view file_name_view = file_name_;

  space_->save(file_name_view);

  SQ4Space<> new_space;
  new_space.load(file_name_view);
  EXPECT_EQ(new_space.get_data_num(), 2);
}

TEST_F(SQ4SpaceTest, QueryComputerWithQuery) {
  float data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  space_->fit(reinterpret_cast<float *>(data), 2);
  float query[4] = {1.0, 2.0, 3.0, 4.0};
  auto query_computer = space_->get_query_computer(query);
  EXPECT_GE(query_computer(1), 64);
}

TEST_F(SQ4SpaceTest, QueryComputerWithId) {
  float data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  space_->fit(reinterpret_cast<float *>(data), 2);
  uint32_t id = 0;
  auto query_computer = space_->get_query_computer(id);
  EXPECT_GE(query_computer(1), 64);
}

TEST_F(SQ4SpaceTest, PrefetchById) {
  float data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  space_->fit(reinterpret_cast<float *>(data), 2);
  space_->prefetch_by_id(1);
}

TEST_F(SQ4SpaceTest, PrefetchByAddress) {
  float data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  space_->fit(reinterpret_cast<float *>(data), 2);
  space_->prefetch_by_address(data);
}

TEST_F(SQ4SpaceTest, HandleInvalidInsert) {
  float data[4] = {1.0, 2.0, 3.0, 4.0};
  for (uint32_t i = 0; i < capacity_; ++i) {
    space_->insert(data);
  }
  uint32_t id = space_->insert(data);
  EXPECT_EQ(id, -1);
}

TEST_F(SQ4SpaceTest, HandleFileErrors) {
  std::string filename = "non_existent_file.bin";
  std::string_view file_name_view = filename;
  SQ4Space<> new_space;
  EXPECT_THROW(new_space.load(file_name_view), std::runtime_error);
}

// ============================================================================
// Metadata Tests
// ============================================================================

class SQ4SpaceMetadataTest : public ::testing::Test {
 protected:
  struct TestMetadata {
    int id;
    float value;
    char name[32];
  };

  using SpaceType =
      SQ4Space<float, float, uint32_t, SequentialStorage<uint8_t, uint32_t>, TestMetadata>;

  void SetUp() override {
    dim_ = 4;
    capacity_ = 10;
    metric_ = MetricType::L2;
    file_name_ = "test_sq4_space_metadata.bin";
    db_path_ = "./test_sq4_rocksdb";
    cleanup_test_files();
  }

  void TearDown() override { cleanup_test_files(); }

  void cleanup_test_files() {
    if (std::filesystem::exists(file_name_)) {
      std::filesystem::remove(file_name_);
    }
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove_all(db_path_);
    }
  }

  std::vector<TestMetadata> make_test_metadata(uint32_t item_cnt) {
    std::vector<TestMetadata> metadata(item_cnt);
    for (uint32_t i = 0; i < item_cnt; ++i) {
      metadata[i].id = static_cast<int>(i);
      metadata[i].value = static_cast<float>(i) * 2.5f;
      snprintf(metadata[i].name, sizeof(metadata[i].name), "item_%u", i);
    }
    return metadata;
  }

  std::shared_ptr<SpaceType> space_;
  std::string file_name_;
  std::string db_path_;
  size_t dim_;
  uint32_t capacity_;
  MetricType metric_;
};

TEST_F(SQ4SpaceMetadataTest, ConstructionWithMetadata) {
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, metric_, config);

  EXPECT_EQ(space_->get_dim(), dim_);
  EXPECT_EQ(space_->get_capacity(), capacity_);
}

TEST_F(SQ4SpaceMetadataTest, FitWithMetadata) {
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, metric_, config);

  float data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  auto metadata = make_test_metadata(2);

  space_->fit(data, 2, metadata.data());

  EXPECT_EQ(space_->get_data_num(), 2);
  EXPECT_TRUE(std::filesystem::exists(db_path_));
}

TEST_F(SQ4SpaceMetadataTest, GetMetadata) {
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, metric_, config);

  float data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  auto metadata = make_test_metadata(2);

  space_->fit(data, 2, metadata.data());

  for (uint32_t i = 0; i < 2; ++i) {
    auto retrieved = space_->get_metadata(i);
    EXPECT_EQ(retrieved.id, metadata[i].id);
    EXPECT_FLOAT_EQ(retrieved.value, metadata[i].value);
    EXPECT_STREQ(retrieved.name, metadata[i].name);
  }
}

TEST_F(SQ4SpaceMetadataTest, SaveAndLoadWithMetadata) {
  RocksDBConfig config;
  config.db_path_ = db_path_;

  float data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  auto metadata = make_test_metadata(2);
  {
    auto save_space = std::make_shared<SpaceType>(capacity_, dim_, metric_, config);
    save_space->fit(data, 2, metadata.data());
    save_space->save(file_name_);
  }
  {
    space_ = std::make_shared<SpaceType>();
    space_->load(file_name_);

    EXPECT_EQ(space_->get_data_num(), 2);

    // Verify metadata persisted
    for (uint32_t i = 0; i < 2; ++i) {
      auto retrieved = space_->get_metadata(i);
      EXPECT_EQ(retrieved.id, metadata[i].id);
      EXPECT_FLOAT_EQ(retrieved.value, metadata[i].value);
      EXPECT_STREQ(retrieved.name, metadata[i].name);
    }
  }
}

TEST_F(SQ4SpaceMetadataTest, FitWithNullMetadata) {
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, metric_, config);

  float data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

  EXPECT_THROW(space_->fit(data, 2, nullptr), std::invalid_argument);
}

TEST_F(SQ4SpaceMetadataTest, FitWithNullVectorData) {
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, metric_, config);

  auto metadata = make_test_metadata(2);

  EXPECT_THROW(space_->fit(nullptr, 2, metadata.data()), std::invalid_argument);
}

TEST_F(SQ4SpaceMetadataTest, GetMetadataWithoutMetadata) {
  using SpaceWithoutMetadata = SQ4Space<>;
  auto space_no_meta = std::make_shared<SpaceWithoutMetadata>(capacity_, dim_, metric_);

  float data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  space_no_meta->fit(data, 2);

  EXPECT_THROW(space_no_meta->get_metadata(0), std::runtime_error);
}

TEST_F(SQ4SpaceMetadataTest, MetadataCorrectness) {
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, metric_, config);

  float data[12] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  auto metadata = make_test_metadata(3);

  space_->fit(data, 3, metadata.data());

  // Verify each metadata entry
  for (uint32_t i = 0; i < 3; ++i) {
    auto retrieved = space_->get_metadata(i);
    EXPECT_EQ(retrieved.id, static_cast<int>(i));
    EXPECT_FLOAT_EQ(retrieved.value, static_cast<float>(i) * 2.5f);

    char expected_name[32];
    snprintf(expected_name, sizeof(expected_name), "item_%u", i);
    EXPECT_STREQ(retrieved.name, expected_name);
  }
}

}  // namespace alaya
