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
#include "utils/metadata_filter.hpp"
#include "utils/scalar_data.hpp"

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
// Metadata Tests with ScalarData
// ============================================================================

class SQ4SpaceMetadataTest : public ::testing::Test {
 protected:
  using SpaceType = SQ4Space<float, float, uint32_t, SequentialStorage<uint8_t, uint32_t>, ScalarData>;

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

  auto make_test_metadata(uint32_t item_cnt) -> std::vector<ScalarData> {
    std::vector<ScalarData> metadata(item_cnt);
    for (uint32_t i = 0; i < item_cnt; ++i) {
      MetadataMap meta;
      meta["index"] = static_cast<int64_t>(i);
      meta["value"] = static_cast<double>(i) * 2.5;
      meta["name"] = std::string("item_") + std::to_string(i);
      metadata[i] = ScalarData("item_" + std::to_string(i), "doc_content_" + std::to_string(i), meta);
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
    auto retrieved = space_->get_scalar_data(i);
    EXPECT_EQ(retrieved.item_id, metadata[i].item_id);
    EXPECT_EQ(retrieved.document, metadata[i].document);
    EXPECT_EQ(std::get<int64_t>(retrieved.metadata.at("index")), i);
    EXPECT_DOUBLE_EQ(std::get<double>(retrieved.metadata.at("value")), static_cast<double>(i) * 2.5);
    EXPECT_EQ(std::get<std::string>(retrieved.metadata.at("name")), "item_" + std::to_string(i));
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
      auto retrieved = space_->get_scalar_data(i);
      EXPECT_EQ(retrieved.item_id, metadata[i].item_id);
      EXPECT_EQ(retrieved.document, metadata[i].document);
      EXPECT_EQ(std::get<int64_t>(retrieved.metadata.at("index")), i);
      EXPECT_DOUBLE_EQ(std::get<double>(retrieved.metadata.at("value")), static_cast<double>(i) * 2.5);
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

  EXPECT_THROW(space_no_meta->get_scalar_data(0), std::runtime_error);
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
    auto retrieved = space_->get_scalar_data(i);
    EXPECT_EQ(retrieved.item_id, "item_" + std::to_string(i));
    EXPECT_EQ(std::get<int64_t>(retrieved.metadata.at("index")), static_cast<int64_t>(i));
    EXPECT_DOUBLE_EQ(std::get<double>(retrieved.metadata.at("value")), static_cast<double>(i) * 2.5);
  }
}

TEST_F(SQ4SpaceMetadataTest, GetScalarDataWithEmptyFilter) {
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, metric_, config);

  float data[20] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0};
  auto metadata = make_test_metadata(5);
  space_->fit(data, 5, metadata.data());

  MetadataFilter empty_filter;
  auto results = space_->get_scalar_data(empty_filter, 10);

  EXPECT_EQ(results.size(), 5);
}

TEST_F(SQ4SpaceMetadataTest, GetScalarDataWithEqFilter) {
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, metric_, config);

  float data[20] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0};
  auto metadata = make_test_metadata(5);
  space_->fit(data, 5, metadata.data());

  MetadataFilter filter;
  filter.add_eq("index", static_cast<int64_t>(2));

  auto results = space_->get_scalar_data(filter, 10);

  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].first, 2);
  EXPECT_EQ(results[0].second.item_id, "item_2");
}

TEST_F(SQ4SpaceMetadataTest, GetScalarDataWithGtFilter) {
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, metric_, config);

  float data[20] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0};
  auto metadata = make_test_metadata(5);
  space_->fit(data, 5, metadata.data());

  MetadataFilter filter;
  filter.add_gt("value", 5.0);  // value = i * 2.5

  auto results = space_->get_scalar_data(filter, 10);

  // Items with value > 5.0: item_3 (7.5), item_4 (10.0)
  EXPECT_EQ(results.size(), 2);
}

TEST_F(SQ4SpaceMetadataTest, GetScalarDataWithLimit) {
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, metric_, config);

  float data[20] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0};
  auto metadata = make_test_metadata(5);
  space_->fit(data, 5, metadata.data());

  MetadataFilter empty_filter;
  auto results = space_->get_scalar_data(empty_filter, 2);

  EXPECT_EQ(results.size(), 2);
}

TEST_F(SQ4SpaceMetadataTest, GetScalarDataWithOrFilter) {
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, metric_, config);

  float data[20] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0};
  auto metadata = make_test_metadata(5);
  space_->fit(data, 5, metadata.data());

  // OR filter: index == 0 OR index == 4
  MetadataFilter filter;
  filter.logic_op = LogicOp::OR;

  auto sub1 = std::make_shared<MetadataFilter>();
  sub1->add_eq("index", static_cast<int64_t>(0));
  filter.add_sub_filter(*sub1);

  auto sub2 = std::make_shared<MetadataFilter>();
  sub2->add_eq("index", static_cast<int64_t>(4));
  filter.add_sub_filter(*sub2);

  auto results = space_->get_scalar_data(filter, 10);

  EXPECT_EQ(results.size(), 2);
}

}  // namespace alaya
