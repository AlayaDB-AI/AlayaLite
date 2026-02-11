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
#include <filesystem>
#include <memory>
#include <vector>

#include "space/rabitq_space.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/scalar_data.hpp"

namespace alaya {
// NOLINTBEGIN
class RaBitQSpaceTest : public ::testing::Test {
 protected:
  using SpaceType = RaBitQSpace<float, float, uint32_t>;

  void SetUp() override {
    file_name_ = "test_rabitq_space.bin";
    if (std::filesystem::exists(file_name_)) {
      std::filesystem::remove(file_name_);
    }
  }

  void TearDown() override {
    if (std::filesystem::exists(file_name_)) {
      std::filesystem::remove(file_name_);
    }
  }

  // Helper to create a simple 2D dataset, has default dim and capacity
  std::vector<float> make_test_data(uint32_t item_cnt) {
    std::vector<float> data(item_cnt * dim_);
    for (uint32_t i = 0; i < item_cnt; ++i) {
      for (size_t j = 0; j < dim_; ++j) {
        data[i * dim_ + j] = static_cast<float>(i * dim_ + j + 1);
      }
    }
    return data;
  }

  std::shared_ptr<SpaceType> space_;
  const size_t dim_ = 64;
  const uint32_t capacity_ = 10;
  std::string file_name_;
};

TEST_F(RaBitQSpaceTest, ConstructionAndFit) {
  const uint32_t item_cnt = 3;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2);
  auto data = make_test_data(item_cnt);
  space_->fit(data.data(), item_cnt);

  EXPECT_EQ(space_->get_data_num(), item_cnt);
  EXPECT_EQ(space_->get_dim(), dim_);
  EXPECT_EQ(space_->get_capacity(), capacity_);

  for (uint32_t i = 0; i < item_cnt; ++i) {
    const float *vec = space_->get_data_by_id(i);
    for (size_t j = 0; j < dim_; ++j) {
      EXPECT_FLOAT_EQ(vec[j], static_cast<float>(i * dim_ + j + 1));
    }
  }
}

TEST_F(RaBitQSpaceTest, DistanceComputation) {
  const uint32_t item_cnt = 2;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2);

  // vec0 = [0,0,...,0], vec1 = [1,1,...,1]
  std::vector<float> data(2 * dim_, 0.0f);
  std::fill(data.begin() + dim_, data.end(), 1.0f);  // every dimension in the second vector is 1

  space_->fit(data.data(), item_cnt);

  float dist = space_->get_distance(0, 1);
  EXPECT_FLOAT_EQ(dist, static_cast<float>(dim_));  // L2^2 = 64 * (1-0)^2 = 64
}

TEST_F(RaBitQSpaceTest, SaveAndLoad) {
  const uint32_t item_cnt = 2;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2);
  space_->set_ep(1);

  auto data = make_test_data(item_cnt);
  space_->fit(data.data(), item_cnt);

  std::string_view filename = file_name_;
  space_->save(filename);

  auto loaded_space = std::make_shared<SpaceType>();
  loaded_space->load(filename);

  EXPECT_EQ(loaded_space->get_dim(), dim_);
  EXPECT_EQ(loaded_space->get_data_num(), item_cnt);
  EXPECT_EQ(loaded_space->get_capacity(), capacity_);
  EXPECT_EQ(loaded_space->get_ep(), 1u);

  for (uint32_t i = 0; i < item_cnt; ++i) {
    const float *orig = space_->get_data_by_id(i);
    const float *load = loaded_space->get_data_by_id(i);
    for (size_t j = 0; j < dim_; ++j) {
      EXPECT_FLOAT_EQ(orig[j], load[j]);
    }
  }

  EXPECT_FLOAT_EQ(space_->get_distance(0, 1), loaded_space->get_distance(0, 1));

  std::filesystem::remove(filename);
}

TEST_F(RaBitQSpaceTest, InvalidMetric3) {
  EXPECT_THROW(space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::NONE),
               std::runtime_error);
}

TEST_F(RaBitQSpaceTest, ItemCntOverflow) {
  const uint32_t item_cnt = 11;  // item_cnt > capacity_
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2);

  std::vector<float> data(item_cnt * dim_, 0.0f);

  EXPECT_THROW(space_->fit(data.data(), item_cnt), std::length_error);
}

TEST_F(RaBitQSpaceTest, SaveNonExistentPath) {
  const uint32_t item_cnt = 2;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2);
  space_->set_ep(1);

  auto data = make_test_data(item_cnt);
  space_->fit(data.data(), item_cnt);

  std::string_view invalid_path = "/nonexistent_dir/invalid_file.bin";

  EXPECT_THROW(space_->save(invalid_path), std::runtime_error);
}

TEST_F(RaBitQSpaceTest, LoadNonExistentPath) {
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2);

  std::string_view invalid_path = "/nonexistent_dir/invalid_file.bin";

  EXPECT_THROW(space_->load(invalid_path), std::runtime_error);
}

// ============================================================================
// Metadata Tests with ScalarData
// ============================================================================

class RaBitQSpaceMetadataTest : public ::testing::Test {
 protected:
  using SpaceType = RaBitQSpace<float, float, uint32_t, ScalarData>;

  void SetUp() override {
    file_name_ = "test_rabitq_space_metadata.bin";
    db_path_ = "./test_rabitq_rocksdb";
    cleanup_test_files();
  }

  void TearDown() override {
    cleanup_test_files();
  }

  void cleanup_test_files() {
    if (std::filesystem::exists(file_name_)) {
      std::filesystem::remove(file_name_);
    }
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove_all(db_path_);
    }
  }

  auto make_test_data(uint32_t item_cnt) -> std::vector<float> {
    std::vector<float> data(item_cnt * dim_);
    for (uint32_t i = 0; i < item_cnt; ++i) {
      for (size_t j = 0; j < dim_; ++j) {
        data[i * dim_ + j] = static_cast<float>(i * dim_ + j + 1);
      }
    }
    return data;
  }

  auto make_test_metadata(uint32_t item_cnt) -> std::vector<ScalarData> {
    std::vector<ScalarData> metadata(item_cnt);
    for (uint32_t i = 0; i < item_cnt; ++i) {
      MetadataMap meta;
      meta["label"] = static_cast<int64_t>(i);
      meta["score"] = static_cast<double>(i) * 1.5;
      meta["tag"] = std::string("tag_") + std::to_string(i);
      metadata[i] = ScalarData("item_" + std::to_string(i), "content_" + std::to_string(i), meta);
    }
    return metadata;
  }

  std::shared_ptr<SpaceType> space_;
  const size_t dim_ = 64;
  const uint32_t capacity_ = 10;
  std::string file_name_;
  std::string db_path_;
};

TEST_F(RaBitQSpaceMetadataTest, ConstructionWithMetadata) {
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2, config);

  EXPECT_EQ(space_->get_dim(), dim_);
  EXPECT_EQ(space_->get_capacity(), capacity_);
}

TEST_F(RaBitQSpaceMetadataTest, FitWithMetadata) {
  const uint32_t item_cnt = 3;
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2, config);

  auto data = make_test_data(item_cnt);
  auto metadata = make_test_metadata(item_cnt);

  space_->fit(data.data(), item_cnt, metadata.data());

  EXPECT_EQ(space_->get_data_num(), item_cnt);
  EXPECT_TRUE(std::filesystem::exists(db_path_));
}

TEST_F(RaBitQSpaceMetadataTest, GetMetadata) {
  const uint32_t item_cnt = 3;
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2, config);

  auto data = make_test_data(item_cnt);
  auto metadata = make_test_metadata(item_cnt);

  space_->fit(data.data(), item_cnt, metadata.data());

  for (uint32_t i = 0; i < item_cnt; ++i) {
    auto retrieved = space_->get_scalar_data(i);
    EXPECT_EQ(retrieved.item_id, metadata[i].item_id);
    EXPECT_EQ(retrieved.document, metadata[i].document);
    EXPECT_EQ(std::get<int64_t>(retrieved.metadata.at("label")), static_cast<int64_t>(i));
    EXPECT_DOUBLE_EQ(std::get<double>(retrieved.metadata.at("score")), static_cast<double>(i) * 1.5);
    EXPECT_EQ(std::get<std::string>(retrieved.metadata.at("tag")), "tag_" + std::to_string(i));
  }
}

TEST_F(RaBitQSpaceMetadataTest, SaveAndLoadWithMetadata) {
  const uint32_t item_cnt = 3;
  RocksDBConfig config;
  config.db_path_ = db_path_;
  auto data = make_test_data(item_cnt);
  auto metadata = make_test_metadata(item_cnt);
  {
    auto save_space = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2, config);
    save_space->set_ep(1);
    save_space->fit(data.data(), item_cnt, metadata.data());
    save_space->save(file_name_);
  }
  {
    space_ = std::make_shared<SpaceType>();
    space_->load(file_name_);

    EXPECT_EQ(space_->get_dim(), dim_);
    EXPECT_EQ(space_->get_data_num(), item_cnt);
    EXPECT_EQ(space_->get_ep(), 1u);

    // Verify metadata persisted
    for (uint32_t i = 0; i < item_cnt; ++i) {
      auto retrieved = space_->get_scalar_data(i);
      EXPECT_EQ(retrieved.item_id, metadata[i].item_id);
      EXPECT_EQ(retrieved.document, metadata[i].document);
      EXPECT_EQ(std::get<int64_t>(retrieved.metadata.at("label")), static_cast<int64_t>(i));
      EXPECT_DOUBLE_EQ(std::get<double>(retrieved.metadata.at("score")), static_cast<double>(i) * 1.5);
      EXPECT_EQ(std::get<std::string>(retrieved.metadata.at("tag")), "tag_" + std::to_string(i));
    }
  }
}

TEST_F(RaBitQSpaceMetadataTest, FitWithNullMetadata) {
  const uint32_t item_cnt = 3;
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2, config);

  auto data = make_test_data(item_cnt);

  EXPECT_THROW(space_->fit(data.data(), item_cnt, nullptr), std::invalid_argument);
}

TEST_F(RaBitQSpaceMetadataTest, FitWithNullVectorData) {
  const uint32_t item_cnt = 3;
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2, config);

  auto metadata = make_test_metadata(item_cnt);

  EXPECT_THROW(space_->fit(nullptr, item_cnt, metadata.data()), std::invalid_argument);
}

TEST_F(RaBitQSpaceMetadataTest, GetMetadataWithoutMetadata) {
  using SpaceWithoutMetadata = RaBitQSpace<float, float, uint32_t>;
  auto space_no_meta = std::make_shared<SpaceWithoutMetadata>(capacity_, dim_, MetricType::L2);

  const uint32_t item_cnt = 2;
  auto data = make_test_data(item_cnt);
  space_no_meta->fit(data.data(), item_cnt);

  EXPECT_THROW(space_no_meta->get_scalar_data(0), std::runtime_error);
}

TEST_F(RaBitQSpaceMetadataTest, CustomRocksDBConfig) {
  const uint32_t item_cnt = 2;
  RocksDBConfig config;
  config.db_path_ = db_path_;
  config.write_buffer_size_ = 32 << 20;  // 32MB
  config.block_cache_size_mb_ = 256;

  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2, config);

  auto data = make_test_data(item_cnt);
  auto metadata = make_test_metadata(item_cnt);

  space_->fit(data.data(), item_cnt, metadata.data());

  EXPECT_EQ(space_->get_data_num(), item_cnt);
  EXPECT_TRUE(std::filesystem::exists(db_path_));
}

TEST_F(RaBitQSpaceMetadataTest, GetScalarDataWithEmptyFilter) {
  const uint32_t item_cnt = 5;
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2, config);

  auto data = make_test_data(item_cnt);
  auto metadata = make_test_metadata(item_cnt);
  space_->fit(data.data(), item_cnt, metadata.data());

  MetadataFilter empty_filter;
  auto results = space_->get_scalar_data(empty_filter, 10);

  EXPECT_EQ(results.size(), item_cnt);
}

TEST_F(RaBitQSpaceMetadataTest, GetScalarDataWithEqFilter) {
  const uint32_t item_cnt = 5;
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2, config);

  auto data = make_test_data(item_cnt);
  auto metadata = make_test_metadata(item_cnt);
  space_->fit(data.data(), item_cnt, metadata.data());

  MetadataFilter filter;
  filter.add_eq("label", static_cast<int64_t>(2));

  auto results = space_->get_scalar_data(filter, 10);

  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].first, 2);
  EXPECT_EQ(results[0].second.item_id, "item_2");
}

TEST_F(RaBitQSpaceMetadataTest, GetScalarDataWithGtFilter) {
  const uint32_t item_cnt = 5;
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2, config);

  auto data = make_test_data(item_cnt);
  auto metadata = make_test_metadata(item_cnt);
  space_->fit(data.data(), item_cnt, metadata.data());

  MetadataFilter filter;
  filter.add_gt("score", 4.0);  // score = i * 1.5

  auto results = space_->get_scalar_data(filter, 10);

  // Items with score > 4.0: item_3 (4.5), item_4 (6.0)
  EXPECT_EQ(results.size(), 2);
}

TEST_F(RaBitQSpaceMetadataTest, GetScalarDataWithLimit) {
  const uint32_t item_cnt = 5;
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2, config);

  auto data = make_test_data(item_cnt);
  auto metadata = make_test_metadata(item_cnt);
  space_->fit(data.data(), item_cnt, metadata.data());

  MetadataFilter empty_filter;
  auto results = space_->get_scalar_data(empty_filter, 2);

  EXPECT_EQ(results.size(), 2);
}

TEST_F(RaBitQSpaceMetadataTest, GetScalarDataWithOrFilter) {
  const uint32_t item_cnt = 5;
  RocksDBConfig config;
  config.db_path_ = db_path_;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::L2, config);

  auto data = make_test_data(item_cnt);
  auto metadata = make_test_metadata(item_cnt);
  space_->fit(data.data(), item_cnt, metadata.data());

  // OR filter: label == 0 OR label == 4
  MetadataFilter filter;
  filter.logic_op = LogicOp::OR;

  auto sub1 = std::make_shared<MetadataFilter>();
  sub1->add_eq("label", static_cast<int64_t>(0));
  filter.add_sub_filter(*sub1);

  auto sub2 = std::make_shared<MetadataFilter>();
  sub2->add_eq("label", static_cast<int64_t>(4));
  filter.add_sub_filter(*sub2);

  auto results = space_->get_scalar_data(filter, 10);

  EXPECT_EQ(results.size(), 2);
}

// NOLINTEND
}  // namespace alaya
