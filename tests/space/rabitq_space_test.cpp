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
#include "utils/log.hpp"

#include "space/rabitq_space.hpp"

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

TEST_F(RaBitQSpaceTest, InvalidMetric1) {
  EXPECT_THROW(space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::COS),
               std::runtime_error);
}

TEST_F(RaBitQSpaceTest, InvalidMetric2) {
  EXPECT_THROW(space_ = std::make_shared<SpaceType>(capacity_, dim_, MetricType::IP),
               std::runtime_error);
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
// Metadata Tests
// ============================================================================

class RaBitQSpaceMetadataTest : public ::testing::Test {
 protected:
  struct TestMetadata {
    int label;
    float score;
    char tag[16];
  };

  using SpaceType = RaBitQSpace<float, float, uint32_t, TestMetadata>;

  void SetUp() override {
    file_name_ = "test_rabitq_space_metadata.bin";
    db_path_ = "./test_rabitq_rocksdb";
    cleanup_test_files();
  }

  void TearDown() override {
    cleanup_test_files();
    LOG_INFO("TearDown删除文件");
  }

  void cleanup_test_files() {
    if (std::filesystem::exists(file_name_)) {
      std::filesystem::remove(file_name_);
    }
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove_all(db_path_);
    }
  }

  std::vector<float> make_test_data(uint32_t item_cnt) {
    std::vector<float> data(item_cnt * dim_);
    for (uint32_t i = 0; i < item_cnt; ++i) {
      for (size_t j = 0; j < dim_; ++j) {
        data[i * dim_ + j] = static_cast<float>(i * dim_ + j + 1);
      }
    }
    return data;
  }

  std::vector<TestMetadata> make_test_metadata(uint32_t item_cnt) {
    std::vector<TestMetadata> metadata(item_cnt);
    for (uint32_t i = 0; i < item_cnt; ++i) {
      metadata[i].label = static_cast<int>(i);
      metadata[i].score = static_cast<float>(i) * 1.5f;
      snprintf(metadata[i].tag, sizeof(metadata[i].tag), "tag_%u", i);
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
    auto retrieved = space_->get_metadata(i);
    EXPECT_EQ(retrieved.label, metadata[i].label);
    EXPECT_FLOAT_EQ(retrieved.score, metadata[i].score);
    EXPECT_STREQ(retrieved.tag, metadata[i].tag);
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
  LOG_INFO("First space destroyed.");
  {
    space_ = std::make_shared<SpaceType>();
    space_->load(file_name_);

    EXPECT_EQ(space_->get_dim(), dim_);
    EXPECT_EQ(space_->get_data_num(), item_cnt);
    EXPECT_EQ(space_->get_ep(), 1u);

    // Verify metadata persisted
    for (uint32_t i = 0; i < item_cnt; ++i) {
      auto retrieved = space_->get_metadata(i);
      EXPECT_EQ(retrieved.label, metadata[i].label);
      EXPECT_FLOAT_EQ(retrieved.score, metadata[i].score);
      EXPECT_STREQ(retrieved.tag, metadata[i].tag);
    }
  }
  LOG_INFO("Loaded space destroyed.");
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

  EXPECT_THROW(space_no_meta->get_metadata(0), std::runtime_error);
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

// NOLINTEND
}  // namespace alaya
