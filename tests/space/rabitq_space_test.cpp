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

  EXPECT_THROW(space_->fit(data.data(), item_cnt), std::runtime_error);
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

// NOLINTEND
}  // namespace alaya
