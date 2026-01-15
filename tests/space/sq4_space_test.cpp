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
// #include "spdlog/fmt/bundled/core.h"
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
  EXPECT_EQ(space_->get_data_num(), 1);  // remove 只是标记删除
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

}  // namespace alaya
