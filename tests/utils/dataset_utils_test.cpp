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

#include <gtest/gtest.h>
#include <filesystem>

#include "utils/dataset_utils.hpp"

namespace alaya {

class DatasetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    data_dir_ = std::filesystem::current_path().parent_path() / "data";
  }

  std::filesystem::path data_dir_;
};

TEST_F(DatasetTest, LoadSiftSmall) {
  auto config = sift_small(data_dir_);
  if (std::filesystem::exists(config.dir_)) {
    std::filesystem::remove_all(config.dir_);
  }

  auto ds = load_dataset(config);

  EXPECT_EQ(ds.name_, "siftsmall");
  EXPECT_GT(ds.data_num_, 0);
  EXPECT_GT(ds.query_num_, 0);
  EXPECT_GT(ds.dim_, 0);
  EXPECT_EQ(ds.data_.size(), ds.data_num_ * ds.dim_);
  EXPECT_EQ(ds.queries_.size(), ds.query_num_ * ds.dim_);

  EXPECT_TRUE(std::filesystem::exists(config.data_file_));
  EXPECT_TRUE(std::filesystem::exists(config.query_file_));
  EXPECT_TRUE(std::filesystem::exists(config.gt_file_));
}

TEST_F(DatasetTest, LoadDeep1M) {
  auto config = deep1m(data_dir_);
  if (std::filesystem::exists(config.dir_)) {
    std::filesystem::remove_all(config.dir_);
  }

  auto ds = load_dataset(config);

  EXPECT_EQ(ds.name_, "deep1M");
  EXPECT_GT(ds.data_num_, 0);
  EXPECT_GT(ds.query_num_, 0);
  EXPECT_GT(ds.dim_, 0);
  EXPECT_EQ(ds.data_.size(), ds.data_num_ * ds.dim_);
  EXPECT_EQ(ds.queries_.size(), ds.query_num_ * ds.dim_);

  EXPECT_TRUE(std::filesystem::exists(config.data_file_));
  EXPECT_TRUE(std::filesystem::exists(config.query_file_));
  EXPECT_TRUE(std::filesystem::exists(config.gt_file_));
}

}  // namespace alaya
