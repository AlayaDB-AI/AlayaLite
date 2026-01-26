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
#include <cstdlib>
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

TEST_F(DatasetTest, DISABLED_LoadDeep1M) {
  // Disabled: too slow due to large dataset download

  auto config = deep1m(data_dir_);

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

TEST_F(DatasetTest, SiftMicroConfig) {
  auto config = sift_micro(data_dir_);

  EXPECT_EQ(config.name_, "siftmicro");
  EXPECT_EQ(config.max_data_num_, 1000);
  EXPECT_EQ(config.max_query_num_, 50);
  // sift_micro uses siftsmall files
  EXPECT_TRUE(config.data_file_.string().find("siftsmall") != std::string::npos);
}

TEST_F(DatasetTest, LoadSiftMicro) {
  auto config = sift_micro(data_dir_);
  auto ds = load_dataset(config);

  EXPECT_EQ(ds.name_, "siftmicro");
  // Verify data is truncated to max limits
  EXPECT_EQ(ds.data_num_, config.max_data_num_);
  EXPECT_EQ(ds.query_num_, config.max_query_num_);
  EXPECT_EQ(ds.dim_, 128);  // SIFT dimension
  EXPECT_EQ(ds.data_.size(), ds.data_num_ * ds.dim_);
  EXPECT_EQ(ds.queries_.size(), ds.query_num_ * ds.dim_);
  // Ground truth should be recomputed for truncated data
  EXPECT_EQ(ds.ground_truth_.size(), ds.query_num_ * ds.gt_dim_);
}

TEST_F(DatasetTest, DataTruncation) {
  // First load full siftsmall
  auto full_config = sift_small(data_dir_);
  auto full_ds = load_dataset(full_config);

  // Then load truncated version
  auto micro_config = sift_micro(data_dir_);
  auto micro_ds = load_dataset(micro_config);

  // Verify truncation
  EXPECT_LT(micro_ds.data_num_, full_ds.data_num_);
  EXPECT_LT(micro_ds.query_num_, full_ds.query_num_);

  // Verify ground truth IDs are valid (within truncated data range)
  for (uint32_t i = 0; i < micro_ds.query_num_; ++i) {
    for (uint32_t j = 0; j < micro_ds.gt_dim_; ++j) {
      uint32_t gt_id = micro_ds.ground_truth_[i * micro_ds.gt_dim_ + j];
      EXPECT_LT(gt_id, micro_ds.data_num_) << "GT ID " << gt_id << " exceeds data_num " << micro_ds.data_num_;
    }
  }
}

}  // namespace alaya
