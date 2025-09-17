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
#include <string>

#include "utils/dataset_utils.hpp"

namespace alaya {

class SIFTTestDataTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path data_dir = std::filesystem::current_path().parent_path() / "data";
    sift_data = std::make_unique<SIFTTestData>(data_dir.string());

    auto dataset_dir = sift_data->get_dataset_dir();
    if (std::filesystem::exists(dataset_dir)) {
      std::filesystem::remove_all(dataset_dir);
    }
    printf("dataset_dir: %s\n", dataset_dir.string().c_str());
    EXPECT_FALSE(sift_data->ensure_dataset());
  }

  std::unique_ptr<SIFTTestData> sift_data;
};

TEST_F(SIFTTestDataTest, ConstructorInitializesCorrectly) {
  EXPECT_NE(sift_data, nullptr);
  EXPECT_TRUE(sift_data->ensure_dataset());

  // Test that dataset name is set correctly
  auto dataset_name = sift_data->get_dataset_name();
  EXPECT_EQ(dataset_name, "siftsmall");

  // check file is exist
  EXPECT_TRUE(std::filesystem::exists(sift_data->get_data_file()));
  EXPECT_TRUE(std::filesystem::exists(sift_data->get_query_file()));
  EXPECT_TRUE(std::filesystem::exists(sift_data->get_gt_file()));
}

}  // namespace alaya
