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

#include "space/raw_space.hpp"
#include <gtest/gtest.h>
#include <sys/types.h>
#include <cmath>
#include "utils/metric_type.hpp"
namespace alaya {

using IDType = uint32_t;
using DataType = float;
using DistanceType = float;

class RawSpaceTest : public ::testing::Test {
 protected:
  RawSpaceTest() {
    // First, we initialize the RawSpace object with a capacity of 100 and a dimensionality of 3.
    space_ = std::make_unique<RawSpace<DataType, DistanceType, IDType>>(100, 3, MetricType::L2);
  }

  std::unique_ptr<RawSpace<DataType, DistanceType, IDType>> space_;
};

// Let's test the fit method by inserting some data points.
TEST_F(RawSpaceTest, TestFit) {
  // Here's some test data: 3 points in 3D space.
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  // We call the fit method to load this data into our RawSpace.
  space_->fit(data.data(), 3);

  // Now, let's check if the number of data points has been correctly updated.
  // We expect it to be 3 since we inserted 3 points.
  ASSERT_EQ(space_->get_data_num(), 3);

  // The dimensionality of each data point should be 3 as well.
  ASSERT_EQ(space_->get_dim(), 3);
}

// Now let's test the insertion and deletion of data points.
TEST_F(RawSpaceTest, TestInsertDelete) {
  // Prepare some data points to insert.
  std::vector<float> data1 = {1.0, 2.0, 3.0};
  std::vector<float> data2 = {4.0, 5.0, 6.0};

  // Insert the first data point and store the ID returned by the insert function.
  IDType id1 = space_->insert(data1.data());

  // Insert the second data point.
  IDType id2 = space_->insert(data2.data());

  // After inserting, we should have 2 data points in total.
  ASSERT_EQ(space_->get_avl_data_num(), 2);

  // Now, let's delete the first data point by its ID.
  space_->remove(id1);

  // After deletion, there should only be 1 data point left.
  ASSERT_EQ(space_->get_data_num(), 2);
  ASSERT_EQ(space_->get_avl_data_num(), 1);

  space_->remove(id2);
  // After deletion, there should only be 0 data point left.
  ASSERT_EQ(space_->get_data_num(), 2);
  ASSERT_EQ(space_->get_avl_data_num(), 0);
}

// Let's test if the distance calculation is working as expected.
TEST_F(RawSpaceTest, TestDistance) {
  // Prepare two data points.
  std::vector<float> data1 = {1.0, 2.0, 3.0};
  std::vector<float> data2 = {4.0, 5.0, 6.0};

  // Insert both data points into the RawSpace.
  space_->insert(data1.data());
  space_->insert(data2.data());

  // Now we calculate the L2 distance between the two points.
  float distance = space_->get_distance(0, 1);

  // We know the L2 distance between these two points should be:
  float expected_distance =
      (1.0 - 4.0) * (1.0 - 4.0) + (2.0 - 5.0) * (2.0 - 5.0) + (3.0 - 6.0) * (3.0 - 6.0);

  // Check if the calculated distance matches the expected distance.
  ASSERT_FLOAT_EQ(distance, expected_distance);
}

TEST_F(RawSpaceTest, TestDistanceUInt8) {
  // Prepare two data points.
  std::vector<uint8_t> data1 = {183, 0, 0};
  std::vector<uint8_t> data2 = {107, 2, 3};

  RawSpace<uint8_t> space(100, 3, MetricType::L2);

  // Insert both data points into the RawSpace.
  space.insert(data1.data());
  space.insert(data2.data());

  // Now we calculate the L2 distance between the two points.
  float distance = space.get_distance(0, 1);

  // We know the L2 distance between these two points should be:
  float expected_distance = 5789;

  // Check if the calculated distance matches the expected distance.
  ASSERT_FLOAT_EQ(distance, expected_distance);
}

}  // namespace alaya
