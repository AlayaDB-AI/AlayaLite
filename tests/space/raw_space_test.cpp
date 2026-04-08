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
#include <filesystem>
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

TEST_F(RawSpaceTest, TestDistanceInnerProductMetric) {
  std::vector<float> data1 = {1.0F, 2.0F, 3.0F};
  std::vector<float> data2 = {4.0F, 5.0F, 6.0F};

  RawSpace<DataType, DistanceType, IDType> space(100, 3, MetricType::IP);
  space.insert(data1.data());
  space.insert(data2.data());

  float distance = space.get_distance(0, 1);
  float expected_distance = -((1.0F * 4.0F) + (2.0F * 5.0F) + (3.0F * 6.0F));

  ASSERT_FLOAT_EQ(distance, expected_distance);
}

TEST_F(RawSpaceTest, TestDistanceCosineMetricUsesInnerProductPath) {
  std::vector<float> data1 = {1.0F, 0.0F, 2.0F};
  std::vector<float> data2 = {0.5F, 3.0F, 4.0F};

  RawSpace<DataType, DistanceType, IDType> space(100, 3, MetricType::COS);
  space.insert(data1.data());
  space.insert(data2.data());

  float distance = space.get_distance(0, 1);
  float expected_distance = -((1.0F * 0.5F) + (0.0F * 3.0F) + (2.0F * 4.0F));

  ASSERT_FLOAT_EQ(distance, expected_distance);
}

TEST_F(RawSpaceTest, TestFitRejectsNullDataAndCapacityOverflow) {
  std::vector<float> data = {1.0F, 2.0F, 3.0F};

  EXPECT_THROW(space_->fit(static_cast<const float *>(nullptr), 1), std::invalid_argument);
  EXPECT_THROW(space_->fit(data.data(), 101), std::length_error);
}

TEST_F(RawSpaceTest, TestScalarApisThrowWithoutMetadataStorage) {
  MetadataFilter filter;
  filter.add_eq("category", std::string("A"));

  EXPECT_THROW(space_->remove(std::string("missing")), std::runtime_error);
  EXPECT_THROW(space_->get_scalar_data(0), std::runtime_error);
  EXPECT_THROW(space_->get_scalar_data(std::string("missing")), std::runtime_error);
  EXPECT_THROW(space_->get_scalar_data(filter, 1), std::runtime_error);
}

TEST(RawSpaceScalarTest, TestFitInsertAndFilterScalarData) {
  const std::string db_path = "./test_raw_space_scalar_db";
  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove_all(db_path);
  }

  RocksDBConfig config;
  config.db_path_ = db_path;
  config.indexed_fields_ = {"category"};

  using RawSpaceWithScalar =
      RawSpace<float, float, uint32_t, SequentialStorage<float, uint32_t>, ScalarData>;
  RawSpaceWithScalar space(10, 3, MetricType::L2, config);

  std::vector<float> data = {
      1.0F,
      2.0F,
      3.0F,
      4.0F,
      5.0F,
      6.0F,
  };

  std::vector<ScalarData> scalar = {
      ScalarData("id_1", "doc_1", MetadataMap{{"category", std::string("A")}}),
      ScalarData("id_2", "doc_2", MetadataMap{{"category", std::string("B")}}),
  };

  space.fit(data.data(), 2, scalar.data());

  auto [internal_id, scalar_data] = space.get_scalar_data("id_1");
  EXPECT_EQ(internal_id, 0U);
  EXPECT_EQ(scalar_data.document, "doc_1");

  MetadataFilter filter;
  filter.add_eq("category", std::string("A"));
  auto results = space.get_scalar_data(filter, 10);
  ASSERT_EQ(results.size(), 1U);
  EXPECT_EQ(results[0].second.item_id, "id_1");

  std::vector<float> extra = {7.0F, 8.0F, 9.0F};
  ScalarData extra_scalar("id_3", "doc_3", MetadataMap{{"category", std::string("A")}});
  auto new_id = space.insert(extra.data(), &extra_scalar);
  EXPECT_EQ(new_id, 2U);
  EXPECT_EQ(space.get_scalar_data(new_id).item_id, "id_3");

  space.remove(new_id);
  EXPECT_FALSE(space.get_scalar_storage()->find_by_item_id("id_3").has_value());
  space.close_db();
  std::filesystem::remove_all(db_path);
}

TEST(RawSpaceScalarTest, TestFitRejectsMissingScalarDataAndUnknownItemId) {
  const std::string db_path = "./test_raw_space_scalar_invalid_db";
  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove_all(db_path);
  }

  RocksDBConfig config;
  config.db_path_ = db_path;
  config.indexed_fields_ = {"category"};

  using RawSpaceWithScalar =
      RawSpace<float, float, uint32_t, SequentialStorage<float, uint32_t>, ScalarData>;
  RawSpaceWithScalar space(2, 3, MetricType::L2, config);

  std::vector<float> data = {1.0F, 2.0F, 3.0F};
  std::vector<ScalarData> scalar = {
      ScalarData("id_1", "doc_1", MetadataMap{{"category", std::string("A")}}),
  };

  EXPECT_THROW(space.fit(static_cast<const float *>(nullptr), 1, scalar.data()),
               std::invalid_argument);
  EXPECT_THROW(space.fit(data.data(), 3, scalar.data()), std::length_error);
  EXPECT_THROW(space.fit(data.data(), 1, static_cast<const ScalarData *>(nullptr)),
               std::invalid_argument);

  space.fit(data.data(), 1, scalar.data());

  EXPECT_THROW(space.remove(std::string("missing")), std::runtime_error);
  EXPECT_THROW(space.get_scalar_data(std::string("missing")), std::runtime_error);

  space.close_db();
  std::filesystem::remove_all(db_path);
}

TEST(RawSpaceScalarTest, TestRemoveByItemIdAndRespectFilterLimit) {
  const std::string db_path = "./test_raw_space_scalar_remove_db";
  if (std::filesystem::exists(db_path)) {
    std::filesystem::remove_all(db_path);
  }

  RocksDBConfig config;
  config.db_path_ = db_path;
  config.indexed_fields_ = {"category"};

  using RawSpaceWithScalar =
      RawSpace<float, float, uint32_t, SequentialStorage<float, uint32_t>, ScalarData>;
  RawSpaceWithScalar space(10, 3, MetricType::L2, config);

  std::vector<float> data = {
      1.0F,
      2.0F,
      3.0F,
      4.0F,
      5.0F,
      6.0F,
      7.0F,
      8.0F,
      9.0F,
  };

  std::vector<ScalarData> scalar = {
      ScalarData("id_1", "doc_1", MetadataMap{{"category", std::string("A")}}),
      ScalarData("id_2", "doc_2", MetadataMap{{"category", std::string("A")}}),
      ScalarData("id_3", "doc_3", MetadataMap{{"category", std::string("B")}}),
  };

  space.fit(data.data(), 3, scalar.data());

  MetadataFilter filter;
  filter.add_eq("category", std::string("A"));
  auto results = space.get_scalar_data(filter, 1);
  ASSERT_EQ(results.size(), 1U);
  EXPECT_EQ(std::get<std::string>(results[0].second.metadata.at("category")), "A");

  auto removed_id = space.remove(std::string("id_2"));
  EXPECT_EQ(removed_id, 1U);
  EXPECT_FALSE(space.get_scalar_storage()->find_by_item_id("id_2").has_value());

  space.close_db();
  std::filesystem::remove_all(db_path);
}

}  // namespace alaya
