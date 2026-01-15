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
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <memory>
#include <string_view>
#include <thread>

#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "space/raw_space.hpp"
namespace alaya {

class HNSWTest : public ::testing::Test {
 protected:
  // NOLINTBEGIN
  void SetUp() {
    // TODO: Abstracted into a rand test data class for easy reuse
    // NOLINTEND
    max_node_ = 100;
    dim_ = 1024;
    data_ = new float[max_node_ * dim_];

    // Init the vector data.
    srand(time(nullptr));
    for (uint32_t i = 0; i < max_node_; ++i) {
      for (uint32_t j = 0; j < dim_; ++j) {
        data_[i * dim_ + j] = rand() % max_node_;
      }
    }
    // build the unified data manager to compute distance.
    space_ = std::make_shared<RawSpace<>>(max_node_, dim_, MetricType::L2);
    space_->fit(data_, max_node_);
    hnsw_ = std::make_unique<HNSWBuilder<RawSpace<>>>(space_);
  }
  // NOLINTBEGIN
  void TearDown() {
    // NOLINTEND
    delete[] data_;
    if (std::filesystem::exists(filename_)) {
      remove(filename_.data());
    }
  }

  uint32_t max_thread_num_ = std::thread::hardware_concurrency();
  uint32_t max_node_;               ///< The number of vector data.
  uint32_t dim_;                    ///< The dim of vector data.
  std::string_view metric_ = "L2";  /// The metric type for building graph.
  float *data_ = nullptr;           // Store the vector data.
  std::unique_ptr<HNSWBuilder<RawSpace<>>> hnsw_ = nullptr;
  std::shared_ptr<RawSpace<>> space_ = nullptr;
  std::string_view filename_ = "hnsw.graph";
};

TEST_F(HNSWTest, BuildGraphTest) {
  auto built_graph = hnsw_->build_graph(max_thread_num_);
  built_graph->save(filename_);

  auto graph = std::make_unique<Graph<uint32_t>>(max_node_, hnsw_->max_nbrs_underlay_);
  graph->load(filename_);

  // Test the upper layer graph.
  for (uint32_t i = 0; i < graph->max_nodes_; ++i) {
    for (uint32_t j = 0; j < graph->max_nbrs_; ++j) {
      EXPECT_EQ(graph->at(i, j), built_graph->at(i, j));
    }
  }
  // Test the over layer graph.
  for (uint32_t i = 0; i < graph->max_nodes_; ++i) {
    int level = graph->overlay_graph_->levels_[i];
    for (int j = 1; j <= level; ++j) {
      for (uint32_t k = 0; k < graph->max_nbrs_; ++k) {
        EXPECT_EQ(graph->overlay_graph_->at(j, i, k), built_graph->overlay_graph_->at(j, i, k));
      }
    }
  }
}

TEST_F(HNSWTest, MultipleThreadBuildGraphTest) {
  auto hnsw_graph = hnsw_->build_graph(max_thread_num_);
  hnsw_graph->save(filename_);

  auto graph = std::make_unique<Graph<uint32_t>>(max_node_, hnsw_->max_nbrs_underlay_);
  graph->load(filename_);

  // Test the upper layer graph.
  for (uint32_t i = 0; i < graph->max_nodes_; ++i) {
    for (uint32_t j = 0; j < graph->max_nbrs_; ++j) {
      EXPECT_EQ(graph->at(i, j), hnsw_graph->at(i, j));
    }
  }
  // Test the over layer graph.
  for (uint32_t i = 0; i < graph->max_nodes_; ++i) {
    int level = graph->overlay_graph_->levels_[i];

    for (int j = 1; j <= level; ++j) {
      for (uint32_t k = 0; k < graph->max_nbrs_; ++k) {
        EXPECT_EQ(graph->overlay_graph_->at(j, i, k), hnsw_graph->overlay_graph_->at(j, i, k));
      }
    }
  }
}

}  // namespace alaya
