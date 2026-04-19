// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string_view>

#include "index/graph/graph.hpp"
#include "index/graph/overlay_graph.hpp"
namespace alaya {
/**
 * @brief Init the GraphTest.
 *
 */
class GraphTest : public ::testing::Test {
 protected:
  // NOLINTBEGIN
  void SetUp() {
    // NOLINTEND
    uint32_t node_nums = 100;
    uint32_t max_nbr_num = 100;
    graph_ = std::make_unique<alaya::Graph<uint32_t, uint32_t>>(node_nums, max_nbr_num);
    // Init the graph at layer 0.
    for (uint32_t i = 0; i < graph_->max_nodes_; ++i) {
      for (uint32_t j = 0; j < graph_->max_nbrs_; ++j) {
        graph_->at(i, j) = j;
      }
    }
    // Init the over layer graphs.
    graph_->overlay_graph_ =
        std::make_unique<OverlayGraph<uint32_t>>(graph_->max_nodes_, graph_->max_nbrs_);

    for (uint32_t i = 0; i < node_nums; ++i) {
      int level = i / 10;
      graph_->overlay_graph_->set_level(i, level);
      if (level > 0) {
        graph_->overlay_graph_->lists_[i].assign(level * graph_->max_nbrs_, -1);
        for (int j = 1; j <= level; ++j) {
          for (uint32_t k = 1; k <= graph_->max_nbrs_; ++k) {
            graph_->overlay_graph_->at(j, i, k - 1) = k;
          }
        }
      }
    }
  }
  // NOLINTBEGIN
  void TearDown() {
    // NOLINTEND
    // Delete graph file
    if (std::filesystem::exists(filename_)) {
      remove(filename_.data());
    }
  }

  std::unique_ptr<alaya::Graph<uint32_t, uint32_t>> graph_ = nullptr;
  std::string_view filename_ = "test_graph.graph";
};

void random_graph(std::unique_ptr<alaya::Graph<uint32_t, uint32_t>> &graph) {
  uint32_t max_node_id = graph->max_nodes_;
  uint32_t max_edge_count = graph->max_nbrs_;

  srand(time(nullptr));
  for (uint32_t i = 0; i < max_node_id; ++i) {
    for (uint32_t j = 0; j < max_edge_count; ++j) {
      graph->at(i, j) = rand() % max_node_id;
    }
  }
}

TEST_F(GraphTest, SimpleTest) {
  EXPECT_EQ(graph_->max_nodes_, 100);
  EXPECT_EQ(graph_->max_nbrs_, 100);

  // Test the iterator.
  for (uint32_t i = 0; i < graph_->max_nodes_; ++i) {
    for (uint32_t j = 0; j < graph_->max_nbrs_; ++j) {
      EXPECT_EQ(graph_->at(i, j), j);
    }
  }

  random_graph(graph_);
  // Testing the save and load function.
  graph_->save(filename_);

  Graph<uint32_t, uint32_t> load_graph(graph_->max_nodes_, graph_->max_nbrs_);
  load_graph.load(filename_);

  for (uint32_t i = 0; i < graph_->max_nodes_; ++i) {
    for (uint32_t j = 0; j < graph_->max_nbrs_; ++j) {
      EXPECT_EQ(graph_->at(i, j), load_graph.at(i, j));
    }
  }
}

TEST_F(GraphTest, OverlayTest) {
  graph_->save(filename_);

  Graph<uint32_t, uint32_t> load_graph(graph_->max_nodes_, graph_->max_nbrs_);
  load_graph.load(filename_);

  // Test the over layer graph.
  for (uint32_t i = 0; i < graph_->max_nodes_; ++i) {
    int level = graph_->overlay_graph_->levels_[i];

    for (int j = 1; j <= level; ++j) {
      for (uint32_t k = 0; k < graph_->max_nbrs_; ++k) {
        EXPECT_EQ(graph_->overlay_graph_->at(j, i, k), load_graph.overlay_graph_->at(j, i, k));
      }
    }
  }
}

}  // namespace alaya
