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

#pragma once

#include <algorithm>
#include <memory>
#include <utility>
#include "graph.hpp"
#include "graph_concepts.hpp"
#include "space/space_concepts.hpp"

namespace alaya {

template <typename DistanceSpaceType,
          typename PrimaryGraph,
          typename SecondaryGraph,
          typename DataType = typename DistanceSpaceType::DataTypeAlias,
          typename DistanceType = typename DistanceSpaceType::DistanceTypeAlias,
          typename IDType = typename DistanceSpaceType::IDTypeAlias>
  requires(Space<DistanceSpaceType> && GraphBuilder<PrimaryGraph> && GraphBuilder<SecondaryGraph>)
struct FusionGraphBuilder {
  using DistanceSpaceTypeAlias = DistanceSpaceType;
  std::shared_ptr<DistanceSpaceType> space_ = nullptr;
  uint32_t max_nbrs_;         ///< Maximum number of neighbors for each node.
  uint32_t ef_construction_;  ///< Size of the search pool during graph construction.
  explicit FusionGraphBuilder(const std::shared_ptr<DistanceSpaceType> &space,
                              uint32_t R = 32,
                              uint32_t L = 200)
      : space_(space), max_nbrs_(R) {
    space_ = space;
    ef_construction_ = L;
  }

  /**
   * @brief Build the fusion graph by combining primary and secondary graphs.
   *
   * This function builds the fusion graph by first constructing the primary and secondary graphs
   * using their respective graph builders. It then combines the neighbors from both graphs into
   * a single fusion graph, ensuring no duplicate neighbors are included.
   *
   * @return std::unique_ptr<Graph<DataType, IDType>> The constructed fusion graph.
   */
  auto build_graph(uint32_t thread_num = 1) -> std::unique_ptr<Graph<DataType, IDType>> {
    // Create primary and secondary graph builders
    auto primary_graph_builder =
        std::make_unique<PrimaryGraph>(space_, max_nbrs_, ef_construction_);
    auto secondary_graph_builder =
        std::make_unique<SecondaryGraph>(space_, max_nbrs_, ef_construction_);

    // Build primary and secondary graphs
    auto primary_graph = primary_graph_builder->build_graph(thread_num);
    auto secondary_graph = secondary_graph_builder->build_graph(thread_num);

    // Initialize the fusion graph with double the maximum number of neighbors
    auto fusion_graph =
        std::make_unique<Graph<DataType, IDType>>(space_->get_capacity(), 2 * max_nbrs_);

    uint32_t max_edge = 0;
    for (IDType i = 0; i < space_->get_data_num(); i++) {
      uint32_t idx = 0;
      // Add neighbors from the primary graph
      for (IDType j = 0; j < max_nbrs_; j++) {
        if (primary_graph->at(i, j) == Graph<DataType, IDType>::kEmptyId) {
          break;
        }
        fusion_graph->at(i, idx++) = primary_graph->at(i, j);
      }
      // Add neighbors from the secondary graph
      for (IDType j = 0; j < max_nbrs_; j++) {
        if (secondary_graph->at(i, j) == Graph<DataType, IDType>::kEmptyId) {
          break;
        }

        bool duplicate = false;
        for (IDType k = 0; k < idx; k++) {
          if (fusion_graph->at(i, k) == secondary_graph->at(i, j)) {
            duplicate = true;
            break;
          }
        }

        if (duplicate) {
          continue;
        }
        fusion_graph->at(i, idx++) = secondary_graph->at(i, j);
      }

      max_edge = std::max(max_edge, idx);
    }

    // copy the graph to the fusion graph
    auto final_graph = std::make_unique<Graph<DataType, IDType>>(space_->get_capacity(), max_edge);
    for (IDType i = 0; i < space_->get_data_num(); i++) {
      for (IDType j = 0; j < max_edge; j++) {
        final_graph->at(i, j) = fusion_graph->at(i, j);
      }
    }

    // Copy the overlay graph if it exists
    if (primary_graph->overlay_graph_ != nullptr) {
      final_graph->overlay_graph_ = std::move(primary_graph->overlay_graph_);
    } else if (secondary_graph->overlay_graph_ != nullptr) {
      final_graph->overlay_graph_ = std::move(secondary_graph->overlay_graph_);
    } else {
      for (size_t i = 0; i < primary_graph->eps_.size(); i++) {
        final_graph->eps_.push_back(primary_graph->eps_[i]);
      }
      for (size_t i = 0; i < secondary_graph->eps_.size(); i++) {
        final_graph->eps_.push_back(secondary_graph->eps_[i]);
      }
    }
    return final_graph;
  }
  /**
   * @TODO Implement the function to prune the graph.
   *
   */
  void prune_graph(const std::unique_ptr<Graph<DataType, IDType>> &graph) {
    // Prune the graph
  }
};

}  // namespace alaya
