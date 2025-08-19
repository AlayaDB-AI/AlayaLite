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

#include <sys/types.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include "../../utils/log.hpp"
#include "overlay_graph.hpp"
#include "storage/sequential_storage.hpp"
namespace alaya {

constexpr static int kEmptyId = -1;  ///< The id of empty node.

/**
 * @brief Unified structure for a graph , supports searching for vectors.
 *
 * @tparam DataType The data type for storing the vectors that need to be indexed/
 * @tparam NodeIDType The data type for storing IDs of nodes.
 * @tparam EdgeIDType The data type for storing IDs of edges.
 */
template <typename DataType = float, typename NodeIDType = uint32_t,
          typename DataStorage = SequentialStorage<NodeIDType, NodeIDType>>
struct Graph {
  using EdgeIDType = uint32_t;
  constexpr static NodeIDType kEmptyId = -1;
  using OverlayGraphType = OverlayGraph<NodeIDType>;  ///< the overlay graph type

  NodeIDType max_nodes_;      ///< node_num_ is the number of nodes in the graph
  EdgeIDType max_nbrs_;       ///< max_nbrs_ is the maximum number of neighbors
  DataStorage data_storage_;  ///< the data of the grpah: each node has max_nbrs_ edges, i.e., the
                              ///< node ids of its neighbors
  std::unique_ptr<OverlayGraphType> overlay_graph_ = nullptr;  ///< the overlay graph of HNSW
  std::vector<NodeIDType> eps_;                                ///< the entry points

  // bool include_raw_data_;  ///< include_raw_data_ is a flag to indicate whether the raw data is
  ///< included in the graph
  // DataType *base_data_ = nullptr;  ///< the raw data of the graph

  Graph() = default;

  Graph(NodeIDType max_nodes, EdgeIDType max_nbrs) : max_nodes_(max_nodes), max_nbrs_(max_nbrs) {
    uint32_t item_size = max_nbrs * sizeof(NodeIDType);
    data_storage_.init(item_size, max_nodes, -1);
  }

  Graph(const Graph &) = delete;
  auto operator=(const Graph &) -> Graph & = delete;
  Graph(Graph &&) = delete;
  auto operator=(Graph &&) -> Graph & = delete;

  ~Graph() = default;

  /**
   * @brief Get the Edges object
   *
   * @param node_id The node id.
   * @return NodeIDType* The edges of the node, i.e., the node ids of its neighbours.
   */
  auto edges(NodeIDType node_id) const -> const NodeIDType * { return data_storage_[node_id]; }

  /**
   * @brief Get the Edges object
   *
   * @param node_id The node id.
   * @return NodeIDType* The edges of the node,  i.e., the node ids of its neighbours.
   */
  auto edges(NodeIDType node_id) -> NodeIDType * { return data_storage_[node_id]; }

  /**
   * @brief Get the j-th edge of the i-th node.
   *
   * @param i Index of the node.
   * @param j Index of the edge.
   * @return NodeIDType
   */
  auto at(NodeIDType i, EdgeIDType j) const -> NodeIDType { return *(edges(i) + j); }

  /**
   * @brief Get the j-th edge of the i-th node.
   *
   * @param i Index of the node.
   * @param j Index of the edge.
   * @return NodeIDType
   */
  auto at(NodeIDType i, EdgeIDType j) -> NodeIDType & { return *(edges(i) + j); }

  /**
   * @brief Insert a node into the graph.
   *
   * @param edges The edges of the node, i.e., the node ids of its neighbours.
   * @return NodeIDType The id of the inserted node.
   */
  auto insert(NodeIDType *edges) -> NodeIDType { return data_storage_.insert(edges); }

  /**
   * @brief Remove a node from the graph.
   *
   * @param node The id of the node to remove.
   * @return NodeIDType The id of the removed node.
   */
  auto remove(NodeIDType node) -> NodeIDType { return data_storage_.remove(node); }

  /**
   * @brief Update the edges of a node.
   *
   * @param node the node id to update
   * @param edges the updated edges, i.e., the new node ids of its neighbours.
   * @return NodeIDType
   */
  auto update(NodeIDType node, NodeIDType *edges) -> NodeIDType {
    return data_storage_.update(node, edges);
  }

  auto get_ep()const->NodeIDType{
    return overlay_graph_?overlay_graph_->ep_:eps_[0];
  }

  /**
   * @brief Initialize the eps for search.
   *
   * @tparam CandPoolType The pool type for search.
   * @tparam DistFuncType The Computer function for search.
   * @param cand_pool  The candidate pool for search.
   * @param dist_func The computer function of distance computation.
   */
  template <typename CandPoolType, typename DistFuncType>
  void initialize_search(CandPoolType &cand_pool, const DistFuncType &dist_func) const {
    if (overlay_graph_) {
      overlay_graph_->initialize(cand_pool, dist_func);
    } else {
      for (auto ep : eps_) {
        cand_pool.insert(ep, dist_func(ep));
        cand_pool.vis_.set(ep);
      }
    }
  }

  /**
   * @brief Save the graph to a file.
   *
   * @param filename File path.
   */
  void save(std::string_view &filename) const {
    static_assert(std::is_trivial<NodeIDType>::value && std::is_standard_layout<NodeIDType>::value,
                  "IDType must be a POD type");
    std::ofstream writer(std::string(filename), std::ios::binary);
    if (!writer.is_open()) {
      throw std::runtime_error("Cannot open file " + std::string(filename));
    }

    int nep = eps_.size();
    writer.write(reinterpret_cast<char *>(&nep), 4);
    writer.write(const_cast<char *>(reinterpret_cast<const char *>(eps_.data())),
                 nep * sizeof(NodeIDType));
    writer.write(const_cast<char *>(reinterpret_cast<const char *>(&max_nodes_)),
                 sizeof(NodeIDType));
    writer.write(const_cast<char *>(reinterpret_cast<const char *>(&max_nbrs_)),
                 sizeof(NodeIDType));
    data_storage_.save(writer);

    // raw data
    // include_raw_data = include_raw_data;
    // writer.write(const_cast<char *>(reinterpret_cast<const char *>(&include_raw_data_)),
    //  sizeof(bool));
    // if (include_raw_data_) {
    //   if (raw_data == nullptr) {
    //     throw std::runtime_error("raw_data is nullptr");
    //   }
    //   writer.write(const_cast<char *>(reinterpret_cast<const char *>(&dim)), sizeof(uint32_t));
    //   writer.write(const_cast<char *>(reinterpret_cast<const char *>(raw_data)),
    //                max_nodes_ * dim * sizeof(DataType));
    // }

    if (overlay_graph_) {
      overlay_graph_->save(writer);
    }
    LOG_INFO("Graph Saving done in {}\n", filename);
  }

  /**
   * @brief Load the graph from a file.
   *
   * @param filename File path.
   */
  void load(std::string_view &filename) {
    static_assert(std::is_trivial<NodeIDType>::value && std::is_standard_layout<NodeIDType>::value,
                  "IDType must be a POD type");
    std::ifstream reader(filename.data(), std::ios::binary);
    if (!reader.is_open()) {
      throw std::runtime_error("Cannot open file " + std::string(filename));
    }

    int nep;

    reader.read(reinterpret_cast<char *>(&nep), 4);
    eps_.resize(nep);
    reader.read(reinterpret_cast<char *>(eps_.data()), nep * sizeof(NodeIDType));
    reader.read(reinterpret_cast<char *>(&max_nodes_), sizeof(NodeIDType));
    reader.read(reinterpret_cast<char *>(&max_nbrs_), sizeof(NodeIDType));

    data_storage_.load(reader);

    // reader.read(reinterpret_cast<char *>(&include_raw_data_), sizeof(bool));
    // if (include_raw_data_) {
    //   reader.read(reinterpret_cast<char *>(&dim_), sizeof(uint32_t));
    //   base_data_ = static_cast<DataType *>(
    //       alaya::alloc_2m(static_cast<size_t>(max_nodes_) * dim_ * sizeof(DataType)));
    //   reader.read(reinterpret_cast<char *>(base_data_), max_nodes_ * dim_ * sizeof(DataType));
    // }

    if (reader.peek() != EOF) {
      overlay_graph_ = std::make_unique<OverlayGraph<NodeIDType>>(max_nodes_, max_nbrs_);
      overlay_graph_->load(reader);
    }
    LOG_INFO("Graph Loading done\n");
  }

  /**
   * @brief Get the Graph object
   *
   * @return Graph<Id> The final graph of the HNSW .
   */
  auto print_graph() -> void {
    for (int i = 0; i < max_nodes_; i++) {
      for (int j = 0; j < max_nbrs_; j++) {
        if (at(i, j) == -1) {
          break;
        }
        LOG_INFO("u id {} -> v id {}", i, at(i, j));
      }
    }
  }
};

}  // namespace alaya
