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
#include <fstream>
#include <utility>
#include <vector>

#include "../../utils/memory.hpp"
namespace alaya {

/**
 * @brief The upper layer of the graph, including multiple layers.
 *
 * @tparam NodeIDType The data type for storing IDs of nodes.
 * @tparam EdgeIDType The data type for storing IDs of edges.
 vectors that need to be indexed, with the default type being uint32_t.
 */
template <typename NodeIDType = uint32_t, typename EdgeIDType = NodeIDType>
class OverlayGraph {
 public:
  NodeIDType node_num_;  ///< The number of nodes in the graph.
  EdgeIDType max_nbrs_;  ///< The number of edges in each node.
  NodeIDType ep_;        ///< The entry point of the graph.

  std::vector<uint32_t> levels_;  ///< Each entry stores the highest layer of each node.
  std::vector<std::vector<EdgeIDType, AlignedAlloc<EdgeIDType>>>
      lists_;  ///< Each entry stores the edges (of all levels) of each node.

  OverlayGraph() = delete;

  /**
   * @brief Construct a new OverlayGraph object.
   *
   * @param node_num The number of nodes in the graph.
   * @param max_nbrs The maximum number of neighbors.
   */
  OverlayGraph(NodeIDType node_num, EdgeIDType max_nbrs)
      : node_num_(node_num), max_nbrs_(max_nbrs), levels_(node_num), lists_(node_num) {}

  ~OverlayGraph() = default;

  OverlayGraph(const OverlayGraph &rhs) = delete;
  OverlayGraph(OverlayGraph &&rhs) = delete;

  /**
   * @brief Set the level for ecah node.
   *
   * @param node_id  The id of current node.
   * @param level    The maximum layer for current node.
   */
  auto set_level(NodeIDType node_id, uint32_t level) -> void { levels_[node_id] = level; }

  /**
   * @brief Get the j-th edge of the i-th node at the level-th level.
   *
   * @param i Index of the node.
   * @param j Index of the edge.
   * @return NodeIDType
   */
  auto at(uint32_t level, NodeIDType i, EdgeIDType j) const -> NodeIDType {
    return lists_[i][(level - 1) * max_nbrs_ + j];
  }

  /**
   * @brief Get the j-th edge of the i-th node at the level-th level.
   *
   * @param i Index of the node.
   * @param j Index of the edge.
   * @return NodeIDType
   */
  auto at(uint32_t level, NodeIDType i, EdgeIDType j) -> EdgeIDType & {
    return lists_[i][(level - 1) * max_nbrs_ + j];
  }

  /**
   * @brief Get the edges of the i-th node at the level-th level.
   *
   * @param level The level of the node.
   * @param i Index of the node.
   * @return NodeIDType The edges of the node,  i.e., the node ids of its neighbours.
   */
  auto edges(uint32_t level, NodeIDType i) const -> const NodeIDType * {
    return lists_[i].data() + (level - 1) * max_nbrs_;
  }

  /**
   * @brief Get the edges of the i-th node at the level-th level.
   *
   * @param level The level of the node.
   * @param i Index of the node.
   * @return NodeIDType The edges of the node,  i.e., the node ids of its neighbours.
   */
  auto edges(uint32_t level, NodeIDType u) -> NodeIDType * {
    return lists_[u].data() + (level - 1) * max_nbrs_;
  }

  /**
   * @brief Initialize the ep for search.
   *
   * @tparam CandPoolType The pool type for search.
   * @tparam DistFuncType The Computer function for search.
   * @param cand_pool  The candidate pool for search.
   * @param dist_func The computer function of distance computation.
   */
  template <typename CandPoolType, typename DistFuncType>
  void initialize(CandPoolType &cand_pool, const DistFuncType &dist_func) const {
    uint32_t u = ep_;
    auto cur_dist = dist_func(u);
    for (int level = levels_[u]; level > 0; --level) {
      bool changed = true;
      while (changed) {
        changed = false;
        auto list = edges(level, u);
        for (uint32_t i = 0; i < max_nbrs_ && list[i] != static_cast<NodeIDType>(-1); ++i) {
          auto v = list[i];
          auto dist = dist_func(v);
          if (dist < cur_dist) {
            cur_dist = dist;
            u = v;
            changed = true;
          }
        }
      }
    }
    cand_pool.insert(u, cur_dist);
    cand_pool.vis_.set(u);
  }

  /**
   * @brief Load the graph from a file.
   *
   * @param filename File path.
   */
  void load(std::ifstream &reader) {
    static_assert(std::is_trivial<NodeIDType>::value && std::is_standard_layout<NodeIDType>::value,
                  "IDType must be a POD type");
    reader.read(reinterpret_cast<char *>(&node_num_), 4);
    reader.read(reinterpret_cast<char *>(&max_nbrs_), 4);
    reader.read(reinterpret_cast<char *>(&ep_), 4);

    levels_.clear();
    lists_.clear();
    levels_.resize(node_num_);
    lists_.resize(node_num_);

    for (NodeIDType i = 0; i < node_num_; ++i) {
      int cur;
      reader.read(reinterpret_cast<char *>(&cur), 4);
      levels_[i] = cur / max_nbrs_;

      if (lists_[i].capacity() < static_cast<size_t>(cur)) {
        lists_[i].reserve(cur);
      }
      lists_[i].clear();
      lists_[i].resize(cur, -1);

      reader.read(reinterpret_cast<char *>(lists_[i].data()), cur * 4);
    }
  }

  /**
   * @brief Save the graph to a file.
   *
   * @param filename File path.
   */
  void save(std::ofstream &writer) const {
    static_assert(std::is_trivial<NodeIDType>::value && std::is_standard_layout<NodeIDType>::value,
                  "IDType must be a POD type");
    writer.write(const_cast<char *>(reinterpret_cast<const char *>(&node_num_)), 4);
    writer.write(const_cast<char *>(reinterpret_cast<const char *>(&max_nbrs_)), 4);
    writer.write(const_cast<char *>(reinterpret_cast<const char *>(&ep_)), 4);
    for (NodeIDType i = 0; i < node_num_; ++i) {
      int cur = levels_[i] * max_nbrs_;
      writer.write(reinterpret_cast<char *>(&cur), 4);
      writer.write(const_cast<char *>(reinterpret_cast<const char *>(lists_[i].data())), cur * 4);
    }
  }
};

}  // namespace alaya
