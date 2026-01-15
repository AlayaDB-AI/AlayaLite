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

#include <chrono>  //NOLINT [build/c++11]
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include "../../../space/space_concepts.hpp"
#include "../../../utils/log.hpp"
#include "../../../utils/thread_pool.hpp"
#include "../../../utils/timer.hpp"
#include "../graph.hpp"
#include "hnswlib.hpp"
#include "space/raw_space.hpp"

namespace alaya {
/**
 * @brief The structure for HNSW , supports L2, IP distance.
 *
 * @tparam DistanceSpaceType
  The distance computer function type for building graph.
 * @tparam IDType The data type for storing IDs is determined by the number of
 vectors that need to be indexed, with the default type being uint32_t.
 */
template <typename DistanceSpaceType>
  requires Space<DistanceSpaceType, typename DistanceSpaceType::DataTypeAlias,
                 typename DistanceSpaceType::DistanceTypeAlias,
                 typename DistanceSpaceType::IDTypeAlias>
struct HNSWBuilder {
  using DataType = typename DistanceSpaceType::DataTypeAlias;
  using DistanceType = typename DistanceSpaceType::DistanceTypeAlias;
  using IDType = typename DistanceSpaceType::IDTypeAlias;
  using DistanceSpaceTypeAlias = DistanceSpaceType;

  uint16_t dim_;  ///< The dimension of the vector data.
  uint16_t
      ef_construction_;  ///< The size of the dynamic candidate list for the construction phase.
  uint32_t max_nbrs_underlay_;  ///< The maximum number of neighbors of the overlay graph.
  uint32_t max_nbrs_overlay_;   ///< The maximum number of neighbors of the overlay graph.

  std::shared_ptr<HNSWImpl<DistanceSpaceType, DataType, DistanceType, IDType>> hnsw_ =
      nullptr;  ///< The HNSW graph.
  std::shared_ptr<DistanceSpaceType> space_ =
      nullptr;  ///< The data manager interface for the HNSW graph.

  /**
   * @brief Construct a new HNSW object.
   *
   * @param dim    The dimension of the vector data.
   * @param R      R is the maximum out-degree of the underlay graph, and half of R is the maximum
   * out-degree of the overlay graph.
   * @param L      The size of the dynamic candidate list for the construction phase.
   */
  explicit HNSWBuilder(const std::shared_ptr<DistanceSpaceType> &space, uint32_t R = 32,
                       uint32_t L = 200)
      : dim_(space->get_dim()), ef_construction_(L), max_nbrs_underlay_(R) {
    max_nbrs_overlay_ = R / 2;
    space_ = space;
  }

  HNSWBuilder(const HNSWBuilder &) = delete;
  auto operator=(const HNSWBuilder &) -> HNSWBuilder & = delete;
  HNSWBuilder(HNSWBuilder &&) = delete;
  auto operator=(HNSWBuilder &&) -> HNSWBuilder & = delete;

  /**
   * @brief Constructs a graph representation from the provided vector data.
   *
   * This function builds a graph using the specified number of vectors and their associated data.
   * The graph is constructed based on the distance computed between the vectors, utilizing a
   * templated threading model to optimize performance.
   *
   * @param thread_num The number of threads to use for distance computation.
   *                      Default is 1, which indicates single-threaded execution.
   * @param vec_num      The total number of vectors to be processed.
   * @param kVecData     Pointer to the array containing the vector data.
   *                     Each vector is expected to be in a contiguous memory layout.
   *
   * @note Ensure that the vector data is correctly formatted and that vec_num
   *       accurately reflects the number of vectors in kVecData to avoid
   *       out-of-bounds access.
   */
  auto build_graph(uint32_t thread_num = 1) -> std::unique_ptr<Graph<DataType, IDType>> {
    // Init the unified graph.
    auto vec_num = space_->get_data_num();
    auto graph =
        std::make_unique<Graph<DataType, IDType>>(space_->get_capacity(), max_nbrs_underlay_);

    hnsw_ = std::make_shared<HNSWImpl<DistanceSpaceType>>(space_, vec_num, max_nbrs_overlay_,
                                                          ef_construction_);
    std::atomic<int> cnt{0};

    // Build the graph by adding the node.
    Timer timer;
    hnsw_->add_point(0);

    LOG_INFO("graph->max_nodes_: {}", graph->max_nodes_);
    ThreadPool thread_pool(thread_num);
    for (IDType i = 1; i < vec_num; ++i) {
      thread_pool.enqueue([i, &cnt, &vec_num, this]() {  // Capture 'i' by value
        hnsw_->add_point(i);

        // Increment the counter and log progress outside the lambda
        int cur = cnt.fetch_add(1) + 1;  // Increment and get the current count
        if ((cur + 1) % 100000 == 0) {
          LOG_INFO("HNSW building progress: [{}/{}]", cur + 1, vec_num);
        }
      });
    }
    thread_pool.wait_until_all_tasks_completed(vec_num - 1);

    LOG_INFO("HNSW building cost: {}s\n", timer.elapsed() / 1000 / 1000);

    // {
    //   for (int i = 0; i < graph->max_nodes_; i++) {
    //     auto internal_id = hnsw_->label_lookup_[i];
    //     auto edges = reinterpret_cast<uint32_t *>(hnsw_->get_linklist0(internal_id));
    //     for (int j = 1; j <= edges[0]; ++j) {
    //       auto external_id = hnsw_->get_external_label(edges[j]);
    //     }
    //   }
    // }

    // Initialize the unified graph's memory space
    // this->init(this->node_num_, 2 * this->max_nbrs_);
    thread_pool.reset_task();
    // Copy the graph from hnsw to unified graph at level 0.
    for (IDType i = 0; i < vec_num; ++i) {
      // thread_pool.enqueue([i, this, &graph] {
      std::vector<IDType> ids(max_nbrs_underlay_, -1);
      auto internal_id = hnsw_->label_lookup_[i];
      auto edges = reinterpret_cast<uint32_t *>(hnsw_->get_linklist0(internal_id));
      for (uint32_t j = 1; j <= edges[0]; ++j) {
        auto external_id = hnsw_->get_external_label(edges[j]);
        ids[j - 1] = external_id;
        // graph->at(i, j - 1) = external_id;
      }
      graph->insert(ids.data());
      // });
    }
    // thread_pool.wait_until_all_tasks_completed(vec_num);
    LOG_DEBUG("Finish level 0 graph building.");

    // Initialize the overlay graph.
    auto overlay_graph =
        std::make_unique<OverlayGraph<IDType, IDType>>(graph->max_nodes_, graph->max_nbrs_);
    overlay_graph->ep_ = hnsw_->get_external_label(hnsw_->enterpoint_node_);
    thread_pool.reset_task();

    // Copy the overlay graph's data for unified graph.
    for (IDType i = 0; i < vec_num; ++i) {
      thread_pool.enqueue([this, i, &overlay_graph, &graph] {
        auto internal_id = hnsw_->label_lookup_[i];
        int level = hnsw_->element_levels_[internal_id];
        overlay_graph->levels_[i] = level;

        if (level > 0) {
          if (overlay_graph->lists_[i].capacity() < level * graph->max_nbrs_) {
            overlay_graph->lists_[i].reserve(level * graph->max_nbrs_);
          }
          overlay_graph->lists_[i].clear();
          overlay_graph->lists_[i].resize(level * graph->max_nbrs_, -1);
          // overlay_graph->lists_[i].assign(level * graph->max_nbrs_, -1);
          for (int j = 1; j <= level; ++j) {
            auto *edges = reinterpret_cast<IDType *>(hnsw_->get_linklist(internal_id, j));
            for (IDType k = 1; k <= edges[0]; ++k) {
              overlay_graph->at(j, i, k - 1) = hnsw_->get_external_label(edges[k]);
            }
          }
        }
      });
    }
    thread_pool.wait_until_all_tasks_completed(vec_num);

    graph->overlay_graph_ = std::move(overlay_graph);
    return graph;
  }
};

// static_assert(GraphBuilder<HNSWBuilder<RawSpace<>>>,
// "HNSWBuilder does not satisfy GraphBuilder concept");

}  // namespace alaya
