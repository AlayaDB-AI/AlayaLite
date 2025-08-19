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

#include <cstddef>
#include <cstdint>
#include <memory>

#include "../../index/graph/graph.hpp"
#include "../../space/space_concepts.hpp"
#include "../../utils/prefetch.hpp"
#include "../../utils/query_utils.hpp"
#include "coro/task.hpp"
#include "index/graph/graph_refiner.hpp"
#include "job_context.hpp"
#include "space/rabitq_space.hpp"
#include "utils/rbq_utils/search_utils/buffer.hpp"
#include "utils/rbq_utils/search_utils/hashset.hpp"

namespace alaya {

template <typename DistanceSpaceType, typename DataType = DistanceSpaceType::DataTypeAlias,
          typename DistanceType = DistanceSpaceType::DistanceTypeAlias,
          typename IDType = DistanceSpaceType::IDTypeAlias>
  requires Space<DistanceSpaceType, DataType, DistanceType, IDType>
struct GraphSearchJob {
  std::shared_ptr<DistanceSpaceType> space_ = nullptr;        ///< The is a data manager interface .
  std::shared_ptr<Graph<DataType, IDType>> graph_ = nullptr;  ///< The search graph.
  std::shared_ptr<JobContext<IDType>> job_context_;           ///< The shared job context

  explicit GraphSearchJob(std::shared_ptr<DistanceSpaceType> space,
                          std::shared_ptr<Graph<DataType, IDType>> graph,
                          std::shared_ptr<JobContext<IDType>> job_context = nullptr)
      : space_(space), graph_(graph) {
    if (!job_context_) {
      job_context_ = std::make_shared<JobContext<IDType>>();
    }
  }

  void rabitq_search_optimized(const DataType *query, uint32_t k, IDType *ids, uint32_t ef) {
    static_assert(is_rbqspace_v<DistanceSpaceType>, "Only support RBQSpace instance!");
    // init
    size_t degree_bound = RBQSpace<>::kDegreeBound;
    auto entry = graph_->get_ep();
    mem_prefetch_l1(space_->get_data_by_id(entry), 10);
    auto q_computer = space_->get_query_computer(query);

    // sorted by estimated distance
    SearchBuffer<DistanceType> search_pool(ef);
    search_pool.insert(entry, std::numeric_limits<DistanceType>::max());

    // sorted by exact distance (implict rerank)
    SearchBuffer res_pool(k); 
    auto vis = HashBasedBooleanSet(space_->get_data_num() / 10);

    while (search_pool.has_next()) {
      auto cur_node = search_pool.pop();
      if (vis.get(cur_node)) {
        continue;
      }
      vis.set(cur_node);

      // calculate est_dist for centroid's neighbors in batch after loading centroid
      const IDType *cand_neighbors = graph_->edges(cur_node);
      q_computer.load_centroid(cur_node, cand_neighbors);

      // scan cur_node's neighbors, insert them with estimated distances
      for (size_t i = 0; i < degree_bound; ++i) {
        auto cand_nei = cand_neighbors[i];
        DistanceType est_dist = q_computer(i);
        if (search_pool.is_full(est_dist) || vis.get(cand_nei)) {
          continue;
        }
        // try insert
        search_pool.insert(cand_nei, est_dist);
        mem_prefetch_l2(space_->get_data_by_id(search_pool.next_id()), 10);
      }

      // implict rerank
      res_pool.insert(cur_node, q_computer.get_exact_qr_c_dist());
    }

    if (!res_pool.is_full()) {
      LOG_INFO("Failed to return enough knn, res_pool current size: {}", res_pool.size());
      /// todo: supplement result if necessary
    }
    // return result
    res_pool.copy_results_to(ids);
  }

  void rabitq_search_solo(const DataType *query, uint32_t k, IDType *ids, uint32_t ef) {
    static_assert(is_rbqspace_v<DistanceSpaceType>, "Only support RBQSpace instance!");

    // init
    size_t degree_bound = RBQSpace<>::kDegreeBound;
    auto entry = graph_->get_ep();
    mem_prefetch_l1(space_->get_data_by_id(entry), 10);
    auto q_computer = space_->get_query_computer(query);

    // sorted by estimated distance
    LinearPool<DistanceType, IDType> search_pool(space_->get_data_num(), ef);
    search_pool.insert(entry, std::numeric_limits<DistanceType>::max());

    // sorted by exact distance (implict rerank)
    LinearPool<DistanceType, IDType> res_pool(space_->get_data_num(), k);

    while (search_pool.has_next()) {
      auto cur_node = search_pool.pop();
      if (search_pool.vis_.get(cur_node)) {
        continue;
      }
      search_pool.vis_.set(cur_node);

      // calculate est_dist for centroid's neighbors in batch after loading centroid
      const IDType *cand_neighbors = graph_->edges(cur_node);
      q_computer.load_centroid(cur_node, cand_neighbors);

      // scan cur_node's neighbors, insert them with estimated distances
      for (size_t i = 0; i < degree_bound; ++i) {
        auto cand_nei = cand_neighbors[i];
        DistanceType est_dist = q_computer(i);
        if (!search_pool.small_enough(est_dist)||search_pool.vis_.get(cand_nei)) {
          continue;
        }
        // try insert
        search_pool.insert(cand_nei, est_dist);
        mem_prefetch_l2(space_->get_data_by_id(search_pool.next_id()), 10);
      }
      // implict rerank
      res_pool.insert(cur_node, q_computer.get_exact_qr_c_dist());
    }

    if (!res_pool.is_full()) {
      LOG_INFO("Failed to return enough knn, res_pool current size: {}", res_pool.size());
      /// todo: supplement result if necessary
    }

    // load result
    for (int i = 0; i < res_pool.size(); i++) {
      ids[i] = res_pool.id(i);
    }
  }

  /// todo: to be refined
  auto rabitq_search(const DataType *query, uint32_t k, IDType *ids, uint32_t ef) -> coro::task<> {
    static_assert(is_rbqspace_v<DistanceSpaceType>, "Only support RBQSpace instance!");

    // init
    auto entry = graph_->get_ep();
    mem_prefetch_l1(space_->get_data_by_id(entry), 10);
    space_->prefetch_by_address(query);
    co_await std::suspend_always{};
    auto q_computer = space_->get_query_computer(query);

    // sorted by estimated distance
    LinearPool<DistanceType, IDType> search_pool(space_->get_data_num(), ef);
    search_pool.insert(entry, std::numeric_limits<DistanceType>::max());

    // sorted by exact distance (implict rerank)
    LinearPool<DistanceType, IDType> res_pool(space_->get_data_num(), k);

    while (search_pool.has_next()) {
      auto cur_node = search_pool.pop();
      if (search_pool.vis_.get(cur_node)) {
        continue;
      }
      search_pool.vis_.set(cur_node);

      // calculate est_dist for centroid's neighbors in batch after loading centroid
      mem_prefetch_l1(graph_->edges(cur_node), graph_->max_nbrs_ * sizeof(IDType) / 64);
      co_await std::suspend_always{};
      const IDType *cand_neighbors = graph_->edges(cur_node);
      q_computer.load_centroid(cur_node, cand_neighbors);

      // scan cur_node's neighbors, insert them with estimated distances
      for (size_t i = 0; i < RBQSpace<>::kDegreeBound; ++i) {
        auto cand_nei = cand_neighbors[i];
        DistanceType est_dist = q_computer(i);
        if (search_pool.vis_.get(cand_nei)) {
          continue;
        }
        // try insert
        search_pool.insert(cand_nei, est_dist);
        mem_prefetch_l2(space_->get_data_by_id(search_pool.next_id()), 10);
      }

      // implict rerank
      res_pool.insert(cur_node, q_computer.get_exact_qr_c_dist());
    }

    if (!res_pool.is_full()) {
      LOG_INFO("Failed to return enough knn, res_pool current size: {}", res_pool.size());
      /// todo: supplement result if necessary
    }

    // load result
    for (int i = 0; i < res_pool.size(); i++) {
      ids[i] = res_pool.id(i);
    }
    co_return;
  }

  auto search(DataType *query, uint32_t k, IDType *ids, uint32_t ef) -> coro::task<> {
    auto query_computer = space_->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(space_->get_data_num(), ef);
    graph_->initialize_search(pool, query_computer);

    space_->prefetch_by_address(query);

    while (pool.has_next()) {
      auto u = pool.pop();

      mem_prefetch_l1(graph_->edges(u), graph_->max_nbrs_ * sizeof(IDType) / 64);
      co_await std::suspend_always{};

      for (int i = 0; i < graph_->max_nbrs_; ++i) {
        int v = graph_->at(u, i);

        if (v == -1) {
          break;
        }

        if (pool.vis_.get(v)) {
          continue;
        }
        pool.vis_.set(v);

        space_->prefetch_by_id(v);
        co_await std::suspend_always{};

        auto cur_dist = query_computer(v);
        pool.insert(v, cur_dist);
      }
    }

    for (int i = 0; i < k; i++) {
      ids[i] = pool.id(i);
    }
    co_return;
  }

  auto search(DataType *query, uint32_t k, IDType *ids, DistanceType *distances, uint32_t ef)
      -> coro::task<> {
    auto query_computer = space_->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(space_->get_data_num(), ef);
    graph_->initialize_search(pool, query_computer);

    space_->prefetch_by_address(query);

    while (pool.has_next()) {
      auto u = pool.pop();

      mem_prefetch_l1(graph_->edges(u), graph_->max_nbrs_ * sizeof(IDType) / 64);
      co_await std::suspend_always{};

      for (int i = 0; i < graph_->max_nbrs_; ++i) {
        int v = graph_->at(u, i);

        if (v == -1) {
          break;
        }

        if (pool.vis_.get(v)) {
          continue;
        }
        pool.vis_.set(v);

        space_->prefetch_by_id(v);
        co_await std::suspend_always{};

        auto cur_dist = query_computer(v);
        pool.insert(v, cur_dist);
      }
    }

    for (int i = 0; i < k; i++) {
      ids[i] = pool.id(i);
      distances[i] = pool.dist(i);
    }
    co_return;
  }

  void search_solo(DataType *query, uint32_t k, IDType *ids, uint32_t ef) {
    auto query_computer = space_->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(space_->get_data_num(), ef);
    graph_->initialize_search(pool, query_computer);

    while (pool.has_next()) {
      auto u = pool.pop();
      for (int i = 0; i < graph_->max_nbrs_; ++i) {
        int v = graph_->at(u, i);

        if (v == -1) {
          break;
        }

        if (pool.vis_.get(v)) {
          continue;
        }
        pool.vis_.set(v);

        auto jump_prefetch = i + 3;
        if (jump_prefetch < graph_->max_nbrs_) {
          auto prefetch_id = graph_->at(u, jump_prefetch);
          if (prefetch_id != -1) {
            space_->prefetch_by_id(prefetch_id);
          }
        }
        auto cur_dist = query_computer(v);
        pool.insert(v, cur_dist);
      }
    }
    for (int i = 0; i < k; i++) {
      ids[i] = pool.id(i);
    }
  }

  void search_solo_updated(DataType *query, uint32_t k, IDType *ids, uint32_t ef) {
    auto query_computer = space_->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(space_->get_data_num(), ef);
    graph_->initialize_search(pool, query_computer);

    while (pool.has_next()) {
      auto u = pool.pop();
      if (job_context_->removed_node_nbrs_.find(u) != job_context_->removed_node_nbrs_.end()) {
        for (auto &second_hop_nbr : job_context_->removed_node_nbrs_.at(u)) {
          if (pool.vis_.get(u)) {
            continue;
          }
          pool.vis_.set(u);
          auto dist = query_computer(u);
          pool.insert(u, dist);
        }
        continue;
      }
      for (int i = 0; i < graph_->max_nbrs_; ++i) {
        int v = graph_->at(u, i);

        if (v == -1) {
          break;
        }

        if (pool.vis_.get(v)) {
          continue;
        }
        pool.vis_.set(v);

        auto jump_prefetch = i + 3;
        if (jump_prefetch < graph_->max_nbrs_) {
          auto prefetch_id = graph_->at(u, jump_prefetch);
          if (prefetch_id != -1) {
            space_->prefetch_by_id(prefetch_id);
          }
        }
        auto cur_dist = query_computer(v);
        pool.insert(v, cur_dist);
      }
    }
    for (int i = 0; i < k; i++) {
      ids[i] = pool.id(i);
    }
  }
};

}  // namespace alaya
