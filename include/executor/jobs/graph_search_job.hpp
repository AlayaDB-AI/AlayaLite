/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <memory>

#include "../../index/graph/graph.hpp"
#include "../../space/space_concepts.hpp"
#include "../../utils/prefetch.hpp"
#include "../../utils/query_utils.hpp"
#include "job_context.hpp"

#if defined(__linux__)
#include "coro/task.hpp"
#endif

namespace alaya {

template <typename DistanceSpaceType,
          typename DataType = typename DistanceSpaceType::DataTypeAlias,
          typename DistanceType = typename DistanceSpaceType::DistanceTypeAlias,
          typename IDType = typename DistanceSpaceType::IDTypeAlias>
  requires Space<DistanceSpaceType, DataType, DistanceType, IDType>
struct GraphSearchJob {
  std::shared_ptr<DistanceSpaceType> space_ = nullptr;  ///< The is a data manager interface .
  std::shared_ptr<Graph<DataType, IDType>> graph_ = nullptr;  ///< The search graph.
  std::shared_ptr<JobContext<IDType>> job_context_;           ///< The shared job context

  explicit GraphSearchJob(std::shared_ptr<DistanceSpaceType> space,
                          std::shared_ptr<Graph<DataType, IDType>> graph,
                          std::shared_ptr<JobContext<IDType>> job_context = nullptr)
      : space_(space), graph_(graph), job_context_(job_context) {
    if (!job_context_) {
      job_context_ = std::make_shared<JobContext<IDType>>();
    }
  }

#if defined(__linux__)
  auto search(DataType *query, uint32_t k, IDType *ids, uint32_t ef)
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

    for (uint32_t i = 0; i < k; i++) {
      ids[i] = pool.id(i);
    }
    co_return;
  }

  auto search(DataType *query, uint32_t k, IDType *ids, DistanceType *distances,
              uint32_t ef) -> coro::task<> {
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

    for (uint32_t i = 0; i < k; i++) {
      ids[i] = pool.id(i);
      distances[i] = pool.dist(i);
    }
    co_return;
  }
#endif

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
    for (uint32_t i = 0; i < k; i++) {
      ids[i] = pool.id(i);
    }
  }

  void search_solo(DataType *query, uint32_t k, IDType *ids,
                   DistanceType *distances, uint32_t ef) {
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
    for (uint32_t i = 0; i < k; i++) {
      ids[i] = pool.id(i);
      distances[i] = pool.dist(i);
    }
  }

  void search_solo_updated(DataType *query, uint32_t k, IDType *ids,
                           uint32_t ef) {
    auto query_computer = space_->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(space_->get_data_num(), ef);
    graph_->initialize_search(pool, query_computer);

    while (pool.has_next()) {
      auto u = pool.pop();
      if (job_context_->removed_node_nbrs_.count(u)) {
        for (auto &second_hop_nbr : job_context_->removed_node_nbrs_.at(u)) {
          if (pool.vis_.get(second_hop_nbr)) {
            continue;
          }
          pool.vis_.set(second_hop_nbr);
          auto dist = query_computer(second_hop_nbr);
          pool.insert(second_hop_nbr, dist);
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
    for (uint32_t i = 0; i < k; i++) {
      ids[i] = pool.id(i);
    }
  }
};

}  // namespace alaya