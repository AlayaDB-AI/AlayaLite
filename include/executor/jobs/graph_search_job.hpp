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

#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "../../index/graph/graph.hpp"
#include "../../space/space_concepts.hpp"
#include "../../utils/prefetch.hpp"
#include "../../utils/query_utils.hpp"
#include "job_context.hpp"
#include "space/rabitq_space.hpp"
#include "utils/log.hpp"
#include "utils/rabitq_utils/search_utils/buffer.hpp"
#include "utils/rabitq_utils/search_utils/hashset.hpp"

#if defined(__linux__)
  #include "coro/task.hpp"
#endif

namespace alaya {

template <typename DistanceSpaceType,
          typename DataType = typename DistanceSpaceType::DataTypeAlias,
          typename DistanceType = typename DistanceSpaceType::DistanceTypeAlias,
          typename IDType = typename DistanceSpaceType::IDTypeAlias>
  requires Space<DistanceSpaceType>
struct GraphSearchJob {
  std::shared_ptr<DistanceSpaceType> space_ = nullptr;        ///< The is a data manager interface .
  std::shared_ptr<Graph<DataType, IDType>> graph_ = nullptr;  ///< The search graph.
  std::shared_ptr<JobContext<IDType>> job_context_;           ///< The shared job context

#if defined(__AVX512F__)
  /**
   * @brief Supplement results for rabitq_search if rabitq_search failed to find enough knn
   *
   * @param result_pool
   * @param vis record whether current neighbor has been visited
   * @param query raw data pointer of the query
   */
  void rabitq_supplement_result(SearchBuffer<DistanceType> &result_pool,
                                HashBasedBooleanSet &vis,
                                const DataType *query) {
    // Add unvisited neighbors of the result nodes as supplementary result nodes
    auto data = result_pool.data();
    for (auto record : data) {
      auto *ptr_nb = space_->get_edges(record.id_);
      for (uint32_t i = 0; i < RaBitQSpace<>::kDegreeBound; ++i) {
        auto cur_neighbor = ptr_nb[i];
        if (!vis.get(cur_neighbor)) {
          vis.set(cur_neighbor);
          result_pool.insert(cur_neighbor,
                             space_->get_dist_func()(query,
                                                     space_->get_data_by_id(cur_neighbor),
                                                     space_->get_dim()));
        }
      }
      if (result_pool.is_full()) {
        break;
      }
    }
  }
#endif

  explicit GraphSearchJob(std::shared_ptr<DistanceSpaceType> space,
                          std::shared_ptr<Graph<DataType, IDType>> graph,
                          std::shared_ptr<JobContext<IDType>> job_context = nullptr)
      : space_(space), graph_(graph), job_context_(job_context) {
    if (!job_context_) {
      job_context_ = std::make_shared<JobContext<IDType>>();
    }
  }

  void rabitq_search_solo(const DataType *query, uint32_t k, IDType *ids, uint32_t ef) {
#if defined(__AVX512F__)
    if constexpr (!is_rabitq_space_v<DistanceSpaceType>) {
      throw std::invalid_argument("Only support RaBitQSpace instance!");
    }

    // init
    size_t degree_bound = RaBitQSpace<>::kDegreeBound;
    auto entry = space_->get_ep();
    mem_prefetch_l1(space_->get_data_by_id(entry), 10);
    auto q_computer = space_->get_query_computer(query);

    // sorted by estimated distance
    SearchBuffer<DistanceType> search_pool(ef);
    search_pool.insert(entry, std::numeric_limits<DistanceType>::max());
    auto vis = HashBasedBooleanSet(space_->get_data_num() / 10);
    // auto vis = DynamicBitset(space_->get_data_num());

    // sorted by exact distance (implicit rerank)
    SearchBuffer<DistanceType> res_pool(k);

    while (search_pool.has_next()) {
      auto cur_node = search_pool.pop();
      if (vis.get(cur_node)) {
        continue;
      }

      vis.set(cur_node);

      // calculate est_dist for centroid's neighbors in batch after loading centroid
      q_computer.load_centroid(cur_node);

      // scan cur_node's neighbors, insert them with estimated distances
      const IDType *cand_neighbors = space_->get_edges(cur_node);
      for (size_t i = 0; i < degree_bound; ++i) {
        auto cand_nei = cand_neighbors[i];
        DistanceType est_dist = q_computer(i);
        if (search_pool.is_full(est_dist) || vis.get(cand_nei)) {
          continue;
        }
        // try insert
        search_pool.insert(cand_nei, est_dist);

        auto next_id = search_pool.next_id();
        mem_prefetch_l2(space_->get_data_by_id(next_id), 12);
        // mem_prefetch_l2(space_->get_nei_qc_ptr(next_id), 8);
        // mem_prefetch_l2(space_->get_f_add_ptr(next_id), 6);  // 2+2+2
      }

      // implicit rerank
      res_pool.insert(cur_node, q_computer.get_exact_qr_c_dist());
    }

    if (!res_pool.is_full()) {
      rabitq_supplement_result(res_pool, vis, query);
    }
    // return result
    res_pool.copy_results_to(reinterpret_cast<uint32_t *>(ids));
#else
    throw std::runtime_error("Avx512 instruction is not supported!");
#endif
  }

  auto rabitq_search([[maybe_unused]] const DataType *query,
                     [[maybe_unused]] uint32_t k,
                     [[maybe_unused]] IDType *ids,
                     [[maybe_unused]] uint32_t ef) -> coro::task<> {
#if defined(__AVX512F__)
    if constexpr (!is_rabitq_space_v<DistanceSpaceType>) {
      throw std::invalid_argument("Only support RaBitQSpace instance!");
    }

    // init
    size_t degree_bound = RaBitQSpace<>::kDegreeBound;
    auto entry = space_->get_ep();
    mem_prefetch_l1(space_->get_data_by_id(entry), 10);
    auto q_computer = space_->get_query_computer(query);

    // sorted by estimated distance
    SearchBuffer<DistanceType> search_pool(ef);
    search_pool.insert(entry, std::numeric_limits<DistanceType>::max());

    // sorted by exact distance (implicit rerank)
    SearchBuffer<DistanceType> res_pool(k);
    auto vis = HashBasedBooleanSet(space_->get_data_num() / 10);

    while (search_pool.has_next()) {
      auto cur_node = search_pool.pop();
      if (vis.get(cur_node)) {
        continue;
      }
      vis.set(cur_node);

      // calculate est_dist for centroid's neighbors in batch after loading centroid
      q_computer.load_centroid(cur_node);

      mem_prefetch_l1(space_->get_edges(cur_node), 2);
      co_await std::suspend_always{};

      // scan cur_node's neighbors, insert them with estimated distances
      const IDType *cand_neighbors = space_->get_edges(cur_node);
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

      // implicit rerank
      res_pool.insert(cur_node, q_computer.get_exact_qr_c_dist());
    }

    if (!res_pool.is_full()) {
      LOG_DEBUG("Failed to return enough knn, res_pool current size: {}", res_pool.size());
      rabitq_supplement_result(res_pool, vis, query);
      LOG_DEBUG("Finished supplementing result, res_pool current size: {}", res_pool.size());
    }
    // return result
    res_pool.copy_results_to(reinterpret_cast<uint32_t *>(ids));
    co_return;
#else
    throw std::runtime_error("Avx512 instruction is not supported!");
#endif
  }

#if defined(__linux__)
  auto search(DataType *query, uint32_t k, IDType *ids, uint32_t ef) -> coro::task<> {
    auto query_computer = space_->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(space_->get_data_num(), ef);
    graph_->initialize_search(pool, query_computer);

    space_->prefetch_by_address(query);

    while (pool.has_next()) {
      auto u = pool.pop();

      mem_prefetch_l1(graph_->edges(u), graph_->max_nbrs_ * sizeof(IDType) / 64);
      co_await std::suspend_always{};

      for (uint32_t i = 0; i < graph_->max_nbrs_; ++i) {
        auto v = graph_->at(u, i);

        if (v == static_cast<IDType>(-1)) {
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

      for (uint32_t i = 0; i < graph_->max_nbrs_; ++i) {
        auto v = graph_->at(u, i);

        if (v == static_cast<IDType>(-1)) {
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
      for (uint32_t i = 0; i < graph_->max_nbrs_; ++i) {
        auto v = graph_->at(u, i);

        if (v == static_cast<IDType>(-1)) {
          break;
        }

        if (pool.vis_.get(v)) {
          continue;
        }
        pool.vis_.set(v);

        auto jump_prefetch = i + 3;
        if (jump_prefetch < graph_->max_nbrs_) {
          auto prefetch_id = graph_->at(u, jump_prefetch);
          if (prefetch_id != static_cast<IDType>(-1)) {
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

  void search_solo(DataType *query, uint32_t k, IDType *ids, DistanceType *distances, uint32_t ef) {
    auto query_computer = space_->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(space_->get_data_num(), ef);
    graph_->initialize_search(pool, query_computer);

    while (pool.has_next()) {
      auto u = pool.pop();
      for (uint32_t i = 0; i < graph_->max_nbrs_; ++i) {
        auto v = graph_->at(u, i);

        if (v == static_cast<IDType>(-1)) {
          break;
        }

        if (pool.vis_.get(v)) {
          continue;
        }
        pool.vis_.set(v);

        auto jump_prefetch = i + 3;
        if (jump_prefetch < graph_->max_nbrs_) {
          auto prefetch_id = graph_->at(u, jump_prefetch);
          if (prefetch_id != static_cast<IDType>(-1)) {
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

  void search_solo_updated(DataType *query, uint32_t k, IDType *ids, uint32_t ef) {
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
      for (uint32_t i = 0; i < graph_->max_nbrs_; ++i) {
        auto v = graph_->at(u, i);

        if (v == static_cast<IDType>(-1)) {
          break;
        }

        if (pool.vis_.get(v)) {
          continue;
        }
        pool.vis_.set(v);

        auto jump_prefetch = i + 3;
        if (jump_prefetch < graph_->max_nbrs_) {
          auto prefetch_id = graph_->at(u, jump_prefetch);
          if (prefetch_id != static_cast<IDType>(-1)) {
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
