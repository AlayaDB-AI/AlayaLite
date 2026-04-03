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

#include <algorithm>
#include <climits>
#include <cmath>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../index/graph/graph.hpp"
#include "../../space/space_concepts.hpp"
#include "../../utils/prefetch.hpp"
#include "../../utils/query_utils.hpp"
#include "executor/search_info.hpp"
#include "executor/vector_iterator.hpp"
#include "job_context.hpp"
#include "space/rabitq_space.hpp"
#include "utils/log.hpp"
#include "utils/rabitq_utils/search_utils/buffer.hpp"
#include "utils/rabitq_utils/search_utils/visited_pool.hpp"

#if defined(__linux__)
  #include "coro/task.hpp"
#endif

namespace alaya {

template <typename DistanceSpaceType,
          typename BuildSpaceType = DistanceSpaceType,
          typename DataType = typename DistanceSpaceType::DataTypeAlias,
          typename DistanceType = typename DistanceSpaceType::DistanceTypeAlias,
          typename IDType = typename DistanceSpaceType::IDTypeAlias>
  requires Space<DistanceSpaceType> && Space<BuildSpaceType>
struct GraphSearchJob {
  std::shared_ptr<DistanceSpaceType> space_ = nullptr;     ///< Search space (may be quantized)
  std::shared_ptr<BuildSpaceType> build_space_ = nullptr;  ///< Build space (raw vectors for rerank)
  std::shared_ptr<Graph<DataType, IDType>> graph_ = nullptr;  ///< The search graph.
  std::shared_ptr<JobContext<IDType>> job_context_;           ///< The shared job context
  std::unique_ptr<EpochVisitedPool<>> visited_pool_;  ///< O(1)-clear visited pool for rabitq

  /// Compile-time flag: whether rerank is needed
  static constexpr bool kNeedsRerank = !std::is_same_v<DistanceSpaceType, BuildSpaceType>;

#if defined(__AVX512F__)
  /**
   * @brief Supplement results for rabitq_search if rabitq_search failed to find enough knn
   *
   * @param result_pool
   * @param vis record whether current neighbor has been visited
   * @param query raw data pointer of the query
   */
  template <typename VisitedSet>
  auto rabitq_supplement_result(SearchBuffer<DistanceType> &result_pool,
                                VisitedSet &vis,
                                const DataType *query) -> uint32_t {
    auto *sp = space_.get();
    auto dist_func = sp->get_dist_func();
    auto dim = sp->get_dim();
    uint32_t supplement_count = 0;
    // Add unvisited neighbors of the result nodes as supplementary result nodes
    auto data = result_pool.data();
    for (auto record : data) {
      auto *ptr_nb = sp->get_edges(record.id_);
      for (uint32_t i = 0; i < RaBitQSpace<>::kDegreeBound; ++i) {
        auto cur_neighbor = ptr_nb[i];
        if (!vis.get(cur_neighbor)) {
          vis.set(cur_neighbor);
          supplement_count += static_cast<uint32_t>(
              result_pool.insert(cur_neighbor,
                                 dist_func(query, sp->get_data_by_id(cur_neighbor), dim)));
        }
      }
      if (result_pool.is_full()) {
        break;
      }
    }
    return supplement_count;
  }
#endif

  explicit GraphSearchJob(std::shared_ptr<DistanceSpaceType> space,
                          std::shared_ptr<Graph<DataType, IDType>> graph,
                          std::shared_ptr<JobContext<IDType>> job_context = nullptr,
                          std::shared_ptr<BuildSpaceType> build_space = nullptr)
      : space_(space), build_space_(build_space), graph_(graph), job_context_(job_context) {
    if (!job_context_) {
      job_context_ = std::make_shared<JobContext<IDType>>();
    }
    // If rerank is needed but build_space is not provided, throw exception
    if constexpr (kNeedsRerank && !is_rabitq_space_v<DistanceSpaceType>) {
      if (build_space_ == nullptr) {
        throw std::invalid_argument(
            "build_space is required when SearchSpaceType != BuildSpaceType");
      }
    }
    // Initialize visited list pool for rabitq search
    if constexpr (is_rabitq_space_v<DistanceSpaceType>) {
      visited_pool_ = std::make_unique<EpochVisitedPool<>>(1, space_->get_data_num());
    }
  }
  /**
   * @brief Rerank search results using exact distances from build space
   * @param src Source ID array (ef candidates from graph search)
   * @param desc Destination ID array (topk results after rerank)
   * @param ef Number of candidates
   * @param topk Number of results to return
   * @param dist_compute Distance computer from build space
   */
  void rerank(std::vector<IDType> &src,
              IDType *desc,
              uint32_t ef,
              uint32_t topk,
              auto dist_compute) {
    std::priority_queue<std::pair<DistanceType, IDType>,
                        std::vector<std::pair<DistanceType, IDType>>,
                        std::greater<>>
        pq;
    for (size_t i = 0; i < ef; i++) {
      pq.push({dist_compute(src[i]), src[i]});
    }
    for (size_t i = 0; i < topk; i++) {
      desc[i] = pq.top().second;
      pq.pop();
    }
  }

  /**
   * @brief Rerank search results with distances using exact distances from build space
   * @param src Source ID array (ef candidates from graph search)
   * @param desc Destination ID array (topk results after rerank)
   * @param distances Output distance array
   * @param ef Number of candidates
   * @param topk Number of results to return
   * @param dist_compute Distance computer from build space
   */
  void rerank(std::vector<IDType> &src,
              IDType *desc,
              DistanceType *distances,
              uint32_t ef,
              uint32_t topk,
              auto dist_compute) {
    std::priority_queue<std::pair<DistanceType, IDType>,
                        std::vector<std::pair<DistanceType, IDType>>,
                        std::greater<>>
        pq;
    for (size_t i = 0; i < ef; i++) {
      pq.push({dist_compute(src[i]), src[i]});
    }
    for (size_t i = 0; i < topk; i++) {
      distances[i] = pq.top().first;
      desc[i] = pq.top().second;
      pq.pop();
    }
  }

  void rabitq_search_solo(const DataType *query, uint32_t k, IDType *ids, uint32_t ef) {
#if defined(__AVX512F__)
    if constexpr (!is_rabitq_space_v<DistanceSpaceType>) {
      throw std::invalid_argument("Only support RaBitQSpace instance!");
    }

    if (ef < k) {
      throw std::invalid_argument("ef must be >= k");
    }

    auto *sp = space_.get();

    // init
    size_t degree_bound = RaBitQSpace<>::kDegreeBound;
    auto entry = sp->get_ep();
    mem_prefetch_l1(sp->get_data_by_id(entry), 10);
    auto q_computer = sp->get_query_computer(query);

    // sorted by estimated distance
    SearchBuffer<DistanceType> search_pool(ef);
    search_pool.insert(entry, std::numeric_limits<DistanceType>::max());
    auto *vis = visited_pool_->acquire();

    // sorted by exact distance (implicit rerank)
    SearchBuffer<DistanceType> res_pool(k);

    while (search_pool.has_next()) {
      auto cur_node = search_pool.pop();
      if (vis->get(cur_node)) {
        continue;
      }

      vis->set(cur_node);

      // calculate est_dist for centroid's neighbors in batch after loading centroid
      q_computer.load_centroid(cur_node);

      // scan cur_node's neighbors, insert them with estimated distances
      const IDType *cand_neighbors = sp->get_edges(cur_node);
      for (size_t i = 0; i < degree_bound; ++i) {
        auto cand_nei = cand_neighbors[i];
        DistanceType est_dist = q_computer(i);
        if (search_pool.is_full(est_dist) || vis->get(cand_nei)) {
          continue;
        }

        // try insert
        search_pool.insert(cand_nei, est_dist);

        auto next_id = search_pool.next_id();
        mem_prefetch_l2(sp->get_data_by_id(next_id), 10);
      }

      // implicit rerank
      res_pool.insert(cur_node, q_computer.get_exact_qr_c_dist());
    }

    if (!res_pool.is_full()) [[unlikely]] {
      auto supplement_count = rabitq_supplement_result(res_pool, *vis, query);
      LOG_DEBUG("rabitq_search: supplement produced {} valid results", supplement_count);
    }

    visited_pool_->release(vis);

    // return result
    res_pool.copy_results_to(reinterpret_cast<uint32_t *>(ids));
#else
    throw std::runtime_error("Avx512 instruction is not supported!");
#endif
  }

  auto rabitq_search(const DataType *query, uint32_t k, IDType *ids, uint32_t ef) -> coro::task<> {
#if defined(__AVX512F__)
    if constexpr (!is_rabitq_space_v<DistanceSpaceType>) {
      throw std::invalid_argument("Only support RaBitQSpace instance!");
    }

    if (ef < k) {
      throw std::invalid_argument("ef must be >= k");
    }

    auto *sp = space_.get();

    // init
    size_t degree_bound = RaBitQSpace<>::kDegreeBound;
    auto entry = sp->get_ep();
    mem_prefetch_l1(sp->get_data_by_id(entry), 10);
    auto q_computer = sp->get_query_computer(query);

    // sorted by estimated distance
    SearchBuffer<DistanceType> search_pool(ef);
    search_pool.insert(entry, std::numeric_limits<DistanceType>::max());

    // sorted by exact distance (implicit rerank)
    SearchBuffer<DistanceType> res_pool(k);

    // record (whether a node have expanded or not) rather than (visited or not)
    auto *vis = visited_pool_->acquire();

    while (search_pool.has_next()) {
      auto cur_node = search_pool.pop();
      if (vis->get(cur_node)) {
        continue;
      }
      vis->set(cur_node);

      // calculate est_dist for centroid's neighbors in batch using exact_dist between query and
      // centroid
      q_computer.load_centroid(cur_node);

      mem_prefetch_l1(sp->get_edges(cur_node), 2);
      co_await std::suspend_always{};

      // scan cur_node's neighbors, insert them with estimated distances
      const IDType *cand_neighbors = sp->get_edges(cur_node);
      for (size_t i = 0; i < degree_bound; ++i) {
        auto cand_nei = cand_neighbors[i];
        DistanceType est_dist = q_computer(i);
        if (search_pool.is_full(est_dist) || vis->get(cand_nei)) {
          continue;
        }
        // try insert, same node may be inserted multiple times with different estimated distances,
        // but only the smallest one will be popped and expanded
        search_pool.insert(cand_nei, est_dist);
        mem_prefetch_l2(sp->get_data_by_id(search_pool.next_id()), 10);
        co_await std::suspend_always{};
      }

      // implicit rerank
      res_pool.insert(cur_node, q_computer.get_exact_qr_c_dist());
    }

    if (!res_pool.is_full()) [[unlikely]] {
      auto supplement_count = rabitq_supplement_result(res_pool, *vis, query);
      LOG_DEBUG("rabitq_search: supplement produced {} valid results", supplement_count);
    }

    visited_pool_->release(vis);

    // return result
    res_pool.copy_results_to(reinterpret_cast<uint32_t *>(ids));

    co_return;
#else
    throw std::runtime_error("Avx512 instruction is not supported!");
#endif
  }
#if defined(__linux__)
  /**
   * @brief Search for nearest neighbors (coroutine version with async prefetching)
   *
   * Performs graph-based search and returns topk results. If search space differs
   * from build space (quantized search), automatically reranks using exact distances.
   *
   * @param query Query vector
   * @param ids Output array for topk result IDs
   * @param topk Number of results to return
   * @param ef Number of candidates to explore during search (ef >= topk)
   */
  auto search(DataType *query, IDType *ids, uint32_t topk, uint32_t ef) -> coro::task<> {
    if (ef < topk) {
      throw std::invalid_argument("ef must be >= topk");
    }

    auto *sp = space_.get();
    auto *gr = graph_.get();
    std::vector<IDType> res_pool(ef);

    auto query_computer = sp->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(sp->get_data_num(), ef);
    gr->initialize_search(pool, query_computer);

    sp->prefetch_by_address(query);

    while (pool.has_next()) {
      auto u = pool.pop();

      mem_prefetch_l1(gr->edges(u), gr->max_nbrs_ * sizeof(IDType) / 64);
      co_await std::suspend_always{};

      for (uint32_t i = 0; i < gr->max_nbrs_; ++i) {
        auto v = gr->at(u, i);

        if (v == static_cast<IDType>(-1)) {
          break;
        }

        if (pool.vis_.get(v)) {
          continue;
        }
        pool.vis_.set(v);

        sp->prefetch_by_id(v);
        co_await std::suspend_always{};

        auto cur_dist = query_computer(v);
        pool.insert(v, cur_dist);
      }
    }

    // Copy ef candidates
    for (uint32_t i = 0; i < ef; i++) {
      res_pool[i] = pool.id(i);
    }

    // Rerank if needed, otherwise directly copy topk
    if constexpr (kNeedsRerank) {
      rerank(res_pool, ids, ef, topk, build_space_->get_query_computer(query));
    } else {
      std::copy(res_pool.begin(), res_pool.begin() + topk, ids);
    }

    co_return;
  }

  /**
   * @brief Search for nearest neighbors with distances (coroutine version)
   *
   * Performs graph-based search and returns topk results with distances.
   * If search space differs from build space, automatically reranks using exact distances.
   *
   * @param query Query vector
   * @param ids Output array for topk result IDs
   * @param distances Output array for topk distances
   * @param topk Number of results to return
   * @param ef Number of candidates to explore during search (ef >= topk)
   */
  auto search(DataType *query, IDType *ids, DistanceType *distances, uint32_t topk, uint32_t ef)
      -> coro::task<> {
    if (ef < topk) {
      throw std::invalid_argument("ef must be >= topk");
    }

    auto *sp = space_.get();
    auto *gr = graph_.get();
    std::vector<IDType> res_pool(ef);
    std::vector<DistanceType> dist_pool(ef);

    auto query_computer = sp->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(sp->get_data_num(), ef);
    gr->initialize_search(pool, query_computer);

    sp->prefetch_by_address(query);

    while (pool.has_next()) {
      auto u = pool.pop();

      mem_prefetch_l1(gr->edges(u), gr->max_nbrs_ * sizeof(IDType) / 64);
      co_await std::suspend_always{};

      for (uint32_t i = 0; i < gr->max_nbrs_; ++i) {
        auto v = gr->at(u, i);

        if (v == static_cast<IDType>(-1)) {
          break;
        }

        if (pool.vis_.get(v)) {
          continue;
        }
        pool.vis_.set(v);

        sp->prefetch_by_id(v);
        co_await std::suspend_always{};

        auto cur_dist = query_computer(v);
        pool.insert(v, cur_dist);
      }
    }

    // Copy ef candidates
    for (uint32_t i = 0; i < ef; i++) {
      res_pool[i] = pool.id(i);
      dist_pool[i] = pool.dist(i);
    }

    // Rerank if needed, otherwise directly copy topk
    if constexpr (kNeedsRerank) {
      rerank(res_pool, ids, distances, ef, topk, build_space_->get_query_computer(query));
    } else {
      std::copy(res_pool.begin(), res_pool.begin() + topk, ids);
      std::copy(dist_pool.begin(), dist_pool.begin() + topk, distances);
    }

    co_return;
  }
#endif
  /**
   * @brief Search for nearest neighbors (non-coroutine version)
   *
   * Performs graph-based search and returns topk results. If search space differs
   * from build space (quantized search), automatically reranks using exact distances.
   *
   * @param query Query vector
   * @param ids Output array for topk result IDs
   * @param topk Number of results to return
   * @param ef Number of candidates to explore during search (ef >= topk)
   */
  void search_solo(DataType *query, IDType *ids, uint32_t topk, uint32_t ef) {
    if (ef < topk) {
      throw std::invalid_argument("ef must be >= topk");
    }

    auto *sp = space_.get();
    auto *gr = graph_.get();
    std::vector<IDType> res_pool(ef);

    auto query_computer = sp->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(sp->get_data_num(), ef);
    gr->initialize_search(pool, query_computer);

    while (pool.has_next()) {
      auto u = pool.pop();
      for (uint32_t i = 0; i < gr->max_nbrs_; ++i) {
        auto v = gr->at(u, i);

        if (v == static_cast<IDType>(-1)) {
          break;
        }

        if (pool.vis_.get(v)) {
          continue;
        }
        pool.vis_.set(v);

        auto jump_prefetch = i + 3;
        if (jump_prefetch < gr->max_nbrs_) {
          auto prefetch_id = gr->at(u, jump_prefetch);
          if (prefetch_id != static_cast<IDType>(-1)) {
            sp->prefetch_by_id(prefetch_id);
          }
        }
        auto cur_dist = query_computer(v);
        pool.insert(v, cur_dist);
      }
    }

    // Copy ef candidates
    for (uint32_t i = 0; i < ef; i++) {
      res_pool[i] = pool.id(i);
    }

    // Rerank if needed, otherwise directly copy topk
    if constexpr (kNeedsRerank) {
      rerank(res_pool, ids, ef, topk, build_space_->get_query_computer(query));
    } else {
      std::copy(res_pool.begin(), res_pool.begin() + topk, ids);
    }
  }

  /**
   * @brief Search for nearest neighbors with distances (non-coroutine version)
   *
   * Performs graph-based search and returns topk results with distances.
   * If search space differs from build space, automatically reranks using exact distances.
   *
   * @param query Query vector
   * @param ids Output array for topk result IDs
   * @param distances Output array for topk distances
   * @param topk Number of results to return
   * @param ef Number of candidates to explore during search (ef >= topk)
   */
  void search_solo(DataType *query,
                   IDType *ids,
                   DistanceType *distances,
                   uint32_t topk,
                   uint32_t ef) {
    if (ef < topk) {
      throw std::invalid_argument("ef must be >= topk");
    }

    auto *sp = space_.get();
    auto *gr = graph_.get();
    std::vector<IDType> res_pool(ef);
    std::vector<DistanceType> dist_pool(ef);

    auto query_computer = sp->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(sp->get_data_num(), ef);
    gr->initialize_search(pool, query_computer);

    while (pool.has_next()) {
      auto u = pool.pop();
      for (uint32_t i = 0; i < gr->max_nbrs_; ++i) {
        auto v = gr->at(u, i);

        if (v == static_cast<IDType>(-1)) {
          break;
        }

        if (pool.vis_.get(v)) {
          continue;
        }
        pool.vis_.set(v);

        auto jump_prefetch = i + 3;
        if (jump_prefetch < gr->max_nbrs_) {
          auto prefetch_id = gr->at(u, jump_prefetch);
          if (prefetch_id != static_cast<IDType>(-1)) {
            sp->prefetch_by_id(prefetch_id);
          }
        }
        auto cur_dist = query_computer(v);
        pool.insert(v, cur_dist);
      }
    }

    // Copy ef candidates
    for (uint32_t i = 0; i < ef; i++) {
      res_pool[i] = pool.id(i);
      dist_pool[i] = pool.dist(i);
    }

    // Rerank if needed, otherwise directly copy topk
    if constexpr (kNeedsRerank) {
      rerank(res_pool, ids, distances, ef, topk, build_space_->get_query_computer(query));
    } else {
      std::copy(res_pool.begin(), res_pool.begin() + topk, ids);
      std::copy(dist_pool.begin(), dist_pool.begin() + topk, distances);
    }
  }
  void search_solo_updated(DataType *query, IDType *ids, uint32_t ef, uint32_t topk) {
    if (ef < topk) {
      throw std::invalid_argument("ef must be >= topk");
    }

    auto *sp = space_.get();
    auto *gr = graph_.get();
    auto query_computer = sp->get_query_computer(query);
    LinearPool<DistanceType, IDType> pool(sp->get_data_num(), ef);
    gr->initialize_search(pool, query_computer);

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
      for (uint32_t i = 0; i < gr->max_nbrs_; ++i) {
        auto v = gr->at(u, i);

        if (v == static_cast<IDType>(-1)) {
          break;
        }

        if (pool.vis_.get(v)) {
          continue;
        }
        pool.vis_.set(v);

        auto jump_prefetch = i + 3;
        if (jump_prefetch < gr->max_nbrs_) {
          auto prefetch_id = gr->at(u, jump_prefetch);
          if (prefetch_id != static_cast<IDType>(-1)) {
            sp->prefetch_by_id(prefetch_id);
          }
        }
        auto cur_dist = query_computer(v);
        pool.insert(v, cur_dist);
      }
    }
    for (uint32_t i = 0; i < topk; i++) {
      ids[i] = pool.id(i);
    }
  }

  auto make_vector_iterator(const DataType *query,
                            const SearchInfo &search_info,
                            const DynamicBitset *blocked_mask = nullptr)
      -> std::unique_ptr<VectorIterator<IDType, DistanceType>> {
    if (search_info.ef_ < search_info.topk_) {
      throw std::invalid_argument("ef must be >= topk");
    }

    if constexpr (is_rabitq_space_v<DistanceSpaceType>) {
      return std::make_unique<
          RaBitQVectorIterator<DistanceSpaceType, DataType, DistanceType, IDType>>(space_,
                                                                                   query,
                                                                                   search_info.ef_,
                                                                                   blocked_mask);
    } else {
      return std::make_unique<GraphVectorIterator<DistanceSpaceType,
                                                  BuildSpaceType,
                                                  DataType,
                                                  DistanceType,
                                                  IDType>>(space_,
                                                           build_space_,
                                                           graph_,
                                                           query,
                                                           search_info.ef_,
                                                           blocked_mask);
    }
  }

  void search_solo(DataType *query,
                   IDType *ids,
                   const SearchInfo &search_info,
                   const DynamicBitset *blocked_mask = nullptr) {
    if constexpr (is_rabitq_space_v<DistanceSpaceType>) {
      throw std::invalid_argument("Use search_solo for RaBitQSpace");
    }

    if (blocked_mask == nullptr) {
      search_solo(query, ids, search_info.topk_, search_info.ef_);
      return;
    }

    SearchBuffer<DistanceType> result_pool(search_info.topk_);
    auto iterator = make_vector_iterator(query, search_info, blocked_mask);
    while (iterator->has_next()) {
      auto candidate = iterator->next();
      if (!candidate.has_value()) {
        break;
      }
      result_pool.insert(candidate->id_, candidate->distance_);
    }

    std::fill(ids, ids + search_info.topk_, std::numeric_limits<IDType>::max());
    result_pool.copy_results_to(reinterpret_cast<uint32_t *>(ids), search_info.topk_);
  }

  void rabitq_search_solo(const DataType *query,
                          uint32_t topk,
                          IDType *ids,
                          const SearchInfo &search_info,
                          const DynamicBitset *blocked_mask = nullptr) {
    if constexpr (!is_rabitq_space_v<DistanceSpaceType>) {
      throw std::invalid_argument("Only support RaBitQSpace instance!");
    }

    if (blocked_mask == nullptr) {
      rabitq_search_solo(query, topk, ids, search_info.ef_);
      return;
    }

    SearchBuffer<DistanceType> result_pool(topk);
    auto iterator = make_vector_iterator(query, search_info, blocked_mask);
    while (iterator->has_next()) {
      auto candidate = iterator->next();
      if (!candidate.has_value()) {
        break;
      }
      result_pool.insert(candidate->id_, candidate->distance_);
    }

    std::fill(ids, ids + topk, std::numeric_limits<IDType>::max());
    result_pool.copy_results_to(reinterpret_cast<uint32_t *>(ids), topk);
  }
};

}  // namespace alaya
