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
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../index/graph/graph.hpp"
#include "../../space/space_concepts.hpp"
#include "../../utils/prefetch.hpp"
#include "../../utils/query_utils.hpp"
#include "job_context.hpp"
#include "space/rabitq_space.hpp"
#include "utils/log.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/rabitq_utils/search_utils/buffer.hpp"
#include "utils/rabitq_utils/search_utils/hashset.hpp"
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
  std::unique_ptr<HashSetPool> visited_pool_;                 ///< Pool of visited sets for rabitq

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
  void rabitq_supplement_result(SearchBuffer<DistanceType> &result_pool,
                                HashBasedBooleanSet &vis,
                                const DataType *query) {
    auto *sp = space_.get();
    auto dist_func = sp->get_dist_func();
    auto dim = sp->get_dim();
    // Add unvisited neighbors of the result nodes as supplementary result nodes
    auto data = result_pool.data();
    for (auto record : data) {
      auto *ptr_nb = sp->get_edges(record.id_);
      for (uint32_t i = 0; i < RaBitQSpace<>::kDegreeBound; ++i) {
        auto cur_neighbor = ptr_nb[i];
        if (!vis.get(cur_neighbor)) {
          vis.set(cur_neighbor);
          result_pool.insert(cur_neighbor, dist_func(query, sp->get_data_by_id(cur_neighbor), dim));
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
                          std::shared_ptr<JobContext<IDType>> job_context = nullptr,
                          std::shared_ptr<BuildSpaceType> build_space = nullptr)
      : space_(space), graph_(graph), job_context_(job_context), build_space_(build_space) {
    if (!job_context_) {
      job_context_ = std::make_shared<JobContext<IDType>>();
    }
    // If rerank is needed but build_space is not provided, throw exception
    if constexpr (kNeedsRerank) {
      if (build_space_ == nullptr) {
        throw std::invalid_argument(
            "build_space is required when SearchSpaceType != BuildSpaceType");
      }
    }
    // Initialize visited list pool for rabitq search
    if constexpr (is_rabitq_space_v<DistanceSpaceType>) {
      visited_pool_ = std::make_unique<HashSetPool>(1, space_->get_data_num());
    }
  }

  /**
   * @brief Pre-filter using scalar index for simple single-field filter
   *
   * For a simple filter (single condition on an indexed field), uses the field index
   * to quickly identify matching IDs. Sets all bits in vis first, then resets only
   * the matching IDs (marking them as unvisited/valid for search).
   *
   * @param vis DynamicBitset to mark valid IDs (matching filter)
   * @param filter The metadata filter
   * @param filter_vec Output vector to store matching IDs
   * @return true if pre-filter was applied, false if filter is not simple or field not indexed
   */
  auto pre_filter(DynamicBitset &vis, const MetadataFilter &filter, std::vector<IDType> &filter_vec)
      -> bool {
    // Empty filter matches all records, no filtering needed
    if (filter.is_empty()) {
      return true;
    }

    // Check if filter is simple: one condition, no sub_filters
    // TODO(ljh): add support for more complex filters
    if (filter.conditions.size() != 1 || !filter.sub_filters.empty()) {
      return false;
    }

    const auto &cond = filter.conditions[0];
    auto *storage = space_->get_scalar_storage();
    if (storage == nullptr) {
      return false;
    }

    // Check if field is indexed
    const auto &indexed_fields = storage->config().indexed_fields_;
    bool is_indexed =
        std::find(indexed_fields.begin(), indexed_fields.end(), cond.field) != indexed_fields.end();
    if (!is_indexed) {
      LOG_WARN("Field '{}' is not indexed, cannot pre-filter.", cond.field);
      return false;
    }

    // Get matching IDs based on operator
    switch (cond.op) {
      case FilterOp::EQ:
        filter_vec = storage->get_ids_by_field_value(cond.field, cond.value);
        break;
      case FilterOp::GE: {
        if (std::holds_alternative<int64_t>(cond.value)) {
          filter_vec =
              storage->get_ids_by_int_range(cond.field, std::get<int64_t>(cond.value), INT64_MAX);
        } else if (std::holds_alternative<double>(cond.value)) {
          filter_vec = storage->get_ids_by_double_range(cond.field,
                                                        std::get<double>(cond.value),
                                                        std::numeric_limits<double>::max());
        } else {
          return false;
        }
        break;
      }
      case FilterOp::GT: {
        if (std::holds_alternative<int64_t>(cond.value)) {
          filter_vec = storage->get_ids_by_int_range(cond.field,
                                                     std::get<int64_t>(cond.value) + 1,
                                                     INT64_MAX);
        } else if (std::holds_alternative<double>(cond.value)) {
          // For double, use nextafter to get the next representable value
          filter_vec =
              storage->get_ids_by_double_range(cond.field,
                                               std::nextafter(std::get<double>(cond.value),
                                                              std::numeric_limits<double>::max()),
                                               std::numeric_limits<double>::max());
        } else {
          return false;
        }
        break;
      }
      case FilterOp::LE: {
        if (std::holds_alternative<int64_t>(cond.value)) {
          filter_vec =
              storage->get_ids_by_int_range(cond.field, INT64_MIN, std::get<int64_t>(cond.value));
        } else if (std::holds_alternative<double>(cond.value)) {
          filter_vec = storage->get_ids_by_double_range(cond.field,
                                                        std::numeric_limits<double>::lowest(),
                                                        std::get<double>(cond.value));
        } else {
          return false;
        }
        break;
      }
      case FilterOp::LT: {
        if (std::holds_alternative<int64_t>(cond.value)) {
          filter_vec = storage->get_ids_by_int_range(cond.field,
                                                     INT64_MIN,
                                                     std::get<int64_t>(cond.value) - 1);
        } else if (std::holds_alternative<double>(cond.value)) {
          filter_vec =
              storage
                  ->get_ids_by_double_range(cond.field,
                                            std::numeric_limits<double>::lowest(),
                                            std::nextafter(std::get<double>(cond.value),
                                                           std::numeric_limits<double>::lowest()));
        } else {
          return false;
        }
        break;
      }
      default:
        return false;  // Unsupported operator for index
    }

    // Set all bits (mark all as visited/invalid)
    vis.set_all();

    // Reset matching IDs (mark them as unvisited/valid for search)
    for (auto id : filter_vec) {
      vis.reset(id);
    }

    // LOG_INFO("pre_filter: {} matching IDs for field '{}' with op {}",
    //          filter_vec.size(),
    //          cond.field,
    //          static_cast<int>(cond.op));
    return true;
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

  // clang-format off
  auto rerank(std::vector<IDType> &src,
              IDType *ids,
              uint32_t ef,
              uint32_t topk,
              auto dist_compute,
              std::string *res
              ) -> uint32_t
    requires(DistanceSpaceType::has_scalar_data) {
    // clang-format on
    std::priority_queue<std::pair<DistanceType, IDType>,
                        std::vector<std::pair<DistanceType, IDType>>,
                        std::greater<>>
        pq;

    // Calculate distances for all candidates
    for (size_t i = 0; i < ef; i++) {
      auto node = src[i];
      pq.push({dist_compute(node), node});
    }

    uint32_t result_count = std::min(topk, static_cast<uint32_t>(pq.size()));
    uint32_t actual_count = 0;
    for (uint32_t i = 0; i < result_count; i++) {
      if (pq.empty()) {
        break;
      }
      ids[i] = pq.top().second;
      pq.pop();
      actual_count++;
    }
    auto *storage = space_->get_scalar_storage();
    auto item_ids = storage->batch_get_item_id_only(std::vector<IDType>(ids, ids + actual_count));
    for (uint32_t i = 0; i < actual_count; i++) {
      res[i] = std::move(item_ids[i]);
    }

    return result_count;
  }

  auto rabitq_hybrid_search(const DataType *query,
                            uint32_t k,
                            IDType *ids,
                            uint32_t ef,
                            const MetadataFilter &filter,
                            std::string *res,
                            int reseed_time = 3) -> coro::task<> {
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

    auto q_computer = sp->get_query_computer(query);

    // sorted by estimated distance
    SearchBuffer<DistanceType> search_pool(ef);

    // Use DynamicBitset to support pre_filter optimization
    DynamicBitset vis(sp->get_data_num());

    // Try pre_filter: if successful, vis has all non-matching IDs marked as visited
    std::vector<IDType> filter_vec;
    bool pre_filtered = pre_filter(vis, filter, filter_vec);
    if (!pre_filtered) {
      throw std::runtime_error("please remember to construct scalar index.");
    }

    auto entry = sp->get_ep();
    if (!vis.get(entry)) {
      search_pool.insert(entry, std::numeric_limits<DistanceType>::max());
      mem_prefetch_l1(sp->get_data_by_id(entry), 10);
      co_await std::suspend_always{};
    }

    // sorted by exact distance (implicit rerank)
    SearchBuffer<DistanceType> res_pool(k);

    int filter_vec_pos = 0;
    int reseed_round = 0;
    auto should_reseed = [&]() {
      return filter_vec_pos != filter_vec.size() &&
             (reseed_round != reseed_time || !res_pool.is_full());
    };

    while (search_pool.has_next() || should_reseed()) {
      if (!search_pool.has_next()) {
        // re-seed
        reseed_round++;
        search_pool.clear();
        for (; filter_vec_pos < filter_vec.size(); ++filter_vec_pos) {
          auto id = filter_vec[filter_vec_pos];
          if (!vis.get(id)) {
            search_pool.insert(id, std::numeric_limits<DistanceType>::max());
            ++filter_vec_pos;  // Move to next position before break
            break;             // candidate found, break to start search immediately
          }
        }
        // If we've exhausted the filter_vec and still have no candidates, break to avoid invalid
        // pop
        if (filter_vec_pos == filter_vec.size() && !search_pool.has_next()) {
          break;
        }
      }

      auto cur_node = search_pool.pop();
      if (vis.get(cur_node)) {
        continue;
      }
      vis.set(cur_node);

      // calculate est_dist for centroid's neighbors in batch after loading centroid
      q_computer.load_centroid(cur_node);

      mem_prefetch_l1(sp->get_edges(cur_node), 2);
      co_await std::suspend_always{};

      // scan cur_node's neighbors, insert them with estimated distances
      const IDType *cand_neighbors = sp->get_edges(cur_node);
      for (size_t i = 0; i < degree_bound; ++i) {
        auto cand_nei = cand_neighbors[i];
        DistanceType est_dist = q_computer(i);
        if (search_pool.is_full(est_dist) || vis.get(cand_nei)) {
          continue;
        }
        // try insert
        search_pool.insert(cand_nei, est_dist);
        mem_prefetch_l2(sp->get_data_by_id(search_pool.next_id()), 10);
        co_await std::suspend_always{};
      }

      // implicit rerank
      res_pool.insert(cur_node, q_computer.get_exact_qr_c_dist());
    }

    // return result
    res_pool.copy_results_to(reinterpret_cast<uint32_t *>(ids));
    auto res_size = res_pool.size();
    auto *storage = sp->get_scalar_storage();
    auto item_ids = storage->batch_get_item_id_only(std::vector<IDType>(ids, ids + res_size));
    for (size_t i = 0; i < res_size; ++i) {
      res[i] = std::move(item_ids[i]);
    }
    if (res_size < k) {
      LOG_INFO("not enough result, current result number:{}, required topk:{}", res_size, k);
    }
    co_return;
#else
    throw std::runtime_error("Avx512 instruction is not supported!");
#endif
  }

  void rabitq_hybrid_search_solo(const DataType *query,
                                 uint32_t k,
                                 IDType *ids,
                                 uint32_t ef,
                                 const MetadataFilter &filter,
                                 std::string *res,
                                 int reseed_time = 3) {
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
    auto q_computer = sp->get_query_computer(query);

    // sorted by estimated distance
    SearchBuffer<DistanceType> search_pool(ef);

    // Use DynamicBitset to support pre_filter optimization
    DynamicBitset vis(sp->get_data_num());

    // Try pre_filter: if successful, vis has all non-matching IDs marked as visited
    std::vector<IDType> filter_vec;
    bool pre_filtered = pre_filter(vis, filter, filter_vec);
    if (!pre_filtered) {
      throw std::runtime_error("please remember to construct scalar index.");
    }

    // Insert entry point if it matches filter
    auto entry = sp->get_ep();
    if (!vis.get(entry)) {
      search_pool.insert(entry, std::numeric_limits<DistanceType>::max());
      mem_prefetch_l1(sp->get_data_by_id(entry), 10);
    }

    // sorted by exact distance (implicit rerank)
    SearchBuffer<DistanceType> res_pool(k);

    int filter_vec_pos = 0;
    int reseed_round = 0;
    auto should_reseed = [&]() {
      return filter_vec_pos != filter_vec.size() &&
             (reseed_round != reseed_time || !res_pool.is_full());
    };

    while (search_pool.has_next() || should_reseed()) {
      if (!search_pool.has_next()) {
        // re-seed
        reseed_round++;
        search_pool.clear();
        for (; filter_vec_pos < filter_vec.size(); ++filter_vec_pos) {
          auto id = filter_vec[filter_vec_pos];
          if (!vis.get(id)) {
            search_pool.insert(id, std::numeric_limits<DistanceType>::max());
            ++filter_vec_pos;  // Move to next position before break
            break;             // candidate found, break to start search immediately
          }
        }
        // If we've exhausted the filter_vec and still have no candidates, break to avoid invalid
        // pop
        if (filter_vec_pos == filter_vec.size() && !search_pool.has_next()) {
          break;
        }
      }

      auto cur_node = search_pool.pop();
      if (vis.get(cur_node)) {
        continue;
      }
      vis.set(cur_node);

      // calculate est_dist for centroid's neighbors in batch after loading centroid
      q_computer.load_centroid(cur_node);

      // scan cur_node's neighbors, insert them with estimated distances
      const IDType *cand_neighbors = sp->get_edges(cur_node);
      for (size_t i = 0; i < degree_bound; ++i) {
        auto cand_nei = cand_neighbors[i];
        DistanceType est_dist = q_computer(i);
        if (search_pool.is_full(est_dist) || vis.get(cand_nei)) {
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

    // return result
    res_pool.copy_results_to(reinterpret_cast<uint32_t *>(ids));
    auto res_size = res_pool.size();
    auto *storage = sp->get_scalar_storage();
    auto item_ids = storage->batch_get_item_id_only(std::vector<IDType>(ids, ids + res_size));
    for (size_t i = 0; i < res_size; ++i) {
      res[i] = std::move(item_ids[i]);
    }
    if (res_size < k) {
      LOG_INFO("not enough result, current result number:{}, required topk:{}", res_size, k);
    }
#else
    throw std::runtime_error("Avx512 instruction is not supported!");
#endif
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
      rabitq_supplement_result(res_pool, *vis, query);
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
    auto *vis = visited_pool_->acquire();

    while (search_pool.has_next()) {
      auto cur_node = search_pool.pop();
      if (vis->get(cur_node)) {
        continue;
      }
      vis->set(cur_node);

      // calculate est_dist for centroid's neighbors in batch after loading centroid
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
        // try insert
        search_pool.insert(cand_nei, est_dist);
        mem_prefetch_l2(sp->get_data_by_id(search_pool.next_id()), 10);
        co_await std::suspend_always{};
      }

      // implicit rerank
      res_pool.insert(cur_node, q_computer.get_exact_qr_c_dist());
    }

    if (!res_pool.is_full()) [[unlikely]] {
      rabitq_supplement_result(res_pool, *vis, query);
    }

    visited_pool_->release(vis);

    // return result
    res_pool.copy_results_to(reinterpret_cast<uint32_t *>(ids));

    co_return;
#else
    throw std::runtime_error("Avx512 instruction is not supported!");
#endif
  }

  // clang-format off
  void hybrid_search_solo(DataType *query,
                          IDType *ids,
                          uint32_t topk,
                          uint32_t ef,
                          const MetadataFilter &filter,
                          std::string *res,
                          int reseed_time = 3)
    requires(DistanceSpaceType::has_scalar_data) {
    // clang-format on
    if (ef < topk) {
      throw std::invalid_argument("ef must be >= topk");
    }

    auto *sp = space_.get();
    auto *gr = graph_.get();
    auto query_computer = sp->get_query_computer(query);

    // Use LinearPool with built-in DynamicBitset
    LinearPool<DistanceType, IDType> pool(sp->get_data_num(), ef);

    // Try pre_filter: get list of matching IDs
    std::vector<IDType> filter_vec;
    bool pre_filtered = pre_filter(pool.vis_, filter, filter_vec);
    if (!pre_filtered) {
      throw std::runtime_error("please remember to construct scalar index.");
    }

    // Initialize search with entry point
    gr->initialize_search(pool, query_computer);

    size_t filter_vec_pos = 0;  // Track position in filter_vec
    int reseed_round = 0;
    auto should_reseed = [&]() {
      return filter_vec_pos < filter_vec.size() && reseed_round < reseed_time;
    };

    while (pool.has_next() || should_reseed()) {
      // If search_pool is empty but we have more filtered IDs, insert a batch
      if (!pool.has_next()) {
        reseed_round++;
        size_t batch_size =
            std::min<size_t>(topk, filter_vec.size() - filter_vec_pos);  // arbitrary batch size
        for (size_t i = 0; i < batch_size; ++i) {
          auto id = filter_vec[filter_vec_pos + i];
          if (!pool.vis_.get(id)) {  // Not visited yet
            pool.insert(id, query_computer(id));
            pool.vis_.set(id);
          }
        }
        filter_vec_pos += batch_size;
        if (!pool.has_next()) {
          continue;
        }
      }

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

    // Copy ef candidates from pool
    std::vector<IDType> candidates(ef);
    for (uint32_t i = 0; i < ef; i++) {
      candidates[i] = pool.id(i);
    }

    // Rerank candidates with exact distances
    auto result_count =
        rerank(candidates, ids, ef, topk, build_space_->get_query_computer(query), res);

    if (result_count < topk) {
      LOG_INFO("hybrid_search_solo: Only found {} results, requested {}", result_count, topk);
    }
  }

  // clang-format off
  /**
   * @brief Hybrid search with brute-force exact distance computation
   *
   * Pre-filters using scalar index, then computes exact distances from
   * build_space_ raw data for all matching candidates to find topk.
   *
   * @param query Query vector
   * @param ids Output array for topk result IDs
   * @param topk Number of results to return
   * @param filter Metadata filter for pre-filtering
   * @param res Output array for item_id strings
   */
  void hybrid_search_brute_force_solo(const DataType *query,
                                       IDType *ids,
                                       uint32_t topk,
                                       const MetadataFilter &filter,
                                       std::string *res)
    requires(DistanceSpaceType::has_scalar_data) {
    // clang-format on
    auto *sp = space_.get();

    // Pre-filter to get matching IDs
    DynamicBitset vis(sp->get_data_num());
    std::vector<IDType> filter_vec;
    bool pre_filtered = pre_filter(vis, filter, filter_vec);
    if (!pre_filtered) {
      throw std::runtime_error("please remember to construct scalar index.");
    }

    // Brute-force: compute exact distance for all filtered candidates
    std::priority_queue<std::pair<DistanceType, IDType>,
                        std::vector<std::pair<DistanceType, IDType>>,
                        std::greater<>>
        pq;

    if constexpr (is_rabitq_space_v<DistanceSpaceType>) {
      // For RaBitQ, use search_space directly
      auto dist_func = sp->get_dist_func();
      auto dim = sp->get_dim();
      for (auto id : filter_vec) {
        pq.push({dist_func(query, sp->get_data_by_id(id), dim), id});
      }
    } else {
      // For quantized spaces (SQ4, SQ8), use build_space
      auto dist_func = build_space_->get_dist_func();
      auto dim = build_space_->get_dim();
      for (auto id : filter_vec) {
        pq.push({dist_func(query, build_space_->get_data_by_id(id), dim), id});
      }
    }

    uint32_t result_count = std::min(topk, static_cast<uint32_t>(pq.size()));
    for (uint32_t i = 0; i < result_count; i++) {
      ids[i] = pq.top().second;
      pq.pop();
    }

    // Get item_ids from scalar storage
    auto *storage = sp->get_scalar_storage();
    auto item_ids = storage->batch_get_item_id_only(std::vector<IDType>(ids, ids + result_count));
    for (uint32_t i = 0; i < result_count; i++) {
      res[i] = std::move(item_ids[i]);
    }

    if (result_count < topk) {
      LOG_INFO("hybrid_search_brute_force: Only found {} results, requested {}",
               result_count,
               topk);
    }
  }

#if defined(__linux__)
  auto hybrid_search(DataType *query,
                     IDType *ids,
                     uint32_t topk,
                     uint32_t ef,
                     const MetadataFilter &filter,
                     std::string *res,
                     int reseed_time = 3) -> coro::task<> {
    if (ef < topk) {
      throw std::invalid_argument("ef must be >= topk");
    }
    auto *sp = space_.get();
    auto *gr = graph_.get();
    sp->prefetch_by_address(query);

    LinearPool<DistanceType, IDType> pool(sp->get_data_num(), ef);
    // Try pre_filter: get list of matching IDs
    std::vector<IDType> filter_vec;
    bool pre_filtered = pre_filter(pool.vis_, filter, filter_vec);
    if (!pre_filtered) {
      throw std::runtime_error("please remember to construct scalar index.");
    }

    auto query_computer = sp->get_query_computer(query);

    gr->initialize_search(pool, query_computer);

    size_t filter_vec_pos = 0;  // Track position in filter_vec
    int reseed_round = 0;
    auto should_reseed = [&]() {
      return filter_vec_pos < filter_vec.size() && reseed_round < reseed_time;
    };

    while (pool.has_next() || should_reseed()) {
      // If search_pool is empty but we have more filtered IDs, insert a batch
      if (!pool.has_next()) {
        reseed_round++;
        size_t batch_size =
            std::min<size_t>(topk, filter_vec.size() - filter_vec_pos);  // arbitrary batch size
        for (size_t i = 0; i < batch_size; ++i) {
          auto id = filter_vec[filter_vec_pos + i];
          if (!pool.vis_.get(id)) {  // Not visited yet
            pool.insert(id, query_computer(id));
            pool.vis_.set(id);
          }
        }
        filter_vec_pos += batch_size;
        if (!pool.has_next()) {
          continue;
        }
      }

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

    std::vector<IDType> candidates(ef);
    for (uint32_t i = 0; i < ef; i++) {
      candidates[i] = pool.id(i);
    }
    // Rerank candidates with exact distances
    auto result_count =
        rerank(candidates, ids, ef, topk, build_space_->get_query_computer(query), res);

    if (result_count < topk) {
      LOG_INFO("hybrid_search: Only found {} results, requested {}", result_count, topk);
    }

    co_return;
  }

  // clang-format off
  /**
   * @brief Hybrid search with brute-force exact distance computation (coroutine version)
   *
   * Pre-filters using scalar index, then computes exact distances from
   * build_space_ raw data for all matching candidates to find topk.
   */
  auto hybrid_search_brute_force(const DataType *query,
                                  IDType *ids,
                                  uint32_t topk,
                                  const MetadataFilter &filter,
                                  std::string *res) -> coro::task<>
    requires(DistanceSpaceType::has_scalar_data) {
    // clang-format on
    auto *sp = space_.get();

    // Pre-filter to get matching IDs
    DynamicBitset vis(sp->get_data_num());
    std::vector<IDType> filter_vec;
    bool pre_filtered = pre_filter(vis, filter, filter_vec);
    if (!pre_filtered) {
      throw std::runtime_error("please remember to construct scalar index.");
    }

    // Brute-force: compute exact distance for all filtered candidates
    std::priority_queue<std::pair<DistanceType, IDType>,
                        std::vector<std::pair<DistanceType, IDType>>,
                        std::greater<>>
        pq;

    if constexpr (is_rabitq_space_v<DistanceSpaceType>) {
      // For RaBitQ, use search_space directly
      auto dist_func = sp->get_dist_func();
      auto dim = sp->get_dim();
      for (auto id : filter_vec) {
        pq.push({dist_func(query, sp->get_data_by_id(id), dim), id});
      }
    } else {
      // For quantized spaces (SQ4, SQ8), use build_space
      auto dist_func = build_space_->get_dist_func();
      auto dim = build_space_->get_dim();
      for (auto id : filter_vec) {
        pq.push({dist_func(query, build_space_->get_data_by_id(id), dim), id});
      }
    }

    uint32_t result_count = std::min(topk, static_cast<uint32_t>(pq.size()));
    for (uint32_t i = 0; i < result_count; i++) {
      ids[i] = pq.top().second;
      pq.pop();
    }

    // Get item_ids from scalar storage
    auto *storage = sp->get_scalar_storage();
    auto item_ids = storage->batch_get_item_id_only(std::vector<IDType>(ids, ids + result_count));
    for (uint32_t i = 0; i < result_count; i++) {
      res[i] = std::move(item_ids[i]);
    }

    if (result_count < topk) {
      LOG_INFO("hybrid_search_brute_force: Only found {} results, requested {}",
               result_count,
               topk);
    }

    co_return;
  }

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
};

}  // namespace alaya
