// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
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
  struct QueryScratch;
  class QueryScratchPool;
  class QueryScratchLease;

  std::shared_ptr<DistanceSpaceType> space_ = nullptr;     ///< Search space (may be quantized)
  std::shared_ptr<BuildSpaceType> build_space_ = nullptr;  ///< Build space (raw vectors for rerank)
  std::shared_ptr<Graph<DataType, IDType>> graph_ = nullptr;  ///< The search graph.
  std::shared_ptr<JobContext<IDType>> job_context_;           ///< The shared job context
  std::unique_ptr<HashSetPool> visited_pool_;                 ///< Reused visited pool for rabitq
  std::unique_ptr<QueryScratchPool> query_scratch_pool_;      ///< Reused rerank buffers

  /// Compile-time flag: whether rerank is needed
  static constexpr bool kNeedsRerank = !std::is_same_v<DistanceSpaceType, BuildSpaceType>;

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
    const auto &data = result_pool.data();
    const auto seed_count = result_pool.size();
    std::vector<IDType> seed_ids;
    seed_ids.reserve(seed_count);
    for (size_t i = 0; i < seed_count; ++i) {
      seed_ids.push_back(data[i].id_);
    }

    for (const auto seed_id : seed_ids) {
      auto *ptr_nb = sp->get_edges(seed_id);
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
      visited_pool_ = std::make_unique<HashSetPool>(1, space_->get_data_num());
    }
    query_scratch_pool_ = std::make_unique<QueryScratchPool>(1);
  }

  static constexpr auto kInvalidId = std::numeric_limits<IDType>::max();
  static constexpr auto kInvalidDistance = std::numeric_limits<DistanceType>::max();

  static void fill_invalid_ids(IDType *ids, uint32_t begin, uint32_t end) {
    std::fill(ids + begin, ids + end, kInvalidId);
  }

  static void fill_invalid_distances(DistanceType *distances, uint32_t begin, uint32_t end) {
    std::fill(distances + begin, distances + end, kInvalidDistance);
  }

  struct RerankCandidate {
    DistanceType distance_;
    IDType id_;
  };

  struct QueryScratch {
    std::vector<RerankCandidate> rerank_heap_;

    void prepare(size_t topk) {
      rerank_heap_.clear();
      if (rerank_heap_.capacity() < topk) {
        rerank_heap_.reserve(topk);
      }
    }
  };

  class QueryScratchPool {
   public:
    explicit QueryScratchPool(size_t init_pool_size) {
      for (size_t i = 0; i < init_pool_size; ++i) {
        pool_.push_front(new QueryScratch());
      }
    }

    ~QueryScratchPool() {
      while (!pool_.empty()) {
        auto *ptr = pool_.front();
        pool_.pop_front();
        delete ptr;
      }
    }

    QueryScratchPool(const QueryScratchPool &) = delete;
    auto operator=(const QueryScratchPool &) -> QueryScratchPool & = delete;
    QueryScratchPool(QueryScratchPool &&) = delete;
    auto operator=(QueryScratchPool &&) -> QueryScratchPool & = delete;

    auto acquire(size_t topk) -> QueryScratch * {
      QueryScratch *res = nullptr;
      {
        std::unique_lock<std::mutex> lock(poolguard_);
        if (!pool_.empty()) {
          res = pool_.front();
          pool_.pop_front();
        } else {
          res = new QueryScratch();
        }
      }
      res->prepare(topk);
      return res;
    }

    void release(QueryScratch *scratch) {
      std::unique_lock<std::mutex> lock(poolguard_);
      pool_.push_front(scratch);
    }

   private:
    std::deque<QueryScratch *> pool_;
    std::mutex poolguard_;
  };

  class QueryScratchLease {
   public:
    QueryScratchLease(QueryScratchPool *pool, size_t topk) : pool_(pool) {
      if (pool_ != nullptr) {
        scratch_ = pool_->acquire(topk);
      }
    }

    ~QueryScratchLease() {
      if (pool_ != nullptr && scratch_ != nullptr) {
        pool_->release(scratch_);
      }
    }

    QueryScratchLease(const QueryScratchLease &) = delete;
    auto operator=(const QueryScratchLease &) -> QueryScratchLease & = delete;

    QueryScratchLease(QueryScratchLease &&other) noexcept
        : pool_(other.pool_), scratch_(other.scratch_) {
      other.pool_ = nullptr;
      other.scratch_ = nullptr;
    }

    auto operator=(QueryScratchLease &&other) noexcept -> QueryScratchLease & {
      if (this == &other) {
        return *this;
      }
      if (pool_ != nullptr && scratch_ != nullptr) {
        pool_->release(scratch_);
      }
      pool_ = other.pool_;
      scratch_ = other.scratch_;
      other.pool_ = nullptr;
      other.scratch_ = nullptr;
      return *this;
    }

    auto get() -> QueryScratch * { return scratch_; }

   private:
    QueryScratchPool *pool_ = nullptr;
    QueryScratch *scratch_ = nullptr;
  };

  static auto rerank_candidate_less(const RerankCandidate &lhs, const RerankCandidate &rhs)
      -> bool {
    if (lhs.distance_ != rhs.distance_) {
      return lhs.distance_ < rhs.distance_;
    }
    return lhs.id_ < rhs.id_;
  }
  /**
   * @brief Rerank search results using exact distances from build space
   * @param desc Destination ID array (topk results after rerank)
   * @param topk Number of results to return
   * @param dist_compute Distance computer from build space
   */
  void rerank(const LinearPool<DistanceType, IDType> &src,
              IDType *desc,
              uint32_t topk,
              auto dist_compute) {
    if (topk == 0) {
      return;
    }
    QueryScratchLease scratch(query_scratch_pool_.get(), topk);
    auto *query_scratch = scratch.get();
    if (query_scratch == nullptr) {
      fill_invalid_ids(desc, 0, topk);
      return;
    }

    auto &heap = query_scratch->rerank_heap_;
    auto candidate_count = static_cast<uint32_t>(src.size());
    for (uint32_t i = 0; i < candidate_count; ++i) {
      RerankCandidate candidate{dist_compute(src.id(i)), src.id(i)};
      if (heap.size() < topk) {
        heap.push_back(candidate);
        std::push_heap(heap.begin(), heap.end(), rerank_candidate_less);
        continue;
      }
      if (!rerank_candidate_less(candidate, heap.front())) {
        continue;
      }
      std::pop_heap(heap.begin(), heap.end(), rerank_candidate_less);
      heap.back() = candidate;
      std::push_heap(heap.begin(), heap.end(), rerank_candidate_less);
    }

    std::sort(heap.begin(), heap.end(), rerank_candidate_less);
    auto result_count = static_cast<uint32_t>(std::min<size_t>(heap.size(), topk));
    for (uint32_t i = 0; i < result_count; ++i) {
      desc[i] = heap[i].id_;
    }
    fill_invalid_ids(desc, result_count, topk);
  }

  /**
   * @brief Rerank search results with distances using exact distances from build space
   * @param desc Destination ID array (topk results after rerank)
   * @param distances Output distance array
   * @param topk Number of results to return
   * @param dist_compute Distance computer from build space
   */
  void rerank(const LinearPool<DistanceType, IDType> &src,
              IDType *desc,
              DistanceType *distances,
              uint32_t topk,
              auto dist_compute) {
    if (topk == 0) {
      return;
    }
    QueryScratchLease scratch(query_scratch_pool_.get(), topk);
    auto *query_scratch = scratch.get();
    if (query_scratch == nullptr) {
      fill_invalid_ids(desc, 0, topk);
      fill_invalid_distances(distances, 0, topk);
      return;
    }

    auto &heap = query_scratch->rerank_heap_;
    auto candidate_count = static_cast<uint32_t>(src.size());
    for (uint32_t i = 0; i < candidate_count; ++i) {
      RerankCandidate candidate{dist_compute(src.id(i)), src.id(i)};
      if (heap.size() < topk) {
        heap.push_back(candidate);
        std::push_heap(heap.begin(), heap.end(), rerank_candidate_less);
        continue;
      }
      if (!rerank_candidate_less(candidate, heap.front())) {
        continue;
      }
      std::pop_heap(heap.begin(), heap.end(), rerank_candidate_less);
      heap.back() = candidate;
      std::push_heap(heap.begin(), heap.end(), rerank_candidate_less);
    }

    std::sort(heap.begin(), heap.end(), rerank_candidate_less);
    auto result_count = static_cast<uint32_t>(std::min<size_t>(heap.size(), topk));
    for (uint32_t i = 0; i < result_count; ++i) {
      distances[i] = heap[i].distance_;
      desc[i] = heap[i].id_;
    }
    fill_invalid_ids(desc, result_count, topk);
    fill_invalid_distances(distances, result_count, topk);
  }

  void rabitq_search_solo(const DataType *query, uint32_t k, IDType *ids, uint32_t ef) {
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
  }

#if defined(__linux__)
  auto rabitq_search(const DataType *query, uint32_t k, IDType *ids, uint32_t ef) -> coro::task<> {
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
  }
#endif
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

    // Rerank if needed, otherwise directly copy topk
    if constexpr (kNeedsRerank) {
      rerank(pool, ids, topk, build_space_->get_query_computer(query));
    } else {
      auto result_count = std::min<uint32_t>(static_cast<uint32_t>(pool.size()), topk);
      for (uint32_t i = 0; i < result_count; ++i) {
        ids[i] = pool.id(i);
      }
      fill_invalid_ids(ids, result_count, topk);
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

    // Rerank if needed, otherwise directly copy topk
    if constexpr (kNeedsRerank) {
      rerank(pool, ids, distances, topk, build_space_->get_query_computer(query));
    } else {
      auto result_count = std::min<uint32_t>(static_cast<uint32_t>(pool.size()), topk);
      for (uint32_t i = 0; i < result_count; ++i) {
        ids[i] = pool.id(i);
        distances[i] = pool.dist(i);
      }
      fill_invalid_ids(ids, result_count, topk);
      fill_invalid_distances(distances, result_count, topk);
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

    // Rerank if needed, otherwise directly copy topk
    if constexpr (kNeedsRerank) {
      rerank(pool, ids, topk, build_space_->get_query_computer(query));
    } else {
      auto result_count = std::min<uint32_t>(static_cast<uint32_t>(pool.size()), topk);
      for (uint32_t i = 0; i < result_count; ++i) {
        ids[i] = pool.id(i);
      }
      fill_invalid_ids(ids, result_count, topk);
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

    // Rerank if needed, otherwise directly copy topk
    if constexpr (kNeedsRerank) {
      rerank(pool, ids, distances, topk, build_space_->get_query_computer(query));
    } else {
      auto result_count = std::min<uint32_t>(static_cast<uint32_t>(pool.size()), topk);
      for (uint32_t i = 0; i < result_count; ++i) {
        ids[i] = pool.id(i);
        distances[i] = pool.dist(i);
      }
      fill_invalid_ids(ids, result_count, topk);
      fill_invalid_distances(distances, result_count, topk);
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
    auto result_count = std::min<uint32_t>(static_cast<uint32_t>(pool.size()), topk);
    for (uint32_t i = 0; i < result_count; ++i) {
      ids[i] = pool.id(i);
    }
    fill_invalid_ids(ids, result_count, topk);
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
