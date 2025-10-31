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

#include <omp.h>
#include <cstddef>
#include <cstdio>
#include <unordered_set>
#include "index/graph/graph.hpp"
#include "index/neighbor.hpp"
#include "space/rabitq_space.hpp"
#include "utils/random.hpp"
#include "utils/log.hpp"
#include "utils/prefetch.hpp"
#include "utils/query_utils.hpp"

namespace alaya {
template <typename DistanceSpaceType>
struct GraphRefiner {
  // static_assert(is_rabitq_space_v<DistanceSpaceType>,
  //               "GraphRefiner only supports RaBitQSpace specializations");

 public:
  using DataType = typename DistanceSpaceType::DataTypeAlias;
  using DistanceType = typename DistanceSpaceType::DistanceTypeAlias;
  using IDType = typename DistanceSpaceType::IDTypeAlias;
  explicit GraphRefiner(DistanceSpaceType *space, Graph<DataType, IDType> *graph) : space_(space) {
    if (space_ == nullptr) {
      fprintf(stderr, "FATAL: space_ is null!\n");
      std::abort();
    }

    degree_bound_ = RaBitQSpace<>::kDegreeBound;

    const auto sys_threads_num = std::thread::hardware_concurrency(); // NOLINT
    num_threads_ = sys_threads_num == 0 ? 1 : sys_threads_num;

    num_nodes_ = space_->get_data_num();
    dim_ = space_->get_dim();

    new_neighbors_.resize(num_nodes_);
    pruned_neighbors_.resize(num_nodes_);

    size_t pool_capacity = std::min(ef_build_ * ef_build_, num_nodes_ / 10);
    pool_capacity = std::min(pool_capacity, static_cast<size_t>(10000));
    linear_pools_.clear();
    linear_pools_.reserve(num_threads_);
    for (size_t i = 0; i < num_threads_; ++i) {
      linear_pools_.emplace_back(num_nodes_, pool_capacity);
    }

    omp_set_num_threads(static_cast<int>(num_threads_));

    // load initial neighbors
    if (graph == nullptr) {
      fprintf(stderr, "FATAL: graph_ is null!\n");
      std::abort();
    } else {
      init(graph);
    }

    refine();
  }

 private:
  using CandidateList = std::vector<Neighbor<IDType, DistanceType>>;
  static constexpr size_t kMaxBsIter = 5;  // max iter for binary search of pruning bar

  DistanceSpaceType *space_;

  size_t ef_build_{400};                                // size of search pool for indexing
  size_t num_threads_;                                  // number of threads used for indexing
  size_t num_nodes_;                                    // num of data points
  size_t dim_;                                          // dimension of data
  size_t degree_bound_;                                 // multiple of 32
  static constexpr size_t kMaxCandidatePoolSize = 750;  // max num of candidates for indexing
  static constexpr size_t kMaxPrunedSize = 300;         // max number of recorded pruned candidates
  std::vector<CandidateList> new_neighbors_;            // new neighbors for current iteration
  std::vector<CandidateList> pruned_neighbors_;         // recorded pruned neighbors
  std::vector<LinearPool<DistanceType, IDType>> linear_pools_;  // list of visited hash set

  void refine() {
    search_new_neighbors();
    add_reverse_edges();
    angle_based_supplement();
    insert_refined_neighbors();
  }

  /**
   * @brief Randomly supplement neighbors to degree bound and synchronize with space and graph
   */
  void init(Graph<DataType, IDType> *graph) {
    LOG_DEBUG("Initializing graph refiner...");

    space_->set_ep(graph->get_ep());

#pragma omp parallel for schedule(dynamic)
    for (size_t id = 0; id < num_nodes_; ++id) {
      init_random_supplement(id, graph->edges(id));
    }
  }

  void search_new_neighbors() {
    LOG_DEBUG("Searching for new neighbor candidates...");
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_nodes_; ++i) {
      IDType cur_id = i;
      auto tid = omp_get_thread_num();
      CandidateList candidates;
      auto &pool = linear_pools_[tid];
      candidates.reserve(2 * kMaxCandidatePoolSize);
      find_candidates(cur_id, candidates, pool);

      // add current neighbors
      for (auto &nei : new_neighbors_[cur_id]) {
        auto neighbor_id = nei.id_;
        if (neighbor_id != cur_id && !pool.vis_.get(neighbor_id)) {
          candidates.emplace_back(nei);
        }
      }

      size_t min_size = std::min(candidates.size(), kMaxCandidatePoolSize);
      std::partial_sort(candidates.begin(),
                        candidates.begin() + static_cast<long>(min_size),  // NOLINT
                        candidates.end());
      candidates.resize(min_size);

      // prune and update space
      heuristic_prune(cur_id, candidates, new_neighbors_[cur_id]);
    }
  }

  void add_reverse_edges() {
    LOG_DEBUG("Adding reverse edges...");
    std::vector<std::mutex> locks(num_nodes_);
    std::vector<CandidateList> reverse_buffer(num_nodes_);
#pragma omp parallel for schedule(dynamic)
    for (IDType data_id = 0; data_id < num_nodes_; ++data_id) {  // for every vertex
      for (const auto &nei : new_neighbors_[data_id]) {          // for their neighbors
        IDType dst = nei.id_;

        CandidateList &dst_neighbors = new_neighbors_[dst];  // neighbors' neighbors

        // for parallel process
        std::lock_guard lock(locks[dst]);
        // check if already mutually connected
        bool dup = false;
        for (auto &dst_nei : dst_neighbors) {
          if (dst_nei.id_ == data_id) {
            dup = true;
            break;
          }
        }
        if (dup) {
          continue;
        }

        if (dst_neighbors.size() < degree_bound_) {  // add reverse edge if neighbors are not enough
          dst_neighbors.emplace_back(data_id, nei.distance_);
        } else {  // otherwise add to candidate neighbors list for later pruning
          if (reverse_buffer[dst].size() < kMaxCandidatePoolSize) {
            reverse_buffer[dst].emplace_back(data_id, nei.distance_);
          }
        }
      }
    }

#pragma omp parallel for schedule(dynamic)
    for (IDType data_id = 0; data_id < num_nodes_; ++data_id) {  // prune for every vertex
      CandidateList &tmp_pool = reverse_buffer[data_id];
      tmp_pool.reserve(tmp_pool.size() + degree_bound_);
      // add current neighbors
      tmp_pool.insert(tmp_pool.end(), new_neighbors_[data_id].begin(),
                      new_neighbors_[data_id].end());
      std::sort(tmp_pool.begin(), tmp_pool.end());
      heuristic_prune(data_id, tmp_pool, new_neighbors_[data_id]);
    }
  }

  /**
   * @brief For a candidate of the vertex, if there exists another candidate whose distance to the
   * vertex is smaller and the angle between the edges of the candidates is smaller than a
   * given threshold, then we prune the candidate. A simple BinarySearch will help us find
   * the perfect threshold.
   */
  void angle_based_supplement() {
    LOG_DEBUG("Supplementing edges...");
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_nodes_; ++i) {
      CandidateList &cur_neighbors = new_neighbors_[i];
      size_t cur_degree = cur_neighbors.size();

      // skip vertices with enough neighbors
      if (cur_degree >= degree_bound_) {  // in fact, cur_degree<=degree_bound_ is guaranteed
        continue;
      }

      // supplement candidates
      CandidateList &pruned_list = pruned_neighbors_[i];
      CandidateList new_result;
      new_result.reserve(degree_bound_);

      std::sort(pruned_list.begin(), pruned_list.end());

      // use binary search to get refined results
      float left = 0.5;
      float right = 1.0;
      size_t iter = 0;
      while (iter++ < kMaxBsIter) {
        float mid = (left + right) / 2;
        // the larger the cosine threshold, the smaller the angular threshold, and thus the more
        // edges will be retained
        add_pruned_edges(cur_neighbors, pruned_list, new_result, mid);
        if (new_result.size() < degree_bound_) {  // need more neighbors, increase the threshold
          left = mid;
        } else {
          right = mid;
        }
      }

      // update neighbors with larger cosine value since we want to retain more edges
      add_pruned_edges(cur_neighbors, pruned_list, new_result, right);

      // if the vertex still doesn't have enough neighbors, use random vertices
      if (new_result.size() < degree_bound_) {
        std::unordered_set<IDType> ids;
        ids.reserve(degree_bound_);
        for (auto &neighbor : new_result) {
          ids.emplace(neighbor.id_);
        }
        while (new_result.size() < degree_bound_) {
          IDType rand_id = rand_integer<IDType>(0, static_cast<IDType>(num_nodes_) - 1);
          if (rand_id != static_cast<IDType>(i) && ids.find(rand_id) == ids.end()) {
            new_result.emplace_back(rand_id, space_->get_distance(rand_id, i));
            ids.emplace(rand_id);
          }
        }
      }

      cur_neighbors = new_result;
    }
    LOG_DEBUG("Supplementing finished...");
  }

  /**
   * @brief insert new neighbors into space
   *
   */
  void insert_refined_neighbors() {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_nodes_; ++i) {
      auto &refined_nei = new_neighbors_[i];
      if (refined_nei.size() != degree_bound_) {
        LOG_DEBUG("node_{}'s outdegree is {} ", i, refined_nei.size());
      }
      space_->update_nei(i, refined_nei);
    }
  }

  /**
   * @brief Randomly supplement edges to degree bound for c before refinement and update the graph
   *
   * @param c centroid
   */
  void init_random_supplement(IDType c, const IDType *edges) {
    std::unordered_set<IDType> ids;
    ids.reserve(degree_bound_);
    // load current neighbors in graph
    for (size_t k = 0; k < degree_bound_ && *(edges + k) != Graph<>::kEmptyId; k++) {
      auto nei_id = *(edges + k);
      ids.emplace(nei_id);
      new_neighbors_[c].emplace_back(nei_id, space_->get_distance(c, nei_id));
    }

    // random supplement
    while (degree_bound_ - ids.size() > 0) {
      IDType rand_id = rand_integer(static_cast<IDType>(0), static_cast<IDType>(num_nodes_) - 1);
      if (rand_id != c && ids.find(rand_id) == ids.end()) {
        ids.emplace(rand_id);
        new_neighbors_[c].emplace_back(rand_id, space_->get_distance(c, rand_id));
      }
    }

    space_->update_nei(c, new_neighbors_[c]);
  }

  /**
   * @brief Use edges that are pruned in previous processes to supplement neighbors
   *
   * @param result current neighbors(sorted in heuristic_prune)
   * @param pruned_list neighbors that are pruned in previous heuristic_prune(sorted)
   * @param new_result final neighbors list
   * @param threshold experimental prune threshold
   */
  void add_pruned_edges(const CandidateList &result, const CandidateList &pruned_list,
                        CandidateList &new_result, float threshold) {
    size_t start = 0;
    new_result.clear();
    new_result = result;

    // add current neighbors' id to set for duplicate check
    std::unordered_set<IDType> nei_set;
    nei_set.reserve(degree_bound_);
    for (const auto &nei : result) {
      nei_set.emplace(nei.id_);
    }

    while (new_result.size() < degree_bound_ && start < pruned_list.size()) {
      const auto &cur = pruned_list[start];
      bool occlude = false;
      const DataType *cur_data = space_->get_data_by_id(cur.id_);
      DistanceType dik_sqr = cur.distance_;

      // duplicate check
      if (nei_set.find(cur.id_) != nei_set.end()) {
        ++start;
        continue;
      }

      // For a candidate of the vertex, if there exists another candidate whose distance to the
      // vertex is smaller and the angle between the edges of the candidates is smaller than a
      // given threshold, then we prune the candidate.
      // i: vertex
      // j: current neighbor
      // k: neighbor candidate
      for (auto &nei : new_result) {
        float dij_sqr = nei.distance_;
        if (dij_sqr > dik_sqr) {
          // new_result is sorted every time it inserts a new element, so if this dij_sqr is large
          // enough, dij_sqr after it would only be larger.
          break;
        }
        float djk_sqr = space_->get_dist_func()(space_->get_data_by_id(nei.id_), cur_data, dim_);
        // Cosine Rule
        float cosine = (dik_sqr + dij_sqr - djk_sqr) / (2 * std::sqrt(dij_sqr * dik_sqr));
        // The larger the cosine value, the smaller the angle(∠jik)
        if (cosine > threshold) {
          // the angle is too small, so connection between i and k may not be necessary
          occlude = true;
          break;
        }
      }

      if (!occlude) {
        new_result.emplace_back(cur);
        nei_set.emplace(cur.id_);
        std::sort(new_result.begin(), new_result.end());
      }

      ++start;
    }
  }

  /**
   * @brief Use the pruning rule of NSG to pick new neighbors from candidates pool.
   * In this prune processing, pruned_results will be automatically sorted since pool is sorted.
   *
   * @param cur_id current vertex
   * @param pool candidates pool(sorted by distance to cur_id)
   * @param pruned_results neighbors result after pruning
   */
  void heuristic_prune(IDType cur_id, CandidateList &pool, CandidateList &pruned_results) {
    if (pool.empty()) {
      return;
    }
    pruned_results.clear();
    size_t poolsize = pool.size();

    // if we don't have enough candidates, just keep all neighbors
    if (poolsize <= degree_bound_) {
      pruned_results = pool;
      return;
    }

    // bool vector to record if this neighbor is pruned
    std::vector<bool> pruned(poolsize, false);
    size_t start = 0;

    while (pruned_results.size() < degree_bound_ && start < poolsize) {
      auto candidate_id = pool[start].id_;

      // if already pruned, move to next
      if (pruned[start]) {
        ++start;
        continue;
      }

      pruned_results.emplace_back(pool[start]);
      const DataType *data_j = space_->get_data_by_id(candidate_id);

      // i : current vertex
      // j : neighbor added in this iter
      // k : remained unpruned candidate neighbor
      for (size_t k = start + 1; k < poolsize; ++k) {
        if (pruned[k]) {
          continue;
        }
        auto dik = pool[k].distance_;
        auto djk = space_->get_dist_func()(data_j, space_->get_data_by_id(pool[k].id_), dim_);

        if (djk < dik) {
          if (pruned_neighbors_[cur_id].size() < kMaxPrunedSize) {
            pruned_neighbors_[cur_id].emplace_back(pool[k]);
          }
          pruned[k] = true;
        }
      }

      ++start;
    }
  }

  /**
   * @brief Use estimated distance to find neighbor candidates for cur_id
   *
   * @param cur_id centroid
   * @param results neighbor candidates result
   * @param pool helper pool for iteration
   */
  void find_candidates(IDType cur_id, CandidateList &results,
                       LinearPool<DistanceType, IDType> &pool) const {
    // insert entry point to initialize search pool
    auto entry = space_->get_ep();
    pool.insert(entry, 1e10);
    mem_prefetch_l1(space_->get_data_by_id(entry), 10);

    // initialize query-related information
    auto q_computer = space_->get_query_computer(space_->get_data_by_id(cur_id));
    while (pool.has_next()) {
      // initialize centroid-related information
      auto cur_candi = pool.pop();  // cur_candi currently has the smallest est_dist
      if (pool.vis_.get(cur_candi)) {
        continue;
      }

      pool.vis_.set(cur_candi);

      q_computer.load_centroid(cur_candi);
      if (cur_candi != cur_id) {
        results.emplace_back(cur_candi, q_computer.get_exact_qr_c_dist());
      }

      // scan candidate's neighbors for more candidates
      const IDType *cand_neighbors = space_->get_edges(cur_candi);
      for (size_t i = 0; i < degree_bound_; ++i) {
        auto cand_nei = cand_neighbors[i];
        DistanceType est_dist = q_computer(i);
        if (pool.vis_.get(cand_nei)) {
          continue;
        }
        // try insert
        pool.insert(cand_nei, est_dist);
        mem_prefetch_l2(space_->get_data_by_id(pool.next_id()), 10);
      }
    }
  }
};
}  // namespace alaya
