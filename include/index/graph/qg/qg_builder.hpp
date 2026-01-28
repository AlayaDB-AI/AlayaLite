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

#include <cstddef>
#include <memory>
#include <vector>

#include "index/neighbor.hpp"
#include "space/rabitq_space.hpp"
#include "space/space_concepts.hpp"
#include "utils/log.hpp"
#include "utils/rabitq_utils/search_utils/buffer.hpp"
#include "utils/rabitq_utils/search_utils/hashset.hpp"
#include "utils/random.hpp"

namespace alaya {
template <typename DistanceSpaceType>
  requires Space<DistanceSpaceType>
class QGBuilder {
  // static_assert(is_rabitq_space_v<DistanceSpaceType>, "qg only supports RaBitQSpace
  // specializations");

  using DataType = typename DistanceSpaceType::DataTypeAlias;
  using DistanceType = typename DistanceSpaceType::DistanceTypeAlias;
  using IDType = typename DistanceSpaceType::IDTypeAlias;
  using DistanceSpaceTypeAlias = DistanceSpaceType;

  using CandidateList = std::vector<Neighbor<IDType, DistanceType>>;

 public:
  explicit QGBuilder(std::shared_ptr<DistanceSpaceType> &space,
                     size_t num_threads = std::numeric_limits<size_t>::max())
      : space_(space) {
    if (space_ == nullptr) {
      fprintf(stderr, "FATAL: (qg_builder.hpp) space_ is null!\n");
      std::abort();
    }
    num_nodes_ = space_->get_data_num();
    dim_ = space_->get_dim();
    degree_bound_ = RaBitQSpace<>::kDegreeBound;

    new_neighbors_.resize(num_nodes_);
    pruned_neighbors_.resize(num_nodes_);

    const size_t sys_threads_num = std::thread::hardware_concurrency();  // NOLINT
    num_threads_ = std::min(num_threads, sys_threads_num == 0 ? 1 : sys_threads_num);
    omp_set_num_threads(static_cast<int>(num_threads_));

    size_t pool_capacity = std::min(ef_build_ * ef_build_, num_nodes_ / 10);
    visited_list_.clear();
    visited_list_.reserve(num_threads_);
    for (size_t i = 0; i < num_threads_; ++i) {
      visited_list_.emplace_back(pool_capacity);
    }

    cal_ep();

    random_init();
  }

  void build_graph() {
    for (size_t i = 0; i < kNumIter - 1; ++i) {
      iter(false);
    }
    // we only need to supplement edges in last round
    iter(true);
  }

 private:
  static constexpr size_t kMaxBsIter = 5;  ///< max iter for binary search of pruning bar
  static constexpr size_t kMaxCandidatePoolSize = 750;  ///< max num of candidates for indexing
  static constexpr size_t kMaxPrunedSize = 300;  ///< max number of recorded pruned candidates
  static constexpr size_t kNumIter = 3;          ///< the number of iterations for building qg

  IDType ep_;                                      ///< Entry point for the graph.
  size_t ef_build_{400};                           ///< size of search pool for indexing
  size_t num_threads_;                             ///< number of threads used for indexing
  size_t num_nodes_;                               ///< num of data points
  size_t dim_;                                     ///< dimension of data
  size_t degree_bound_;                            ///< multiple of 32
  std::vector<CandidateList> new_neighbors_;       ///< new neighbors for current iteration
  std::vector<CandidateList> pruned_neighbors_;    ///< recorded pruned neighbors
  std::vector<HashBasedBooleanSet> visited_list_;  // list of visited hash set

  // std::unique_ptr<Graph<DataType, IDType>> final_graph_;
  std::shared_ptr<DistanceSpaceType> space_;

  void iter(bool sup) {
    if (sup) {
      for (size_t i = 0; i < num_nodes_; ++i) {
        pruned_neighbors_[i].clear();
        pruned_neighbors_[i].reserve(kMaxPrunedSize);
      }
    }

    search_new_neighbors(sup);

    add_reverse_edges(sup);

    // Use pruned edges to refine graph
    if (sup) {
      angle_based_supplement();
    }

// update result with space and graph in the end of every iteration
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < static_cast<int64_t>(num_nodes_); ++i) {
      if (sup && new_neighbors_[i].size() < degree_bound_) {
        LOG_ERROR("After supplement, node_{} only has {} neighbors.", i, new_neighbors_[i].size());
      }
      space_->update_nei(i, new_neighbors_[i]);
    }
  }

  void search_new_neighbors(bool sup) {
    LOG_INFO("Searching for new neighbor candidates...");
#if defined(__AVX512F__)
  #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < static_cast<int64_t>(num_nodes_); ++i) {
      IDType cur_id = static_cast<IDType>(i);
      auto tid = omp_get_thread_num();
      CandidateList candidates;
      HashBasedBooleanSet &vis = visited_list_[tid];
      candidates.reserve(2 * kMaxCandidatePoolSize);
      vis.clear();
      find_candidates(cur_id, candidates, vis);

      // add current neighbors
      for (auto &nei : new_neighbors_[cur_id]) {
        auto neighbor_id = nei.id_;
        if (neighbor_id != cur_id && !vis.get(neighbor_id)) {
          candidates.emplace_back(nei);
        }
      }

      size_t min_size = std::min(candidates.size(), kMaxCandidatePoolSize);
      std::partial_sort(candidates.begin(),
                        candidates.begin() + static_cast<long>(min_size),  // NOLINT
                        candidates.end());
      candidates.resize(min_size);

      // prune and update qg
      heuristic_prune(cur_id, candidates, new_neighbors_[cur_id], sup);
    }
#endif
  }

  void add_reverse_edges(bool sup) {
    LOG_INFO("Adding reverse edges...");
#if defined(__AVX512F__)
    std::vector<std::mutex> locks(num_nodes_);
    std::vector<CandidateList> reverse_buffer(num_nodes_);
  #pragma omp parallel for schedule(dynamic)
    for (int64_t data_id = 0; data_id < static_cast<int64_t>(num_nodes_);
         ++data_id) {  // for every vertex
      for (const auto &nei : new_neighbors_[data_id]) {
        auto dst = nei.id_;
        bool dup = false;
        CandidateList &dst_neighbors = new_neighbors_[dst];
        std::lock_guard lock(locks[dst]);
        for (auto &dst_nei : dst_neighbors) {
          if (dst_nei.id_ == data_id) {
            dup = true;
            break;
          }
        }
        if (dup) {
          continue;
        }

        if (dst_neighbors.size() < degree_bound_) {
          dst_neighbors.emplace_back(data_id, nei.distance_);
        } else {
          if (reverse_buffer[dst].size() < kMaxCandidatePoolSize) {
            reverse_buffer[dst].emplace_back(data_id, nei.distance_);
          }
        }
      }
    }
  #pragma omp parallel for schedule(dynamic)
    for (int64_t data_id = 0; data_id < static_cast<int64_t>(num_nodes_);
         ++data_id) {  // prune for every vertex
      CandidateList &tmp_pool = reverse_buffer[data_id];
      tmp_pool.reserve(tmp_pool.size() + degree_bound_);
      // add current neighbors
      tmp_pool.insert(tmp_pool.end(),
                      new_neighbors_[data_id].begin(),
                      new_neighbors_[data_id].end());
      std::sort(tmp_pool.begin(), tmp_pool.end());
      heuristic_prune(data_id, tmp_pool, new_neighbors_[data_id], sup);
    }
#endif
  }

  /**
   * @brief For a candidate of the vertex, if there exists another candidate whose distance to the
   * vertex is smaller and the angle between the edges of the candidates is smaller than a
   * given threshold, then we prune the candidate. A simple BinarySearch will help us find
   * the perfect threshold.
   */
  void angle_based_supplement() {
    LOG_INFO("Supplementing edges...");
#if defined(__AVX512F__)
  #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < static_cast<int64_t>(num_nodes_); ++i) {
      CandidateList &cur_neighbors = new_neighbors_[i];
      size_t cur_degree = cur_neighbors.size();

      // skip vertices with enough neighbors
      if (cur_degree >= degree_bound_) {  // in fact, cur_degree<=degree_bound_ is guaranteed
        continue;
      }

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
        add_pruned_edges(cur_neighbors, pruned_list, new_result, mid);
        if (new_result.size() < degree_bound_) {
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
#endif
    LOG_INFO("Supplementing finished...");
  }

#if defined(__AVX512F__)
  /**
   * @brief Use est_dist to find candidate neighbors for cur_id, exclude the vertex itself
   *
   * @param cur_id find candidate neighbors for cur_id
   * @param results candidate neighbors result for cur_id
   * @param vis record if a node has already been visited
   */
  void find_candidates(IDType cur_id, CandidateList &results, HashBasedBooleanSet &vis) {
    // insert entry point to initialize search pool
    SearchBuffer tmp_pool(ef_build_);
    tmp_pool.insert(ep_, 1e10);
    mem_prefetch_l1(space_->get_data_by_id(ep_), 10);

    // init query
    auto q_computer = space_->get_query_computer(cur_id);
    while (tmp_pool.has_next()) {
      auto cur_candi = tmp_pool.pop();
      if (vis.get(cur_candi)) {
        continue;
      }
      vis.set(cur_candi);

      q_computer.load_centroid(cur_candi);

      // scan cur_candi's neighbors
      auto *nei_ptr = space_->get_edges(cur_candi);
      for (size_t i = 0; i < degree_bound_; ++i) {
        auto cur_nei = nei_ptr[i];
        auto dist = q_computer(i);
        if (tmp_pool.is_full(dist) || vis.get(cur_nei)) {
          continue;
        }
        // try insert
        tmp_pool.insert(cur_nei, dist);
        mem_prefetch_l2(reinterpret_cast<const char *>(space_->get_data_by_id(tmp_pool.next_id())),
                        10);
      }

      if (cur_candi != cur_id) {
        results.emplace_back(cur_candi, q_computer.get_exact_qr_c_dist());
      }
    }
  }

  /**
   * @brief Use the pruning rule of NSG to pick new neighbors from candidates pool.
   * In this prune processing, pruned_results will be automatically sorted since pool is sorted.
   *
   * @param cur_id current vertex
   * @param pool candidates pool(sorted by distance to cur_id)
   * @param pruned_results neighbors result after pruning, not neighbors that are pruned
   */
  void heuristic_prune(IDType cur_id,
                       CandidateList &pool,
                       CandidateList &pruned_results,
                       bool sup) {
    if (pool.empty()) {
      return;
    }
    pruned_results.clear();
    size_t poolsize = pool.size();

    // if we dont have enough candidates, just keep all neighbors
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

      pruned_results.emplace_back(pool[start]);  // add current candidate to result

      // i : current vertex
      // j : neighbor added in this iter
      // k : remained unpruned candidate neighbor
      for (size_t k = start + 1; k < poolsize; ++k) {
        if (pruned[k]) {
          continue;
        }
        auto dik = pool[k].distance_;
        auto djk = space_->get_distance(candidate_id, pool[k].id_);

        if (djk < dik) {
          if (sup && pruned_neighbors_[cur_id].size() < kMaxPrunedSize) {
            pruned_neighbors_[cur_id].emplace_back(pool[k]);
          }
          pruned[k] = true;
        }
      }

      ++start;
    }
  }

  /**
   * @brief Use edges that are pruned in previous processes to supplement neighbors
   *
   * @param result current neighbors(sorted in heuristic_prune)
   * @param pruned_list neighbors that are pruned in previous heuristic_prune(sorted)
   * @param new_result final neighbors list
   * @param threshold experimental prune threshold
   */
  void add_pruned_edges(const CandidateList &result,
                        const CandidateList &pruned_list,
                        CandidateList &new_result,
                        float threshold) {
    size_t start = 0;
    new_result.clear();
    new_result = result;

    std::unordered_set<IDType> nei_set;
    nei_set.reserve(degree_bound_);
    for (const auto &nei : result) {
      nei_set.emplace(nei.id_);
    }

    while (new_result.size() < degree_bound_ && start < pruned_list.size()) {
      const auto &cur = pruned_list[start];
      bool occlude = false;
      auto dik_sqr = cur.distance_;

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
        auto dij_sqr = nei.distance_;
        if (dij_sqr > dik_sqr) {
          // new_result is sorted every time it inserts a new element, so if this dij_sqr is large
          // enough, dij_sqr after it would only be larger.
          break;
        }
        auto djk_sqr = space_->get_distance(cur.id_, nei.id_);
        float cosine = (dik_sqr + dij_sqr - djk_sqr) / (2 * std::sqrt(dij_sqr * dik_sqr));
        if (cosine > threshold) {
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
#endif

  void cal_ep() {
    // compute centroid
    std::vector<std::vector<DataType>> all_results(num_threads_, std::vector<DataType>(dim_, 0));
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < static_cast<int64_t>(num_nodes_); ++i) {
      auto tid = omp_get_thread_num();
      std::vector<DataType> &cur_results = all_results[tid];
      auto cur_data = space_->get_data_by_id(i);
      for (size_t k = 0; k < dim_; ++k) {
        cur_results[k] += cur_data[k];
      }
    }
    std::vector<DataType> centroid(dim_, 0);
    for (auto &one_res : all_results) {
      for (size_t i = 0; i < dim_; ++i) {
        centroid[i] += one_res[i];
      }
    }
    DataType inv_num_points = 1 / static_cast<DataType>(num_nodes_);
    for (size_t i = 0; i < dim_; ++i) {
      centroid[i] = centroid[i] * inv_num_points;
    }

    // find the exact nearest neighbor of the centroid and set it as the entry point of qg.
    std::vector<Neighbor<IDType, DistanceType>>
        best_entries(num_threads_,
                     Neighbor<IDType, DistanceType>{0, std::numeric_limits<DistanceType>::max()});
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < static_cast<int64_t>(num_nodes_); ++i) {
      auto tid = omp_get_thread_num();
      Neighbor<IDType, DistanceType> &cur_entry = best_entries[tid];
      auto cur_data = space_->get_data_by_id(i);
      DistanceType distance = space_->get_dist_func()(cur_data, centroid.data(), dim_);
      if (distance < cur_entry.distance_) {
        cur_entry.id_ = static_cast<IDType>(i);
        cur_entry.distance_ = distance;
      }
    }
    IDType nearest_neighbor = 0;
    DistanceType min_dist = std::numeric_limits<DistanceType>::max();
    for (auto &candi : best_entries) {
      if (candi.distance_ < min_dist) {
        nearest_neighbor = candi.id_;
        min_dist = candi.distance_;
      }
    }

    // final entry point
    ep_ = nearest_neighbor;
    LOG_INFO("final entry point in qg: {}", ep_);
    space_->set_ep(ep_);
  }

  void random_init() {
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < static_cast<int64_t>(num_nodes_); ++i) {
      // generate random neighbors
      std::unordered_set<IDType> neighbor_set;
      neighbor_set.reserve(degree_bound_);
      while (neighbor_set.size() < degree_bound_) {
        IDType rand_id = rand_integer<IDType>(0, num_nodes_ - 1);
        if (rand_id != static_cast<IDType>(i)) {
          neighbor_set.emplace(rand_id);
        }
      }

      // record initial neighbors for later iteration
      new_neighbors_[i].reserve(degree_bound_);
      for (auto cur_neigh : neighbor_set) {
        new_neighbors_[i].emplace_back(cur_neigh, space_->get_distance(i, cur_neigh));
      }

      // record in space
      space_->update_nei(i, new_neighbors_[i]);
    }
  }
};
}  // namespace alaya
