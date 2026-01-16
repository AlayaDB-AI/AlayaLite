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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <stack>
#include <thread>
#include <utility>
#include <vector>
#include "index/graph/graph.hpp"
#include "index/graph/knng/nndescent.hpp"
#include "index/neighbor.hpp"
#include "utils/log.hpp"
#include "utils/random.hpp"
#include "utils/thread_pool.hpp"
#include "utils/timer.hpp"

namespace alaya {

template <typename DistanceSpaceType>
  requires Space<DistanceSpaceType>
struct NSGBuilder {
  using DataType = typename DistanceSpaceType::DataTypeAlias;
  using DistanceType = typename DistanceSpaceType::DistanceTypeAlias;
  using IDType = typename DistanceSpaceType::IDTypeAlias;
  using DistanceSpaceTypeAlias = DistanceSpaceType;

  std::shared_ptr<DistanceSpaceType> space_ =
      nullptr;                ///< The data manager interface for the NSG graph.
  uint32_t dim_;              ///< Dimension of the vectors.
  uint32_t max_nbrs_;         ///< Maximum number of neighbors for each node.
  uint32_t ef_construction_;  ///< Size of the search pool during graph construction.
  uint32_t cut_len_;          ///< Length used for pruning during graph construction.
  IDType vector_num_;         ///< Total number of vectors.
  IDType ep_;                 ///< Entry point for the graph.
  std::unique_ptr<Graph<DataType, IDType>> final_graph_;  ///< Final NSG graph.
  RandomGenerator rng_;                                   ///< Random number generator.

  uint32_t nndescent_max_nbrs_{};             ///< Maximum number of neighbors for Nndescent.
  uint32_t nndescent_selected_sample_num_{};  ///< Number of selected samples for Nndescent.
  uint32_t nndescent_radius_{};  ///< Radius for reverse nearest neighbors in Nndescent.
  uint32_t nndescent_candidate_pool_size_{};  ///< Size of candidate pool in Nndescent.
  uint32_t nndescent_iters_{};                ///< Number of iterations for Nndescent.

  /**
   * @brief Constructor for NSGBuilder.
   *
   * Initializes the NSGBuilder with the given distance space and optional parameters.
   *
   * @param space The data manager interface for the NSG graph.
   * @param R Maximum number of neighbors for each node (default is 32).
   * @param L Size of the search pool during graph construction (default is 200).
   */
  explicit NSGBuilder(std::shared_ptr<DistanceSpaceType> &space, uint32_t R = 32, uint32_t L = 200)
      : space_(space),
        dim_(space->get_dim()),
        max_nbrs_(R),
        ef_construction_(L),
        rng_(0x0903),
        nndescent_max_nbrs_(64),
        nndescent_selected_sample_num_(10),
        nndescent_radius_(100),
        nndescent_candidate_pool_size_(nndescent_max_nbrs_ + 50),
        nndescent_iters_(10) {
    cut_len_ = max_nbrs_ + 100;
    srand(0x1998);
    vector_num_ = space_->get_data_num();
  }

  /**
   * @brief Build the NSG graph.
   *
   * Constructs the NSG graph using the Nndescent algorithm and subsequent linking and pruning
   * steps.
   *
   * @return A unique pointer to the final NSG graph.
   */
  auto build_graph(uint32_t thread_num = 1) -> std::unique_ptr<Graph<DataType, IDType>> {
    NndescentImpl<DistanceSpaceType> nndescent(space_, nndescent_max_nbrs_);
    nndescent.selected_sample_num_ = nndescent_selected_sample_num_;
    nndescent.radius_ = nndescent_radius_;
    nndescent.candidate_pool_size_ = nndescent_candidate_pool_size_;
    nndescent.iterations_ = nndescent_iters_;

    auto nndescent_graph = nndescent.build_graph();

    init(nndescent_graph);
    std::vector<uint32_t> degrees(vector_num_, 0);

    {
      Graph<DataType, IDType> tmp_graph(space_->get_capacity(), max_nbrs_);
      link(nndescent_graph, tmp_graph);

      final_graph_ = std::make_unique<Graph<DataType, IDType>>(space_->get_capacity(), max_nbrs_);
      // std::fill_n(final_graph_->data_, vector_num_ * max_nbrs_,
      // Graph<DataType, IDType>::kEmptyId);
      final_graph_->eps_.push_back(ep_);

      unsigned int num_cores = std::thread::hardware_concurrency();
      ThreadPool pool(num_cores);

      IDType per_core_num = (vector_num_ + num_cores - 1) / num_cores;
      for (uint32_t thread_id = 0; thread_id < num_cores; thread_id++) {
        pool.enqueue([this, &tmp_graph, &degrees, per_core_num, thread_id]() -> auto {
          IDType start = thread_id * per_core_num;
          IDType end = std::min((thread_id + 1) * per_core_num, vector_num_);

          for (IDType i = start; i < end; i++) {
            int cnt = 0;
            for (uint32_t j = 0; j < max_nbrs_; j++) {
              IDType id = tmp_graph.at(i, j);
              if (id != Graph<DataType, IDType>::kEmptyId) {
                final_graph_->at(i, cnt) = id;
                cnt += 1;
              }
            }
            degrees[i] = cnt;
          }
        });
      }
      pool.wait_until_all_tasks_completed(num_cores);
    }

    [[maybe_unused]] int num_attached = tree_grow(degrees);
    int max_degree = 0;
    int min_degree = 1e6;
    double avg_degree = 0;
    for (IDType i = 0; i < vector_num_; i++) {
      int size = 0;
      while (static_cast<uint32_t>(size) < max_nbrs_ &&
             final_graph_->at(i, size) != Graph<DataType, IDType>::kEmptyId) {
        size += 1;
      }
      max_degree = std::max(size, max_degree);
      min_degree = std::min(size, min_degree);
      avg_degree += size;
    }
    avg_degree /= vector_num_;
    LOG_INFO("Degree Statistics: Max = {}, Min = {}, Avg = {}", max_degree, min_degree, avg_degree);

    return std::move(final_graph_);
  }

  /**
   * @brief Initialize the NSG graph.
   *
   * Computes the center of the dataset and selects an entry point using a search on the graph.
   *
   * @param knng The initial graph built by Nndescent.
   */
  void init(const std::unique_ptr<Graph<DataType, IDType>> &knng) {
    std::vector<DataType> center(dim_);
    for (size_t i = 0; i < dim_; ++i) {
      center[i] = 0.0;
    }

    for (size_t i = 0; i < vector_num_; i++) {
      for (size_t j = 0; j < dim_; j++) {
        center[j] += space_->get_data_by_id(i)[j];
      }
    }

    for (size_t i = 0; i < dim_; i++) {
      center[i] /= vector_num_;
    }

    size_t ep_init = rng_.rand_int(vector_num_);

    std::vector<Neighbor<IDType>> retset;
    std::vector<Node<IDType>> tmpset;
    std::vector<bool> vis(vector_num_);
    search_on_graph<false>(center.data(), knng, vis, ep_init, ef_construction_, retset, tmpset);
    // set enterpoint
    this->ep_ = retset[0].id_;
  }

  /**
   * @brief Perform a search on the graph.
   *
   * Searches for the nearest neighbors of a query vector on the graph.
   *
   * @tparam collect_full_set Flag to indicate whether to collect the full set of neighbors.
   * @param q Query vector.
   * @param graph The graph to search on.
   * @param vis Visited nodes.
   * @param ep Entry point for the search.
   * @param pool_size Size of the search pool.
   * @param retset Result set of nearest neighbors.
   * @param full_set Full set of neighbors (if collect_full_set is true).
   */
  template <bool collect_full_set>
  void search_on_graph(const DataType *q,
                       const std::unique_ptr<Graph<DataType, IDType>> &graph,
                       std::vector<bool> &vis,
                       IDType ep,
                       int pool_size,
                       std::vector<Neighbor<IDType>> &retset,
                       std::vector<Node<IDType>> &full_set) const {
    RandomGenerator gen(0x1234);
    retset.resize(pool_size + 1);

    std::vector<IDType> init_ids(pool_size);
    int num_ids = 0;
    for (size_t i = 0; i < init_ids.size() && i < graph->max_nbrs_; i++) {
      IDType id = graph->at(ep, i);
      if (id >= vector_num_) {
        continue;
      }
      init_ids[i] = id;
      vis[id] = true;
      num_ids += 1;
    }
    while (num_ids < pool_size) {
      IDType id = gen.rand_int(vector_num_);
      if (vis[id]) {
        continue;
      }
      init_ids[num_ids] = id;
      num_ids++;
      vis[id] = true;
    }
    for (size_t i = 0; i < init_ids.size(); i++) {
      IDType id = init_ids[i];
      DistanceType dist =
          space_->get_dist_func()(const_cast<DataType *>(q), space_->get_data_by_id(id), dim_);
      retset[i] = Neighbor<IDType>(id, dist, true);
      if (collect_full_set) {
        full_set.emplace_back(id, dist);
      }
    }
    std::sort(retset.begin(), retset.begin() + pool_size);
    int k = 0;
    while (k < pool_size) {
      int updated_pos = pool_size;
      if (retset[k].flag_) {
        retset[k].flag_ = false;
        IDType n = retset[k].id_;
        for (size_t m = 0; m < graph->max_nbrs_; m++) {
          IDType id = graph->at(n, m);
          if (id >= vector_num_ || vis[id]) {
            continue;
          }
          vis[id] = true;
          DistanceType dist =
              space_->get_dist_func()(const_cast<DataType *>(q), space_->get_data_by_id(id), dim_);
          Neighbor<IDType> nn(id, dist, true);
          if (collect_full_set) {
            full_set.emplace_back(id, dist);
          }
          if (dist >= retset[pool_size - 1].distance_) {
            continue;
          }
          int r = insert_into_pool(retset.data(), pool_size, nn);
          updated_pos = std::min(updated_pos, r);
        }
      }
      k = (updated_pos <= k) ? updated_pos : (k + 1);
    }
  }

  /**
   * @brief Link nodes in the graph.
   *
   * Links nodes in the graph using the initial graph built by Nndescent.
   *
   * @param knng The initial graph built by Nndescent.
   * @param graph The graph to be linked.
   */
  void link(const std::unique_ptr<Graph<DataType, IDType>> &knng, Graph<DataType, IDType> &graph) {
    auto t1 = Timer();
    std::atomic<int> cnt{0};
    unsigned int num_cores = std::thread::hardware_concurrency();
    ThreadPool pool(num_cores);

    IDType per_core_num = (vector_num_ + num_cores - 1) / num_cores;
    for (uint32_t thread_id = 0; thread_id < num_cores; thread_id++) {
      pool.enqueue([this, &knng, &graph, &cnt, per_core_num, thread_id]() -> auto {
        for (IDType i = thread_id * per_core_num;
             i < (thread_id + 1) * per_core_num && i < vector_num_;
             i++) {
          std::vector<Node<IDType>> pool;
          std::vector<Neighbor<IDType>> tmp;
          std::vector<bool> vis(vector_num_);
          search_on_graph<true>(space_->get_data_by_id(i),
                                knng,
                                vis,
                                ep_,
                                ef_construction_,
                                tmp,
                                pool);
          sync_prune(i, pool, vis, knng, graph);
          pool.clear();
          tmp.clear();
          int cur = cnt.fetch_add(1);
          if ((cur + 1) % 10000 == 0) {
            LOG_INFO("NSG building progress: [{}/{}]", cur + 1, vector_num_);
          }
        }
      });
    }
    pool.wait_until_all_tasks_completed(num_cores);

    pool.reset_task();
    std::vector<std::mutex> locks(vector_num_);
    for (uint32_t thread_id = 0; thread_id < num_cores; thread_id++) {
      pool.enqueue([this, &graph, &locks, per_core_num, thread_id]() -> auto {
        for (IDType i = thread_id * per_core_num;
             i < (thread_id + 1) * per_core_num && i < vector_num_;
             ++i) {
          add_reverse_links(i, locks, graph);
        }
      });
    }
    pool.wait_until_all_tasks_completed(num_cores);

    LOG_INFO("NSG building cost: {}", t1.elapsed() * 1.0 / 1000 / 1000);
  }

  /**
   * @brief Synchronize and prune the graph.
   *
   * Synchronizes and prunes the graph to ensure valid and efficient connections.
   *
   * @param q Query node.
   * @param pool Pool of neighbors.
   * @param vis Visited nodes.
   * @param knng The initial graph built by Nndescent.
   * @param graph The graph to be synchronized and pruned.
   */
  void sync_prune(IDType q,
                  std::vector<Node<IDType>> &pool,
                  std::vector<bool> &vis,
                  const std::unique_ptr<Graph<DataType, IDType>> &knng,
                  Graph<DataType, IDType> &graph) {
    for (size_t i = 0; i < knng->max_nbrs_; i++) {
      IDType id = knng->at(q, i);
      if (id >= vector_num_ || vis[id]) {
        continue;
      }

      DistanceType dist = space_->get_distance(q, id);
      pool.emplace_back(id, dist);
    }

    std::sort(pool.begin(), pool.end());

    std::vector<Node<IDType>> result;

    size_t start = 0;
    if (pool[start].id_ == q) {
      start++;
    }
    result.push_back(pool[start]);

    while (result.size() < max_nbrs_ && (++start) < pool.size() && start < cut_len_) {  // NOLINT
      auto &p = pool[start];
      bool occlude = false;
      for (size_t t = 0; t < result.size(); t++) {
        if (p.id_ == result[t].id_) {
          occlude = true;
          break;
        }

        DistanceType djk = space_->get_distance(result[t].id_, p.id_);
        if (djk < p.distance_) {
          occlude = true;
          break;
        }
      }
      if (!occlude) {
        result.push_back(p);
      }
    }

    for (uint32_t i = 0; i < max_nbrs_; i++) {
      if (i < result.size()) {
        graph.at(q, i) = result[i].id_;
      } else {
        graph.at(q, i) = Graph<DataType, IDType>::kEmptyId;
      }
    }
  }
  /**
   * @brief Add reverse links to the graph.
   *
   * This function ensures that the graph is bidirectional by adding reverse links.
   * For each node, it checks its neighbors and adds the current node as a neighbor
   * to those nodes if it is not already present. This process helps in maintaining
   * the bidirectional property of the graph.
   *
   * @param q The index of the current node.
   * @param locks A vector of mutexes for thread safety.
   * @param graph The graph to which reverse links are to be added.
   */
  void add_reverse_links(IDType q, std::vector<std::mutex> &locks, Graph<DataType, IDType> &graph) {
    for (uint32_t i = 0; i < max_nbrs_; i++) {
      if (graph.at(q, i) == Graph<DataType, IDType>::kEmptyId) {
        break;
      }

      Node<IDType> sn(q, graph.at(q, i));
      IDType des = graph.at(q, i);

      std::vector<Node<IDType>> tmp_pool;
      int dup = 0;
      {
        std::scoped_lock guard(locks[des]);
        for (uint32_t j = 0; j < max_nbrs_; j++) {
          if (graph.at(des, j) == Graph<DataType, IDType>::kEmptyId) {
            break;
          }
          if (q == graph.at(des, j)) {
            dup = 1;
            break;
          }
          tmp_pool.push_back(Node<IDType>(graph.at(des, j),
                                          0));  // Assuming Node constructor takes ID and distance
        }
      }

      if (dup != 0) {
        continue;
      }

      tmp_pool.push_back(sn);
      if (tmp_pool.size() > max_nbrs_) {
        std::vector<Node<IDType>> result;
        int start = 0;
        std::sort(tmp_pool.begin(), tmp_pool.end());
        result.push_back(tmp_pool[start]);

        while (result.size() < max_nbrs_ && (++start) < static_cast<int>(tmp_pool.size())) {
          auto &p = tmp_pool[start];
          bool occlude = false;
          for (size_t t = 0; t < result.size(); t++) {
            if (p.id_ == result[t].id_) {
              occlude = true;
              break;
            }
            DistanceType djk = space_->get_distance(result[t].id_, p.id_);

            if (djk < p.distance_) {
              occlude = true;
              break;
            }
          }
          if (!occlude) {
            result.push_back(p);
          }
        }

        {
          std::scoped_lock guard(locks[des]);
          for (size_t t = 0; t < result.size(); t++) {
            graph.at(des, t) = result[t].id_;
          }
        }

      } else {
        std::scoped_lock guard(locks[des]);
        for (uint32_t t = 0; t < max_nbrs_; t++) {
          if (graph.at(des, t) == Graph<DataType, IDType>::kEmptyId) {
            graph.at(des, t) = sn.id_;
            break;
          }
        }
      }
    }
  }

  /**
   * @brief Grow a tree to ensure all nodes are connected in the graph.
   *
   * This function ensures that all nodes in the graph are connected by growing a tree.
   * It performs a depth-first search (DFS) to traverse the graph and attaches unlinked
   * nodes to the tree if they are not already connected.
   *
   * @param degrees A vector containing the degree of each node.
   * @return int The number of nodes that were attached to the tree.
   */
  auto tree_grow(std::vector<uint32_t> &degrees) -> int {
    int root = ep_;
    std::vector<bool> vis(vector_num_, false);
    int num_attached = 0;
    int cnt = 0;
    while (true) {
      cnt = dfs(vis, root, cnt);
      if (static_cast<IDType>(cnt) >= vector_num_) {
        break;
      }
      std::vector<bool> vis2(vector_num_, false);
      root = attach_unlinked(vis, vis2, degrees);
      num_attached += 1;
    }
    return num_attached;
  }

  /**
   * @brief Perform a depth-first search (DFS) to traverse the graph.
   *
   * This function performs a DFS to traverse the graph starting from the given root node.
   * It marks visited nodes and counts the number of nodes visited during the traversal.
   *
   * @param vis A vector indicating whether each node has been visited.
   * @param root The root node to start the DFS.
   * @param cnt The current count of visited nodes.
   * @return int The updated count of visited nodes.
   */
  auto dfs(std::vector<bool> &vis, IDType root, int cnt) const -> int {
    IDType node = root;
    std::stack<IDType> stack;
    stack.push(root);
    if (vis[root]) {
      cnt++;
    }
    vis[root] = true;
    while (!stack.empty()) {
      IDType next = Graph<DataType, IDType>::kEmptyId;
      for (uint32_t i = 0; i < max_nbrs_; i++) {
        IDType id = final_graph_->at(node, i);
        if (id != Graph<DataType, IDType>::kEmptyId && !vis[id]) {
          next = id;
          break;
        }
      }
      if (next == Graph<DataType, IDType>::kEmptyId) {
        stack.pop();
        if (stack.empty()) {
          break;
        }
        node = stack.top();
        continue;
      }
      node = next;
      vis[node] = true;
      stack.push(node);
      cnt++;
    }
    return cnt;
  }

  /**
   * @brief Attach unlinked nodes to the tree.
   *
   * This function attaches unlinked nodes to the tree by finding nodes that are not
   * connected and linking them to nodes with available degree slots. It ensures that
   * all nodes are eventually connected in the graph.
   *
   * @param vis A vector indicating whether each node has been visited.
   * @param vis2 A secondary vector indicating whether each node has been visited.
   * @param degrees A vector containing the degree of each node.
   * @return IDType The ID of the node that was attached.
   */
  auto attach_unlinked(std::vector<bool> &vis,
                       std::vector<bool> &vis2,
                       std::vector<uint32_t> &degrees) -> IDType {
    IDType id = Graph<DataType, IDType>::kEmptyId;
    for (IDType i = 0; i < vector_num_; i++) {
      if (vis[i]) {
        id = i;
        break;
      }
    }
    if (id == Graph<DataType, IDType>::kEmptyId) {
      return Graph<DataType, IDType>::kEmptyId;
    }
    std::vector<Neighbor<IDType>> tmp;
    std::vector<Node<IDType>> pool;
    search_on_graph<true>(space_->get_data_by_id(id),
                          final_graph_,
                          vis2,
                          ep_,
                          ef_construction_,
                          tmp,
                          pool);
    std::sort(pool.begin(), pool.end());
    IDType node;
    bool found = false;
    for (size_t i = 0; i < pool.size(); i++) {
      node = pool[i].id_;
      if (degrees[node] < max_nbrs_ && node != id) {
        found = true;
        break;
      }
    }
    if (!found) {
      do {
        node = rng_.rand_int(vector_num_);
        if (!vis[node] && degrees[node] < max_nbrs_ && node != id) {
          found = true;
        }
      } while (!found);
    }
    uint32_t pos = degrees[node];
    final_graph_->at(node, pos) = id;
    degrees[node] += 1;

    return node;
  }

  /**
   * @brief Insert a node into the pool.
   *
   * @param pool The pool to insert the node into.
   * @param pool_size The size of the pool.
   * @param nn The node to insert.
   * @return int The position of the inserted node in the pool.
   */
  auto insert_into_pool(Neighbor<IDType> *pool, int pool_size, const Neighbor<IDType> &nn) const
      -> int {
    for (int i = 0; i < pool_size; i++) {
      if (pool[i].id_ == nn.id_) {
        return pool_size;
      }
    }
    if (nn.distance_ >= pool[pool_size - 1].distance_) {
      return pool_size;
    }
    int pos = pool_size - 1;
    while (pos > 0 && nn.distance_ < pool[pos - 1].distance_) {
      pool[pos] = pool[pos - 1];
      pos--;
    }
    pool[pos] = nn;
    return pos;
  }
};

}  // namespace alaya
