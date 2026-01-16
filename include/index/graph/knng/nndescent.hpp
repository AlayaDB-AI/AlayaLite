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
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <vector>
#include "index/graph/graph.hpp"
#include "index/neighbor.hpp"
#include "space/space_concepts.hpp"
#include "utils/log.hpp"
#include "utils/random.hpp"
#include "utils/thread_pool.hpp"
#include "utils/timer.hpp"

namespace alaya {

template <typename DistanceSpaceType,
          typename DataType = typename DistanceSpaceType::DataTypeAlias,
          typename DistanceType = typename DistanceSpaceType::DistanceTypeAlias,
          typename IDType = typename DistanceSpaceType::IDTypeAlias>
  requires Space<DistanceSpaceType, DataType, DistanceType, IDType>
struct NndescentImpl {
  using DistanceSpaceTypeAlias = DistanceSpaceType;
  struct Nhood {
    std::vector<Neighbor<IDType>> candidate_pool_;  ///< candidate pool (a max heap)
    uint32_t max_edge_;                             ///< max number of edges in the neighborhood
    std::mutex lock_;                               ///< lock for the candidate pool
    std::vector<IDType> nn_new_, nn_old_;           ///< new and old neighbors
    std::vector<IDType> rnn_new_, rnn_old_;         ///< reverse new and old neighbors

    Nhood(std::mt19937 &rng, int s, int64_t N) {
      max_edge_ = s;
      nn_new_.resize(s * 2);
      gen_random(rng, nn_new_.data(), nn_new_.size(), N);
    }

    auto operator=(const Nhood &other) -> Nhood & {
      max_edge_ = other.max_edge_;
      std::copy(other.nn_new_.begin(), other.nn_new_.end(), std::back_inserter(nn_new_));
      nn_new_.reserve(other.nn_new_.capacity());
      candidate_pool_.reserve(other.candidate_pool_.capacity());
      return *this;
    }

    Nhood(const Nhood &other) {
      max_edge_ = other.max_edge_;
      std::copy(other.nn_new_.begin(), other.nn_new_.end(), std::back_inserter(nn_new_));
      nn_new_.reserve(other.nn_new_.capacity());
      candidate_pool_.reserve(other.candidate_pool_.capacity());
    }

    /**
     * @brief Insert a neighbor into the candidate pool.
     *
     * @param id The vector id.
     * @param dist The distance to the node.
     */
    void insert(IDType id, DistanceType dist) {
      std::scoped_lock guard(lock_);

      if (dist > candidate_pool_.front().distance_) {
        return;
      }

      for (size_t i = 0; i < candidate_pool_.size(); i++) {
        if (candidate_pool_[i].id_ == id) {
          return;
        }
      }

      if (candidate_pool_.size() < candidate_pool_.capacity()) {
        candidate_pool_.push_back({id, dist, true});
        std::push_heap(candidate_pool_.begin(), candidate_pool_.end());
      } else {
        std::pop_heap(candidate_pool_.begin(), candidate_pool_.end());
        candidate_pool_.back() = {id, dist, true};
        std::push_heap(candidate_pool_.begin(), candidate_pool_.end());
      }
    }

    /**
     * @brief Update the neighborhood.
     */
    template <typename T>
    void join(T callback) const {
      for (IDType i : nn_new_) {
        for (IDType j : nn_new_) {
          if (i < j) {
            callback(i, j);
          }
        }

        for (IDType j : nn_old_) {
          callback(i, j);
        }
      }
    }
  };

  std::vector<Nhood> graph_;                  ///<  graph
  std::shared_ptr<DistanceSpaceType> space_;  ///< space
  uint32_t dim_;                              ///< dimension of the vectors
  IDType vector_num_;                         ///< number of vectors
  uint32_t max_nbrs_;                         ///< max number of neighbors
  uint32_t selected_sample_num_ = 10;         ///< number of selected samples
  uint32_t radius_ = 100;                     ///< the radius of the neighborhood
  uint32_t iterations_ = 10;                  ///< number of iterations
  uint32_t candidate_pool_size_;              ///< length of candidate list
  uint32_t random_seed_ = 347;

  NndescentImpl(std::shared_ptr<DistanceSpaceType> &space, uint32_t k) {
    space_ = space;
    dim_ = space_->get_dim();
    vector_num_ = space_->get_data_num();
    this->max_nbrs_ = k;
    this->candidate_pool_size_ = k + 50;
  }

  /**
   * @brief Build the graph
   *
   * @return std::unique_ptr<Graph<DataType, IDType>>
   */
  auto build_graph(uint32_t thread_num = 1) -> std::unique_ptr<Graph<DataType, IDType>> {
    // init the graph
    init_graph();
    // descent to build the graph
    descent();

    auto final_graph = std::make_unique<Graph<DataType, IDType>>(space_->get_capacity(), max_nbrs_);
    // copy the graph
    for (IDType i = 0; i < vector_num_; ++i) {
      std::sort(graph_[i].candidate_pool_.begin(), graph_[i].candidate_pool_.end());
      for (uint32_t j = 0; j < max_nbrs_; j++) {
        final_graph->at(i, j) = graph_[i].candidate_pool_[j].id_;
      }
    }
    final_graph->eps_.push_back(0);
    std::vector<Nhood>().swap(graph_);

    return final_graph;
  }
  /**
   * @brief Initialize the graph.
   */
  void init_graph() {
    graph_.reserve(vector_num_);

    // init the graph by random generator.
    {
      std::mt19937 rng(random_seed_ * 6007);

      for (IDType i = 0; i < vector_num_; i++) {
        graph_.emplace_back(rng, selected_sample_num_, vector_num_);
      }
    }

    unsigned int num_cores = std::thread::hardware_concurrency();
    {
      std::mt19937 rng((random_seed_ * 7741) + num_cores);
      ThreadPool pool(num_cores);

      IDType per_core_num = (vector_num_ + num_cores - 1) / num_cores;
      for (uint32_t thread_id = 0; thread_id < num_cores; thread_id++) {
        pool.enqueue([thread_id, &rng, per_core_num, this]() -> auto {
          IDType start = thread_id * per_core_num;
          IDType end = std::min((thread_id + 1) * per_core_num, vector_num_);

          for (; start < end; start++) {
            std::vector<IDType> tmp(selected_sample_num_);
            gen_random(rng, tmp.data(), selected_sample_num_, vector_num_);

            for (uint32_t j = 0; j < selected_sample_num_; j++) {
              IDType id = tmp[j];
              if (id == start) {
                continue;
              }

              DistanceType dist = space_->get_distance(start, id);
              graph_[start].candidate_pool_.push_back({id, dist, true});
            }

            std::make_heap(graph_[start].candidate_pool_.begin(),
                           graph_[start].candidate_pool_.end());
            graph_[start].candidate_pool_.reserve(candidate_pool_size_);
          }
        });
      }
    }
  }

  /**
   * @brief Perform the NNDescent algorithm to build the graph.
   *
   * The NNDescent algorithm iteratively refines the graph by joining nodes and updating their
   * neighborhoods. During each iteration, the algorithm:
   * 1. Joins nodes to form connections based on the current nearest neighbors.
   * 2. Updates the neighborhoods of each node to reflect the new connections.
   * 3. Evaluates the recall of the current graph against a set of evaluation points to monitor
   *   the quality of the graph.
   *
   * The process is repeated for a specified number of iterations, and the recall is logged
   * after each iteration to track the improvement in the graph's accuracy.
   *
   * @note The algorithm uses a random number generator seeded with a combination of the random seed
   * and the number of hardware threads to ensure reproducibility and randomness.
   */
  void descent() {
    uint32_t num_eval = std::min(static_cast<uint64_t>(100), static_cast<uint64_t>(vector_num_));
    std::vector<IDType> eval_points(num_eval);

    std::vector<std::vector<IDType>> eval_gt(num_eval);
    std::mt19937 rng((random_seed_ * 6577) + std::thread::hardware_concurrency());

    gen_random(rng, eval_points.data(), num_eval, vector_num_);
    gen_eval_gt(eval_points, eval_gt);

    auto t1 = Timer();
    for (uint32_t iter = 1; iter <= iterations_; ++iter) {
      join();
      update();

      float recall = eval_recall(eval_points, eval_gt);
      LOG_INFO("NNDescent iter: [{}/{}], recall: {}", iter, iterations_, recall);
    }

    LOG_INFO("NNDescent cost: {}", t1.elapsed() / 1000 / 1000);
  }

  /**
   * @brief Join the graph during the NNDescent process.
   * The function utilizes multi-threading to parallelize the join process
   * and ensures that all nodes are properly connected in the graph.
   */
  void join() {
    auto t1 = Timer();

    // TODO(ljh): Use the thread_num parameter, not the hardware_concurrency
    unsigned int num_cores = std::thread::hardware_concurrency();
    ThreadPool thread_pool(num_cores);

    IDType per_num_cores = (vector_num_ + num_cores - 1) / num_cores;
    for (uint32_t i = 0; i < num_cores; i++) {
      thread_pool.enqueue([this, i, per_num_cores]() -> auto {
        IDType start = i * per_num_cores;
        IDType end = std::min((i + 1) * per_num_cores, vector_num_);

        for (; start < end; start++) {
          graph_[start].join([this](IDType item1, IDType item2) -> auto {
            if (item1 != item2) {
              DistanceType dist = space_->get_distance(item1, item2);
              graph_[item1].insert(item2, dist);
              graph_[item2].insert(item1, dist);
            }
          });
        }
      });
    }
    thread_pool.wait_until_all_tasks_completed(num_cores);

    LOG_INFO("Join cost: {}", t1.elapsed() / 1000 / 1000);
  }

  /**
   * @brief Update the graph during the NNDescent process.
   *
   * This function updates the graph by performing several steps:
   * 1. Clears the new and old neighbor lists for each node.
   * 2. Sorts and resizes the candidate pool for each node.
   * 3. Updates the neighbor lists based on the candidate pool.
   * 4. Merges the reverse neighbor lists into the main neighbor lists.
   *
   * The function utilizes multi-threading to parallelize the update process
   * and ensures thread safety using scoped locks.
   */
  void update() {
    auto t1 = Timer();

    // TODO(ljh): Use the thread_num parameter, not the hardware_concurrency
    unsigned int num_cores = std::thread::hardware_concurrency();
    ThreadPool thread_pool(num_cores);

    IDType per_num_cores = (vector_num_ + num_cores - 1) / num_cores;
    for (uint32_t i = 0; i < num_cores; i++) {
      thread_pool.enqueue([i, this, per_num_cores]() -> auto {
        for (IDType j = i * per_num_cores; j < (i + 1) * per_num_cores && j < vector_num_; j++) {
          std::vector<IDType>().swap(graph_[j].nn_new_);
          std::vector<IDType>().swap(graph_[j].nn_old_);
        }
      });
    }
    thread_pool.wait_until_all_tasks_completed(num_cores);

    thread_pool.reset_task();

    for (uint32_t i = 0; i < num_cores; i++) {
      thread_pool.enqueue([i, this, per_num_cores]() -> auto {
        for (IDType j = i * per_num_cores; j < (i + 1) * per_num_cores && j < vector_num_; j++) {
          auto &nn = graph_[j];
          std::sort(nn.candidate_pool_.begin(), nn.candidate_pool_.end());

          if (nn.candidate_pool_.size() > candidate_pool_size_) {
            nn.candidate_pool_.resize(candidate_pool_size_);
          }
          nn.candidate_pool_.reserve(candidate_pool_size_);

          auto maxl = std::min(nn.max_edge_ + selected_sample_num_,
                               static_cast<uint32_t>(nn.candidate_pool_.size()));
          int c = 0;
          uint32_t l = 0;

          while ((l < maxl) && (c < static_cast<int>(selected_sample_num_))) {
            if (nn.candidate_pool_[l].flag_) {
              ++c;
            }
            ++l;
          }
          nn.max_edge_ = l;
        }
      });
    }

    thread_pool.wait_until_all_tasks_completed(num_cores);

    {
      std::mt19937 rng((random_seed_ * 5081) + num_cores);

      thread_pool.reset_task();

      for (uint32_t i = 0; i < num_cores; ++i) {
        thread_pool.enqueue([&, i, per_num_cores]() -> auto {
          for (auto j = per_num_cores * i; j < per_num_cores * (i + 1) && j < vector_num_; ++j) {
            auto &node = graph_[j];
            auto &nn_new = node.nn_new_;
            auto &nn_old = node.nn_old_;

            for (uint32_t l = 0; l < node.max_edge_; ++l) {
              auto &nn = node.candidate_pool_[l];
              auto &other = graph_[nn.id_];

              if (nn.flag_) {
                nn_new.push_back(nn.id_);
                if (nn.distance_ > other.candidate_pool_.back().distance_) {
                  std::scoped_lock guard(other.lock_);
                  if (other.rnn_new_.size() < radius_) {
                    other.rnn_new_.push_back(j);
                  } else {
                    int pos = rng() % radius_;
                    other.rnn_new_[pos] = j;
                  }
                }
                nn.flag_ = false;
              } else {
                nn_old.push_back(nn.id_);
                if (nn.distance_ > other.candidate_pool_.back().distance_) {
                  std::scoped_lock guard(other.lock_);
                  if (other.rnn_old_.size() < radius_) {
                    other.rnn_old_.push_back(j);
                  } else {
                    int pos = rng() % radius_;
                    other.rnn_old_[pos] = j;
                  }
                }
              }
            }
            std::make_heap(node.candidate_pool_.begin(), node.candidate_pool_.end());
          }
        });
      }

      thread_pool.wait_until_all_tasks_completed(num_cores);
    }

    {
      thread_pool.reset_task();

      for (uint32_t i = 0; i < num_cores; ++i) {
        thread_pool.enqueue([this, i, per_num_cores]() -> auto {
          for (auto j = i * per_num_cores;
               j < (i * per_num_cores) + per_num_cores && j < vector_num_;
               ++j) {
            auto &nn_new = graph_[j].nn_new_;
            auto &nn_old = graph_[j].nn_old_;
            auto &rnn_new = graph_[j].rnn_new_;
            auto &rnn_old = graph_[j].rnn_old_;
            nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
            nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
            if (nn_old.size() > radius_ * 2) {
              nn_old.resize(radius_ * 2);
              nn_old.reserve(radius_ * 2);
            }
            std::vector<IDType>().swap(graph_[j].rnn_new_);
            std::vector<IDType>().swap(graph_[j].rnn_old_);
          }
        });
      }
      thread_pool.wait_until_all_tasks_completed(num_cores);
    }

    LOG_INFO("Update cost: {}", t1.elapsed() / 1000 / 1000);
  }

  /**
   * @brief Generate the ground truth.
   *
   * @param eval_set The evaluation set.
   * @param eval_gt The ground truth.
   */
  void gen_eval_gt(const std::vector<IDType> &eval_set, std::vector<std::vector<IDType>> &eval_gt) {
    auto t1 = Timer();

    // TODO(?): Use the thread_num parameter, not the hardware_concurrency
    unsigned int num_cores = std::thread::hardware_concurrency();
    ThreadPool thread_pool(num_cores);
    auto per_num_cores = (eval_set.size() + num_cores - 1) / num_cores;
    for (unsigned int thread_id = 0; thread_id < num_cores; ++thread_id) {
      thread_pool.enqueue([thread_id, per_num_cores, &eval_set, &eval_gt, this]() -> auto {
        for (uint32_t j = thread_id * per_num_cores;
             j < (thread_id + 1) * per_num_cores && j < eval_set.size();
             ++j) {
          std::vector<Neighbor<IDType>> tmp;

          for (IDType iter = 0; iter < vector_num_; ++iter) {
            if (eval_set[j] == iter) {
              continue;
            }
            DistanceType dist = space_->get_distance(eval_set[j], iter);
            tmp.push_back(Neighbor<IDType>(iter, dist, true));
          }

          std::partial_sort(tmp.begin(), tmp.begin() + max_nbrs_, tmp.end());
          for (uint32_t it = 0; it < max_nbrs_; ++it) {
            eval_gt[j].push_back(tmp[it].id_);
          }
        }
      });
    }
    thread_pool.wait_until_all_tasks_completed(num_cores);

    LOG_INFO("GenEvalGT cost: {}", t1.elapsed() / 1000 / 1000);
  }
  /**
   * @brief Evaluate the recall of the graph
   *
   * This function evaluates the recall of the graph by comparing the
   * generated neighbors with the ground truth neighbors.
   *
   * @param eval_set The set of generated neighbors for evaluation
   * @param eval_gt The ground truth neighbors for evaluation
   * @return float The recall value
   */
  auto eval_recall(const std::vector<IDType> &eval_set,
                   const std::vector<std::vector<IDType>> &eval_gt) -> float {
    auto t1 = Timer();

    float mean_acc = 0.0F;
    for (size_t i = 0; i < eval_set.size(); i++) {
      float acc = 0;
      std::vector<Neighbor<IDType>> &g = graph_[eval_set[i]].candidate_pool_;
      const std::vector<IDType> &v = eval_gt[i];
      for (size_t j = 0; j < g.size(); j++) {
        for (const auto &id : v) {
          if (g[j].id_ == id) {
            acc++;
            break;
          }
        }
      }
      mean_acc += acc / v.size();
    }

    LOG_INFO("Recall cost: {}", t1.elapsed() * 1.0 / 1000 / 1000);
    return mean_acc / eval_set.size();
  }
};

}  // namespace alaya
