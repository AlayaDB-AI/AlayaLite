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
#include <unordered_set>
#include <vector>

#include "../../index/graph/graph.hpp"
#include "../../space/space_concepts.hpp"
#include "./graph_search_job.hpp"
#include "./job_context.hpp"

namespace alaya {

template <typename DistanceSpaceType,
          typename DataType = typename DistanceSpaceType::DataTypeAlias,
          typename DistanceType = typename DistanceSpaceType::DistanceTypeAlias,
          typename IDType = typename DistanceSpaceType::IDTypeAlias>
  requires Space<DistanceSpaceType, DataType, DistanceType, IDType>
class GraphUpdateJob {
 public:
  std::shared_ptr<DistanceSpaceType> space_ = nullptr;  ///< The is a data manager interface .
  std::shared_ptr<Graph<DataType, IDType>> graph_ = nullptr;  ///< The search graph.
  std::shared_ptr<GraphSearchJob<DistanceSpaceType>> search_job_ = nullptr;  ///< The search job
  std::shared_ptr<JobContext<IDType>> job_context_;  ///< The shared job context

  explicit GraphUpdateJob(
      std::shared_ptr<GraphSearchJob<DistanceSpaceType>> search_job)
      : search_job_(search_job),
        space_(search_job->space_),
        graph_(search_job->graph_),
        job_context_(search_job->job_context_) {}

  auto insert(DataType *query, IDType *ids, uint32_t ef) -> IDType {
    std::vector<IDType> search_results(graph_->max_nbrs_, -1);

    search_job_->search_solo(query, graph_->max_nbrs_, search_results.data(), ef);
    auto node_id = graph_->insert(search_results.data());
    space_->insert(query);

    for (IDType i = 0; i < graph_->max_nbrs_; i++) {
      auto invert_node = search_results[i];

      if (invert_node != -1) {
        job_context_->inserted_edges_[invert_node].push_back(node_id);
      }
    }
    return node_id;
  }

  auto insert_and_update(DataType *query, uint32_t ef) -> IDType {
    std::vector<IDType> search_results(graph_->max_nbrs_, -1);

    search_job_->search_solo(query, graph_->max_nbrs_, search_results.data(),
                             ef);
    auto node_id = graph_->insert(search_results.data());
    if (node_id == -1) {
      assert(space_->insert(query) == -1);
      return -1;
    }
    space_->insert(query);

    for (IDType i = 0; i < graph_->max_nbrs_; i++) {
      auto invert_node = search_results[i];

      if (invert_node != -1) {
        job_context_->inserted_edges_[invert_node].push_back(node_id);
      }
    }
    for (const auto &[k, v] : job_context_->inserted_edges_) {
      update(k);
    }
    job_context_->inserted_edges_.clear();
    return node_id;
  }

  auto remove(IDType node_id) -> void {
    auto nbrs = graph_->edges(node_id);
    for (IDType i = 0; i < graph_->max_nbrs_; i++) {
      auto nbr = nbrs[i];
      if (nbr == -1) {
        break;
      }
      job_context_->removed_node_nbrs_[node_id].push_back(nbr);
    }
    job_context_->removed_vertices_.insert(node_id);
    graph_->remove(node_id);
    space_->remove(node_id);
  }

  auto update(IDType node_id) -> void {
    std::unordered_set<IDType> candidate_nbrs;
    auto current_edges = graph_->edges(node_id);
    for (IDType i = 0; i < graph_->max_nbrs_; i++) {
      auto nbr = current_edges[i];
      if (nbr == -1) {
        break;
      }
      if (job_context_->removed_vertices_.count(nbr)) {
        for (auto &second_hop_nbr : job_context_->removed_node_nbrs_.at(nbr)) {
          candidate_nbrs.insert(second_hop_nbr);
        }
      }
      candidate_nbrs.insert(nbr);
    }
    if (job_context_->inserted_edges_.count(node_id)) {
        for (auto inserted_nbr : job_context_->inserted_edges_.at(node_id)) {
            candidate_nbrs.insert(inserted_nbr);
        }
    }
    auto handler = space_->get_query_computer(node_id);
    LinearPool<DistanceType, IDType> pool(space_->get_data_num(),
                                          graph_->max_nbrs_);
    for (auto &nbr : candidate_nbrs) {
      auto dist = handler(nbr);
      pool.insert(nbr, dist);
    }

    std::vector<IDType> updated_edges(graph_->max_nbrs_);
    for (IDType i = 0; i < graph_->max_nbrs_ && i < pool.size(); i++) {
      updated_edges[i] = pool.id(i);
    }
    graph_->update(node_id, updated_edges.data());
  }
};
}  // namespace alaya
