// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "../../index/graph/graph.hpp"
#include "../../space/space_concepts.hpp"
#include "../../utils/scalar_data.hpp"
#include "./graph_search_job.hpp"
#include "./job_context.hpp"

namespace alaya {

template <typename DistanceSpaceType,
          typename BuildSpaceType = DistanceSpaceType,
          typename DataType = typename DistanceSpaceType::DataTypeAlias,
          typename DistanceType = typename DistanceSpaceType::DistanceTypeAlias,
          typename IDType = typename DistanceSpaceType::IDTypeAlias>
  requires Space<DistanceSpaceType> && Space<BuildSpaceType>
class GraphUpdateJob {
 public:
  std::shared_ptr<GraphSearchJob<DistanceSpaceType, BuildSpaceType>> search_job_ =
      nullptr;                                                ///< The search job
  std::shared_ptr<DistanceSpaceType> space_ = nullptr;        ///< The is a data manager interface .
  std::shared_ptr<BuildSpaceType> build_space_ = nullptr;     ///< Build space for rerank/update.
  std::shared_ptr<Graph<DataType, IDType>> graph_ = nullptr;  ///< The search graph.
  std::shared_ptr<JobContext<IDType>> job_context_;           ///< The shared job context

  explicit GraphUpdateJob(
      std::shared_ptr<GraphSearchJob<DistanceSpaceType, BuildSpaceType>> search_job)
      : search_job_(search_job),
        space_(search_job->space_),
        build_space_(search_job->build_space_),
        graph_(search_job->graph_),
        job_context_(search_job->job_context_) {}

  auto insert(DataType *query, IDType *ids, uint32_t ef) -> IDType {
    uint32_t search_size = graph_->max_nbrs_;
    std::vector<IDType> search_results(search_size, -1);

    // Use search_solo_updated for graph update (returns approximate ef results)
    search_job_->search_solo_updated(query, search_results.data(), ef, search_size);
    auto node_id = graph_->insert(search_results.data());
    if (node_id == invalid_id()) {
      if constexpr (!std::is_same_v<DistanceSpaceType, BuildSpaceType>) {
        assert(build_space_->insert(query, nullptr) == invalid_id());
      }
      assert(insert_search_space(query, nullptr) == invalid_id());
      return invalid_id();
    }

    try {
      insert_spaces(query, nullptr, node_id);
    } catch (...) {
      graph_->remove(node_id);
      throw;
    }

    for (IDType i = 0; i < search_size; i++) {
      auto invert_node = search_results[i];

      if (invert_node != static_cast<IDType>(-1)) {
        job_context_->inserted_edges_[invert_node].push_back(node_id);
        *(ids + i) = invert_node;
      }
    }
    return node_id;
  }

  auto insert_and_update(DataType *query, uint32_t ef, const ScalarData *scalar_data = nullptr)
      -> IDType {
    uint32_t search_size = graph_->max_nbrs_;
    std::vector<IDType> search_results(search_size, static_cast<IDType>(-1));

    // Use search_solo_updated for graph update (returns approximate ef results)
    search_job_->search_solo_updated(query, search_results.data(), ef, search_size);
    auto node_id = graph_->insert(search_results.data());
    if (node_id == static_cast<IDType>(-1)) {
      if constexpr (!std::is_same_v<DistanceSpaceType, BuildSpaceType>) {
        assert(build_space_->insert(query, nullptr) == invalid_id());
      }
      assert(insert_search_space(query, scalar_data) == invalid_id());
      return invalid_id();
    }

    try {
      insert_spaces(query, scalar_data, node_id);
    } catch (...) {
      graph_->remove(node_id);
      throw;
    }

    for (IDType i = 0; i < search_size; i++) {
      auto invert_node = search_results[i];

      if (invert_node != static_cast<IDType>(-1)) {
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
      if (nbr == static_cast<IDType>(-1)) {
        break;
      }
      job_context_->removed_node_nbrs_[node_id].push_back(nbr);
    }
    job_context_->removed_vertices_.insert(node_id);
    graph_->remove(node_id);
    if constexpr (!std::is_same_v<DistanceSpaceType, BuildSpaceType>) {
      build_space_->remove(node_id);
    }
    space_->remove(node_id);
  }

  /**
   * @brief Remove a node by its item_id
   * @param item_id The item_id to remove
   * @note This method is only available for spaces that support scalar data
   */
  auto remove(const std::string &item_id) -> void {
    if constexpr (DistanceSpaceType::has_scalar_data) {
      auto internal_id = space_->remove(item_id);  // Space handles lookup and deletion
      if constexpr (!std::is_same_v<DistanceSpaceType, BuildSpaceType>) {
        build_space_->remove(internal_id);
      }
      auto nbrs = graph_->edges(internal_id);
      for (IDType i = 0; i < graph_->max_nbrs_; i++) {
        auto nbr = nbrs[i];
        if (nbr == static_cast<IDType>(-1)) {
          break;
        }
        job_context_->removed_node_nbrs_[internal_id].push_back(nbr);
      }
      job_context_->removed_vertices_.insert(internal_id);
      graph_->remove(internal_id);
    } else {
      throw std::runtime_error("Remove by item_id is not supported for spaces without scalar data");
    }
  }

  auto update(IDType node_id) -> void {
    std::unordered_set<IDType> candidate_nbrs;
    auto current_edges = graph_->edges(node_id);
    for (IDType i = 0; i < graph_->max_nbrs_; i++) {
      auto nbr = current_edges[i];
      if (nbr == static_cast<IDType>(-1)) {
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
    LinearPool<DistanceType, IDType> pool(space_->get_data_num(), graph_->max_nbrs_);
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

 private:
  static constexpr auto invalid_id() -> IDType { return std::numeric_limits<IDType>::max(); }

  auto insert_search_space(DataType *query, const ScalarData *scalar_data) -> IDType {
    if constexpr (DistanceSpaceType::has_scalar_data) {
      return space_->insert(query, scalar_data);
    } else {
      return space_->insert(query, nullptr);
    }
  }

  auto insert_spaces(DataType *query, const ScalarData *scalar_data, IDType expected_id) -> void {
    if constexpr (std::is_same_v<DistanceSpaceType, BuildSpaceType>) {
      auto inserted_id = insert_search_space(query, scalar_data);
      if (inserted_id != expected_id) {
        if (inserted_id != invalid_id()) {
          space_->remove(inserted_id);
        }
        throw std::runtime_error("Search/build space ID drift detected during insert");
      }
      return;
    } else {
      auto build_id = build_space_->insert(query, nullptr);
      if (build_id != expected_id) {
        if (build_id != invalid_id()) {
          build_space_->remove(build_id);
        }
        throw std::runtime_error("Build space ID drift detected during insert");
      }

      try {
        auto search_id = insert_search_space(query, scalar_data);
        if (search_id != expected_id) {
          if (search_id != invalid_id()) {
            space_->remove(search_id);
          }
          build_space_->remove(build_id);
          throw std::runtime_error("Search space ID drift detected during insert");
        }
      } catch (...) {
        build_space_->remove(build_id);
        throw;
      }
    }
  }
};
}  // namespace alaya
