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
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "index/graph/graph.hpp"
#include "space/rabitq_space.hpp"
#include "utils/log.hpp"
#include "utils/prefetch.hpp"
#include "utils/query_utils.hpp"
#include "utils/rabitq_utils/search_utils/buffer.hpp"

namespace alaya {

template <typename IDType, typename DistanceType>
struct VectorCandidate {
  IDType id_{};
  DistanceType distance_{};
};

template <typename IDType, typename DistanceType>
class VectorIterator {
 public:
  virtual ~VectorIterator() = default;

  [[nodiscard]] virtual auto has_next() -> bool = 0;
  virtual auto next() -> std::optional<VectorCandidate<IDType, DistanceType>> = 0;

  void next_batch(size_t batch_size,
                  std::vector<IDType> &ids,
                  std::vector<DistanceType> &distances) {
    ids.clear();
    distances.clear();
    ids.reserve(batch_size);
    distances.reserve(batch_size);

    while (ids.size() < batch_size) {
      auto candidate = next();
      if (!candidate.has_value()) {
        break;
      }
      ids.push_back(candidate->id_);
      distances.push_back(candidate->distance_);
    }
  }
};

template <typename DistanceSpaceType,
          typename BuildSpaceType,
          typename DataType,
          typename DistanceType,
          typename IDType>
class GraphVectorIterator final : public VectorIterator<IDType, DistanceType> {
 public:
  using SearchQC = decltype(std::declval<DistanceSpaceType &>().get_query_computer(
      static_cast<const DataType *>(nullptr)));
  using ExactQC = decltype(std::declval<BuildSpaceType &>().get_query_computer(
      static_cast<const DataType *>(nullptr)));

  GraphVectorIterator(std::shared_ptr<DistanceSpaceType> space,
                      std::shared_ptr<BuildSpaceType> build_space,
                      std::shared_ptr<Graph<DataType, IDType>> graph,
                      const DataType *query,
                      uint32_t ef,
                      const DynamicBitset *blocked_mask)
      : space_(std::move(space)),
        build_space_(std::move(build_space)),
        graph_(std::move(graph)),
        search_qc_(space_->get_query_computer(query)),
        search_pool_(space_->get_data_num(), static_cast<int>(ef)),
        blocked_mask_(blocked_mask) {
    exact_qc_ = make_exact_qc(space_, build_space_, query);
    graph_->initialize_search(search_pool_, search_qc_);
  }

  [[nodiscard]] auto has_next() -> bool override { return prepare_next(); }

  auto next() -> std::optional<VectorCandidate<IDType, DistanceType>> override {
    if (!prepare_next()) {
      return std::nullopt;
    }
    auto candidate = pending_candidate_;
    pending_candidate_.reset();
    return candidate;
  }

 private:
  static auto make_exact_qc(const std::shared_ptr<DistanceSpaceType> &space,
                            const std::shared_ptr<BuildSpaceType> &build_space,
                            const DataType *query) -> std::unique_ptr<ExactQC> {
    if constexpr (std::is_same_v<DistanceSpaceType, BuildSpaceType>) {
      if (build_space != nullptr) {
        return std::make_unique<ExactQC>(*build_space, query);
      }
      return std::make_unique<ExactQC>(*space, query);
    } else {
      return std::make_unique<ExactQC>(*build_space, query);
    }
  }

  [[nodiscard]] auto prepare_next() -> bool {
    if (pending_candidate_.has_value()) {
      return true;
    }

    auto *space = space_.get();
    auto *graph = graph_.get();

    while (search_pool_.has_next()) {
      auto node = search_pool_.pop();
      for (uint32_t i = 0; i < graph->max_nbrs_; ++i) {
        auto neighbor = graph->at(node, i);
        if (neighbor == static_cast<IDType>(-1)) {
          break;
        }

        if (search_pool_.vis_.get(neighbor)) {
          continue;
        }
        search_pool_.vis_.set(neighbor);

        auto prefetch_index = i + 3;
        if (prefetch_index < graph->max_nbrs_) {
          auto prefetch_id = graph->at(node, prefetch_index);
          if (prefetch_id != static_cast<IDType>(-1)) {
            space->prefetch_by_id(prefetch_id);
          }
        }

        search_pool_.insert(neighbor, search_qc_(neighbor));
      }

      if (blocked_mask_ != nullptr && blocked_mask_->get(node)) {
        continue;
      }

      pending_candidate_ = VectorCandidate<IDType, DistanceType>{node, (*exact_qc_)(node)};
      return true;
    }

    return false;
  }

  std::shared_ptr<DistanceSpaceType> space_;
  std::shared_ptr<BuildSpaceType> build_space_;
  std::shared_ptr<Graph<DataType, IDType>> graph_;
  SearchQC search_qc_;
  std::unique_ptr<ExactQC> exact_qc_;
  LinearPool<DistanceType, IDType> search_pool_;
  const DynamicBitset *blocked_mask_ = nullptr;
  std::optional<VectorCandidate<IDType, DistanceType>> pending_candidate_;
};

template <typename DistanceSpaceType, typename DataType, typename DistanceType, typename IDType>
class RaBitQVectorIterator final : public VectorIterator<IDType, DistanceType> {
 public:
  using QueryComputer = decltype(std::declval<const DistanceSpaceType &>().get_query_computer(
      static_cast<const DataType *>(nullptr)));

  RaBitQVectorIterator(std::shared_ptr<DistanceSpaceType> space,
                       const DataType *query,
                       uint32_t ef,
                       const DynamicBitset *blocked_mask)
      : space_(std::move(space)),
        query_(query),
        search_pool_(ef),
        expanded_(space_->get_data_num()),
        q_computer_(space_->get_query_computer(query)),
        blocked_mask_(blocked_mask),
        dist_func_(space_->get_dist_func()),
        dim_(space_->get_dim()) {
    auto entry = space_->get_ep();
    search_pool_.insert(entry, std::numeric_limits<DistanceType>::max());
    mem_prefetch_l1(space_->get_data_by_id(entry), 10);
  }

  [[nodiscard]] auto has_next() -> bool override { return prepare_next(); }

  auto next() -> std::optional<VectorCandidate<IDType, DistanceType>> override {
    if (!prepare_next()) {
      return std::nullopt;
    }
    auto candidate = pending_candidate_;
    pending_candidate_.reset();
    return candidate;
  }

 private:
  void load_supplement_candidates() {
    if (supplement_built_) {
      return;
    }
    supplement_built_ = true;

    std::unordered_set<IDType> supplement_seen;
    supplement_seen.reserve(expanded_nodes_.size() * RaBitQSpace<>::kDegreeBound);
    for (auto node : expanded_nodes_) {
      auto *neighbors = space_->get_edges(node);
      for (uint32_t i = 0; i < RaBitQSpace<>::kDegreeBound; ++i) {
        auto neighbor = neighbors[i];
        if (expanded_.get(neighbor)) {
          continue;
        }
        if (supplement_seen.insert(neighbor).second) {
          supplement_candidates_.push_back(neighbor);
        }
      }
    }
  }

  [[nodiscard]] auto prepare_next() -> bool {
    if (pending_candidate_.has_value()) {
      return true;
    }

    auto *space = space_.get();
    while (search_pool_.has_next()) {
      auto current_node = search_pool_.pop();
      if (expanded_.get(current_node)) {
        continue;
      }

      expanded_.set(current_node);
      expanded_nodes_.push_back(current_node);
      q_computer_.load_centroid(current_node);

      const IDType *neighbors = space->get_edges(current_node);
      for (size_t i = 0; i < RaBitQSpace<>::kDegreeBound; ++i) {
        auto neighbor = neighbors[i];
        if (expanded_.get(neighbor)) {
          continue;
        }

        auto estimated_distance = q_computer_(i);
        if (search_pool_.is_full(estimated_distance)) {
          continue;
        }

        search_pool_.insert(neighbor, estimated_distance);
        mem_prefetch_l2(space->get_data_by_id(search_pool_.next_id()), 10);
      }

      if (blocked_mask_ != nullptr && blocked_mask_->get(current_node)) {
        continue;
      }

      pending_candidate_ =
          VectorCandidate<IDType, DistanceType>{current_node, q_computer_.get_exact_qr_c_dist()};
      return true;
    }

    if (!supplement_built_) {
      load_supplement_candidates();
    }

    while (supplement_cursor_ < supplement_candidates_.size()) {
      auto id = supplement_candidates_[supplement_cursor_++];
      if (blocked_mask_ != nullptr && blocked_mask_->get(id)) {
        continue;
      }
      pending_candidate_ =
          VectorCandidate<IDType, DistanceType>{id,
                                                dist_func_(query_,
                                                           space->get_data_by_id(id),
                                                           dim_)};
      ++supplement_emitted_count_;
      return true;
    }

    if (!supplement_logged_) {
      LOG_DEBUG("rabitq_vector_iterator: supplement produced {} valid results",
                supplement_emitted_count_);
      supplement_logged_ = true;
    }

    return false;
  }

  std::shared_ptr<DistanceSpaceType> space_;
  const DataType *query_ = nullptr;
  SearchBuffer<DistanceType> search_pool_;
  DynamicBitset expanded_;
  QueryComputer q_computer_;
  const DynamicBitset *blocked_mask_ = nullptr;
  DistFuncRaBitQ<DataType, DistanceType> dist_func_;
  uint32_t dim_ = 0;
  bool supplement_built_ = false;
  bool supplement_logged_ = false;
  std::vector<IDType> expanded_nodes_;
  std::vector<IDType> supplement_candidates_;
  size_t supplement_cursor_ = 0;
  size_t supplement_emitted_count_ = 0;
  std::optional<VectorCandidate<IDType, DistanceType>> pending_candidate_;
};

}  // namespace alaya
