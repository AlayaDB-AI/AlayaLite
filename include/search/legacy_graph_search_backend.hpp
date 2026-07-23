// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "executor/jobs/graph_search_job.hpp"
#include "search/vector_search_backend.hpp"
#include "space/rabitq_space.hpp"
#include "utils/query_utils.hpp"

namespace alaya {

template <typename DistanceSpaceType,
          typename BuildSpaceType,
          typename DataType,
          typename DistanceType,
          typename IDType>
class LegacyGraphCandidateCursor final : public CandidateCursor<IDType, DistanceType> {
 public:
  using SearchJob =
      GraphSearchJob<DistanceSpaceType, BuildSpaceType, DataType, DistanceType, IDType>;

  /** @brief Translate and own the request mask, then open the existing graph iterator. */
  LegacyGraphCandidateCursor(std::shared_ptr<SearchJob> job,
                             const VectorSearchRequest<DataType, IDType> &request)
      : blocked_(job->space_->get_data_num()) {
    if (!request.domain_.is_global_) {
      throw std::invalid_argument("legacy graph backend only supports the global domain");
    }
    if (request.accept_mask_ != nullptr) {
      for (size_t raw_id = 0; raw_id < job->space_->get_data_num(); ++raw_id) {
        if (!request.accept_mask_->accepts(static_cast<IDType>(raw_id))) {
          blocked_.set(raw_id);
        }
      }
      blocked_mask_ = &blocked_;
    }
    auto budget = std::max(request.topk_, request.candidate_budget_);
    iterator_ =
        job->make_vector_iterator(request.query_, SearchInfo{request.topk_, budget}, blocked_mask_);
  }

  /** @copydoc CandidateCursor::next_batch */
  [[nodiscard]] auto next_batch(size_t batch_size)
      -> CandidateBatch<IDType, DistanceType> override {
    CandidateBatch<IDType, DistanceType> result;
    result.candidates_.reserve(batch_size);
    while (result.candidates_.size() < batch_size) {
      auto candidate = iterator_->next();
      if (!candidate.has_value()) {
        result.exhausted_ = true;
        stats_.emitted_ += result.candidates_.size();
        return result;
      }
      result.candidates_.push_back({candidate->id_, candidate->distance_});
    }
    result.exhausted_ = !iterator_->has_next();
    stats_.emitted_ += result.candidates_.size();
    return result;
  }

  /** @copydoc CandidateCursor::stats */
  [[nodiscard]] auto stats() const -> CandidateCursorStats override { return stats_; }

 private:
  DynamicBitset blocked_;  ///< Owns the blocked mask referenced by the legacy iterator.
  const DynamicBitset *blocked_mask_ = nullptr;  ///< Null when every ID is accepted.
  std::unique_ptr<VectorIterator<IDType, DistanceType>> iterator_;  ///< Active graph traversal.
  CandidateCursorStats stats_{};  ///< Progress available from the legacy iterator API.
};

/**
 * @brief Adapts GraphSearchJob, including its RaBitQ branch, to VectorSearchBackend.
 *
 * This compatibility layer is the only Phase 1 component that knows GraphSearchJob's concrete API.
 */
template <typename DistanceSpaceType,
          typename BuildSpaceType = DistanceSpaceType,
          typename DataType = typename DistanceSpaceType::DataTypeAlias,
          typename DistanceType = typename DistanceSpaceType::DistanceTypeAlias,
          typename IDType = typename DistanceSpaceType::IDTypeAlias>
class LegacyGraphSearchBackend final : public VectorSearchBackend<DataType, IDType, DistanceType> {
 public:
  using SearchJob =
      GraphSearchJob<DistanceSpaceType, BuildSpaceType, DataType, DistanceType, IDType>;

  /** @brief Bind the adapter to an existing graph job. */
  explicit LegacyGraphSearchBackend(std::shared_ptr<SearchJob> job) : job_(std::move(job)) {
    if (job_ == nullptr) {
      throw std::invalid_argument("GraphSearchJob cannot be null");
    }
  }

  /** @copydoc VectorSearchBackend::capabilities */
  [[nodiscard]] auto capabilities() const -> SearchCapabilities override {
    return SearchCapabilities{
        .supports_accept_mask_ = true,
        .supports_candidate_cursor_ = true,
        .returns_approx_distance_ = false,
        .supports_partition_domain_ = false,
    };
  }

  /** @copydoc VectorSearchBackend::universe_size */
  [[nodiscard]] auto universe_size() const -> size_t override {
    return static_cast<size_t>(job_->space_->get_data_num());
  }

  /** @copydoc VectorSearchBackend::search */
  [[nodiscard]] auto search(const VectorSearchRequest<DataType, IDType> &request) const
      -> CandidateBatch<IDType, DistanceType> override {
    validate_request(request);
    CandidateBatch<IDType, DistanceType> result;
    if (request.topk_ == 0) {
      return result;
    }

    auto budget = std::max(request.topk_, request.candidate_budget_);
    std::vector<IDType> ids(request.topk_, std::numeric_limits<IDType>::max());
    std::vector<DistanceType> distances(request.topk_, std::numeric_limits<DistanceType>::max());
    auto blocked = make_blocked_mask(request.accept_mask_);
    auto *blocked_mask = request.accept_mask_ == nullptr ? nullptr : &blocked;

    if constexpr (is_rabitq_space_v<DistanceSpaceType>) {
      job_->rabitq_search_solo(request.query_,
                               request.topk_,
                               ids.data(),
                               distances.data(),
                               SearchInfo{request.topk_, budget},
                               blocked_mask);
    } else if (blocked_mask == nullptr) {
      job_->search_solo(const_cast<DataType *>(request.query_),
                        ids.data(),
                        distances.data(),
                        request.topk_,
                        budget);
    } else {
      job_->search_solo(const_cast<DataType *>(request.query_),
                        ids.data(),
                        SearchInfo{request.topk_, budget},
                        blocked_mask);
      fill_exact_distances(request.query_, ids, distances);
    }

    for (size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] == std::numeric_limits<IDType>::max()) {
        break;
      }
      result.candidates_.push_back({ids[i], distances[i]});
    }
    return result;
  }

  /** @copydoc VectorSearchBackend::open_cursor */
  [[nodiscard]] auto open_cursor(const VectorSearchRequest<DataType, IDType> &request) const
      -> std::unique_ptr<CandidateCursor<IDType, DistanceType>> override {
    validate_request(request);
    return std::make_unique<LegacyGraphCandidateCursor<DistanceSpaceType,
                                                       BuildSpaceType,
                                                       DataType,
                                                       DistanceType,
                                                       IDType>>(job_, request);
  }

  /** @copydoc VectorSearchBackend::exact_distance */
  [[nodiscard]] auto exact_distance(const DataType *query, IDType id) const
      -> DistanceType override {
    if constexpr (is_rabitq_space_v<DistanceSpaceType>) {
      return job_->space_->get_dist_func()(query,
                                           job_->space_->get_data_by_id(id),
                                           job_->space_->get_dim());
    } else if constexpr (std::is_same_v<DistanceSpaceType, BuildSpaceType>) {
      auto &exact_space = job_->build_space_ == nullptr ? *job_->space_ : *job_->build_space_;
      return exact_space.get_query_computer(query)(id);
    } else {
      return job_->build_space_->get_query_computer(query)(id);
    }
  }

 private:
  /** @brief Reject malformed requests and domains unsupported by the legacy graph. */
  void validate_request(const VectorSearchRequest<DataType, IDType> &request) const {
    if (request.query_ == nullptr && request.topk_ != 0) {
      throw std::invalid_argument("query cannot be null");
    }
    if (!request.domain_.is_global_) {
      throw std::invalid_argument("legacy graph backend only supports the global domain");
    }
  }

  /** @brief Translate an allow-style interface mask into GraphSearchJob's blocked bitset. */
  [[nodiscard]] auto make_blocked_mask(const IdMask<IDType> *accept_mask) const -> DynamicBitset {
    DynamicBitset blocked(job_->space_->get_data_num());
    if (accept_mask == nullptr) {
      return blocked;
    }
    for (size_t raw_id = 0; raw_id < job_->space_->get_data_num(); ++raw_id) {
      if (!accept_mask->accepts(static_cast<IDType>(raw_id))) {
        blocked.set(raw_id);
      }
    }
    return blocked;
  }

  /** @brief Fill exact distances for all valid IDs in a fixed-size result array. */
  void fill_exact_distances(const DataType *query,
                            const std::vector<IDType> &ids,
                            std::vector<DistanceType> &distances) const {
    for (size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] == std::numeric_limits<IDType>::max()) {
        break;
      }
      distances[i] = exact_distance(query, ids[i]);
    }
  }

  std::shared_ptr<SearchJob> job_;  ///< Existing graph job whose behavior is preserved.
};

}  // namespace alaya
