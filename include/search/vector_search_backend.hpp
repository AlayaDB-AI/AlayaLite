// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace alaya {

/** @brief Read-only result-admission mask expressed in internal IDs. */
template <typename IDType>
class IdMask {
 public:
  /** @brief Destroy the ID-mask implementation. */
  virtual ~IdMask() = default;

  /** @brief Return whether an ID may be emitted as a result. */
  [[nodiscard]] virtual auto accepts(IDType id) const -> bool = 0;
};

/**
 * @brief Search scope understood by the vector backend.
 *
 * An accept mask only controls emitted results and still permits graph traversal through rejected
 * nodes. A non-global domain denotes a physically separate graph/partition and may constrain
 * traversal itself.
 */
struct SearchDomain {
  bool is_global_ = true;      ///< True for the collection-wide graph.
  uint64_t partition_id_ = 0;  ///< Backend-defined partition ID when is_global_ is false.
};

/** @brief Features exposed by one vector-index backend instance. */
struct SearchCapabilities {
  bool supports_accept_mask_ = false;       ///< Can reject output IDs without pruning traversal.
  bool supports_candidate_cursor_ = false;  ///< Can continue an existing search incrementally.
  bool returns_approx_distance_ = false;    ///< Returned distances require exact reranking.
  bool supports_partition_domain_ = false;  ///< Can search a physically restricted domain.
};

/** @brief Vector-only request passed from hybrid planning to a backend. */
template <typename DataType, typename IDType>
struct VectorSearchRequest {
  const DataType *query_ = nullptr;  ///< Query vector; valid for the call/cursor lifetime.
  uint32_t topk_ = 0;                ///< Maximum number of results requested.
  uint32_t candidate_budget_ = 0;    ///< Backend traversal budget, such as ef or beam size.
  const IdMask<IDType> *accept_mask_ = nullptr;  ///< Optional output-admission predicate.
  SearchDomain domain_{};                        ///< Global graph or one materialized partition.
};

/** @brief One vector candidate with its backend-reported distance. */
template <typename IDType, typename DistanceType>
struct SearchCandidate {
  IDType id_{};              ///< Internal record ID.
  DistanceType distance_{};  ///< Distance used for final ordering.
};

/** @brief A bounded candidate response from direct search or a cursor. */
template <typename IDType, typename DistanceType>
struct CandidateBatch {
  std::vector<SearchCandidate<IDType, DistanceType>> candidates_;  ///< Ordered candidates.
  bool exhausted_ = true;  ///< True when this search cannot emit additional candidates.
};

/** @brief Observable progress for an incremental candidate cursor. */
struct CandidateCursorStats {
  size_t expanded_ = 0;  ///< Graph nodes expanded, or zero when a legacy backend cannot expose it.
  size_t emitted_ = 0;   ///< Candidates emitted by this cursor so far.
};

/** @brief Incremental candidate stream used only by adaptive hybrid plans. */
template <typename IDType, typename DistanceType>
class CandidateCursor {
 public:
  /** @brief Destroy the candidate cursor. */
  virtual ~CandidateCursor() = default;

  /** @brief Continue the existing traversal and return at most batch_size candidates. */
  [[nodiscard]] virtual auto next_batch(size_t batch_size)
      -> CandidateBatch<IDType, DistanceType> = 0;

  /** @brief Return cursor progress without advancing it. */
  [[nodiscard]] virtual auto stats() const -> CandidateCursorStats = 0;
};

/** @brief Graph-implementation-neutral vector query contract used by hybrid search. */
template <typename DataType, typename IDType, typename DistanceType>
class VectorSearchBackend {
 public:
  /** @brief Destroy the vector backend implementation. */
  virtual ~VectorSearchBackend() = default;

  /** @brief Return immutable capabilities for this backend instance. */
  [[nodiscard]] virtual auto capabilities() const -> SearchCapabilities = 0;

  /** @brief Return the exclusive upper bound of internal IDs visible to vector search. */
  [[nodiscard]] virtual auto universe_size() const -> size_t = 0;

  /** @brief Execute a one-shot vector search. */
  [[nodiscard]] virtual auto search(const VectorSearchRequest<DataType, IDType> &request) const
      -> CandidateBatch<IDType, DistanceType> = 0;

  /** @brief Open an incremental traversal for adaptive filtering. */
  [[nodiscard]] virtual auto open_cursor(const VectorSearchRequest<DataType, IDType> &request) const
      -> std::unique_ptr<CandidateCursor<IDType, DistanceType>> = 0;

  /** @brief Compute the exact distance between a query and one stored vector. */
  [[nodiscard]] virtual auto exact_distance(const DataType *query, IDType id) const
      -> DistanceType = 0;
};

}  // namespace alaya
