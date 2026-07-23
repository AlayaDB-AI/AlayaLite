// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "executor/search_info.hpp"
#include "search/blocked_bitset_id_mask.hpp"
#include "search/vector_search_backend.hpp"
#include "utils/log.hpp"
#include "utils/metadata_filter_matcher.hpp"

namespace alaya {

/**
 * @brief Selects and executes scalar-vector hybrid strategies over a vector backend contract.
 *
 * Scalar predicate compilation and residual evaluation belong to MetadataFilterExecutor. This
 * planner only chooses between exact, post-filter, mask-pushdown and incremental candidate
 * execution; it has no knowledge of graph layout, vector quantization, or Space ownership.
 */
template <typename DataType, typename IDType, typename DistanceType>
class HybridSearchPlanner {
 public:
  static constexpr float kKnnBFFilterThreshold =
      0.93F;                                       ///< Exact search when filter rejects >= 93%.
  static constexpr float kBFTopkThreshold = 0.5F;  ///< Exact search when top-k covers >= 50%.
  static constexpr double kPostFilterPassRateThreshold =
      0.70;  ///< Post-filter only when an exact indexed predicate accepts at least 70%.
  static constexpr double kPostFilterOversamplingFactor =
      1.25;  ///< Safety margin over top-k divided by the known predicate pass rate.
  static constexpr uint32_t kPostFilterMaxAttempts =
      3;  ///< Maximum independent unmasked ANN requests before safe fallback.

  /** @brief Strategy selected for one hybrid query. */
  enum class Mode : uint8_t {
    kPlainSearch,
    kPostFilter,
    kBitsetPrefilter,
    kIterativeFilter,
    kIndexedExact,
  };

  /** @brief Planner decisions and runtime fallback information for one query. */
  struct PlanStats {
    Mode initial_mode_ = Mode::kPlainSearch;   ///< Mode selected before runtime fallback.
    Mode executed_mode_ = Mode::kPlainSearch;  ///< Mode that produced the final results.
    size_t matched_count_ = 0;                 ///< Rows accepted by the scalar filter, when known.
    size_t data_count_ = 0;                    ///< Internal-ID universe exposed by the backend.
    double pass_rate_ = 0.0;                   ///< matched_count / data_count, when known.
    uint32_t requested_ef_ = 0;                ///< Caller-provided candidate traversal budget.
    uint32_t effective_ef_ = 0;                ///< Candidate budget after selectivity adjustment.
    uint32_t fanout_ = 0;                      ///< Physical partitions searched; zero for global.
    uint32_t result_count_ = 0;                ///< Valid results produced by the query.
    size_t post_filter_candidates_examined_ =
        0;  ///< Candidate rows scalar-tested across all post-filter attempts.
    uint32_t post_filter_retry_count_ =
        0;                              ///< Additional post-filter ANN requests after the first.
    bool matched_count_known_ = false;  ///< Whether matched_count and pass_rate are valid.
    bool fallback_ = false;             ///< Whether execution switched modes at runtime.
    std::string fallback_reason_;       ///< Stable reason code; empty without fallback.
  };

  /** @brief Vector candidates plus the plan observation produced by one execution. */
  struct ExecutionResult {
    std::vector<SearchCandidate<IDType, DistanceType>> candidates_;  ///< Final ordered neighbors.
    PlanStats stats_;  ///< Selection and fallback data.
  };

  using Backend = VectorSearchBackend<DataType, IDType, DistanceType>;

  /** @brief Bind strategy execution to one vector-index backend instance. */
  explicit HybridSearchPlanner(std::shared_ptr<Backend> backend) : backend_(std::move(backend)) {
    if (backend_ == nullptr) {
      throw std::invalid_argument("VectorSearchBackend cannot be null");
    }
  }

  /** @brief Return the vector backend's internal-ID universe. */
  [[nodiscard]] auto universe_size() const -> size_t { return backend_->universe_size(); }

  /** @brief Return the stable telemetry name for a strategy. */
  [[nodiscard]] static auto mode_name(Mode mode) -> const char * {
    switch (mode) {
      case Mode::kPlainSearch:
        return "plain_search";
      case Mode::kPostFilter:
        return "post_filter";
      case Mode::kBitsetPrefilter:
        return "bitset_prefilter";
      case Mode::kIterativeFilter:
        return "iterative_filter";
      case Mode::kIndexedExact:
        return "indexed_exact";
    }
    return "unknown";
  }

  /** @brief Select the initial strategy from predicate shape, selectivity and caller hint. */
  [[nodiscard]] auto build_search_mode(const MetadataFilterExecutor<IDType> &filter_executor,
                                       const SearchInfo &search_info) const -> Mode {
    if (filter_executor.is_trivially_true()) {
      return Mode::kPlainSearch;
    }

    switch (search_info.filter_exec_hint_) {
      case FilterExecHint::kAuto:
        if (filter_executor.has_index_fast_path()) {
          auto indexed_count = filter_executor.indexed_count();
          if (filter_executor.index_fast_path_is_exact() &&
              indexed_count == backend_->universe_size()) {
            return Mode::kPlainSearch;
          }
          if (should_use_brute_force_search(search_info, indexed_count)) {
            return Mode::kIndexedExact;
          }
          if (filter_executor.index_fast_path_is_exact() && should_use_post_filter(indexed_count)) {
            return Mode::kPostFilter;
          }
        }
        return Mode::kBitsetPrefilter;
      case FilterExecHint::kIterativeFilter:
        return Mode::kIterativeFilter;
      case FilterExecHint::kDisableIterative:
        return Mode::kBitsetPrefilter;
    }
    return Mode::kBitsetPrefilter;
  }

  /** @brief Build initial plan statistics without forcing a scalar full scan. */
  [[nodiscard]] auto make_plan_stats(Mode mode,
                                     const SearchInfo &search_info,
                                     const MetadataFilterExecutor<IDType> &filter_executor) const
      -> PlanStats {
    PlanStats stats;
    stats.initial_mode_ = mode;
    stats.executed_mode_ = mode;
    stats.data_count_ = backend_->universe_size();
    stats.requested_ef_ = search_info.ef_;
    stats.effective_ef_ = search_info.ef_;

    if (filter_executor.is_trivially_true()) {
      stats.matched_count_ = stats.data_count_;
      stats.matched_count_known_ = true;
    } else if (filter_executor.has_index_fast_path() &&
               filter_executor.index_fast_path_is_exact()) {
      stats.matched_count_ = filter_executor.indexed_count();
      stats.matched_count_known_ = true;
    }
    update_pass_rate(stats);
    return stats;
  }

  /**
   * @brief Execute the selected hybrid strategy and return ordered internal-ID candidates.
   * @param query Raw query vector valid for the duration of this call.
   * @param search_info Requested top-k, candidate budget and filtering hint.
   * @param filter_executor Generation-stable scalar predicate executor.
   * @param search_name Name used in diagnostic logging.
   */
  [[nodiscard]] auto execute(const DataType *query,
                             const SearchInfo &search_info,
                             const MetadataFilterExecutor<IDType> &filter_executor,
                             const char *search_name) const -> ExecutionResult {
    if (query == nullptr) {
      throw std::invalid_argument(std::string(search_name) + ": query cannot be null");
    }
    validate_search_info(search_info, search_name);
    if (filter_executor.data_num() != backend_->universe_size()) {
      throw std::invalid_argument("scalar and vector internal-ID universes do not match");
    }

    auto mode = build_search_mode(filter_executor, search_info);
    ExecutionResult result;
    result.stats_ = make_plan_stats(mode, search_info, filter_executor);
    LOG_DEBUG("{}: plan={}, topk={}, ef={}, hint={}",
              search_name,
              mode_name(mode),
              search_info.topk_,
              search_info.ef_,
              static_cast<int>(search_info.filter_exec_hint_));

    switch (mode) {
      case Mode::kPlainSearch:
        result.candidates_ = execute_plain_search(query, search_info);
        break;
      case Mode::kPostFilter:
        result.candidates_ =
            execute_post_filter(query, search_info, filter_executor, search_name, result.stats_);
        break;
      case Mode::kIndexedExact:
        result.candidates_ =
            execute_brute_force_filter(query, search_info.topk_, filter_executor, search_name);
        break;
      case Mode::kBitsetPrefilter:
        result.candidates_ = execute_bitset_prefilter(query,
                                                      search_info,
                                                      filter_executor,
                                                      search_name,
                                                      result.stats_);
        break;
      case Mode::kIterativeFilter:
        result.candidates_ = execute_iterative_filter(query,
                                                      search_info,
                                                      filter_executor,
                                                      search_name,
                                                      result.stats_);
        break;
    }
    result.stats_.result_count_ = static_cast<uint32_t>(result.candidates_.size());
    return result;
  }

  /** @brief Compute exact filtered top-k independently from ANN backend capabilities. */
  [[nodiscard]] auto execute_brute_force_filter(
      const DataType *query,
      uint32_t topk,
      const MetadataFilterExecutor<IDType> &filter_executor,
      const char *search_name) const -> std::vector<SearchCandidate<IDType, DistanceType>> {
    if (query == nullptr) {
      throw std::invalid_argument(std::string(search_name) + ": query cannot be null");
    }
    if (topk == 0) {
      throw std::invalid_argument(std::string(search_name) + ": topk must be > 0");
    }

    BoundedCandidates result_pool(topk);
    auto add_if_matched = [&](IDType id) {
      if (filter_executor.match(id)) {
        result_pool.insert({id, backend_->exact_distance(query, id)});
      }
    };

    if (filter_executor.has_index_fast_path()) {
      filter_executor.visit_index_fast_path_ids(add_if_matched);
      LOG_DEBUG("{}: exact_filter indexed_candidates={}, results={}, requested={}",
                search_name,
                filter_executor.indexed_count(),
                result_pool.size(),
                topk);
    } else {
      std::vector<IDType> ids;
      std::vector<uint8_t> matches;
      constexpr size_t kBatchSize = 1024;
      ids.reserve(kBatchSize);
      for (size_t begin = 0; begin < backend_->universe_size(); begin += kBatchSize) {
        ids.clear();
        auto end = std::min(backend_->universe_size(), begin + kBatchSize);
        for (size_t raw_id = begin; raw_id < end; ++raw_id) {
          ids.push_back(static_cast<IDType>(raw_id));
        }
        filter_executor.eval_offsets(ids, matches);
        for (size_t i = 0; i < ids.size(); ++i) {
          if (matches[i] != 0) {
            result_pool.insert({ids[i], backend_->exact_distance(query, ids[i])});
          }
        }
      }
      LOG_DEBUG("{}: exact_filter results={}, requested={}", search_name, result_pool.size(), topk);
    }
    return result_pool.take_ordered();
  }

 private:
  struct CandidateWorseFirst {
    /** @brief Keep the largest distance at the priority-queue top. */
    [[nodiscard]] auto operator()(const SearchCandidate<IDType, DistanceType> &left,
                                  const SearchCandidate<IDType, DistanceType> &right) const
        -> bool {
      if (left.distance_ != right.distance_) {
        return left.distance_ < right.distance_;
      }
      return left.id_ < right.id_;
    }
  };

  /** @brief Bounded exact-distance max heap used by cursor and brute-force strategies. */
  class BoundedCandidates {
   public:
    /** @brief Set the maximum number of retained candidates. */
    explicit BoundedCandidates(size_t capacity) : capacity_(capacity) {}

    /** @brief Retain one candidate when it belongs to the best capacity elements. */
    void insert(SearchCandidate<IDType, DistanceType> candidate) {
      if (capacity_ == 0) {
        return;
      }
      if (heap_.size() < capacity_) {
        heap_.push(std::move(candidate));
        return;
      }
      const auto &worst = heap_.top();
      if (candidate.distance_ > worst.distance_ ||
          (candidate.distance_ == worst.distance_ && candidate.id_ >= worst.id_)) {
        return;
      }
      heap_.pop();
      heap_.push(std::move(candidate));
    }

    /** @brief Return the number of currently retained candidates. */
    [[nodiscard]] auto size() const -> size_t { return heap_.size(); }

    /** @brief Drain retained candidates into nearest-first order. */
    [[nodiscard]] auto take_ordered() -> std::vector<SearchCandidate<IDType, DistanceType>> {
      std::vector<SearchCandidate<IDType, DistanceType>> result;
      result.reserve(heap_.size());
      while (!heap_.empty()) {
        result.push_back(heap_.top());
        heap_.pop();
      }
      std::reverse(result.begin(), result.end());
      return result;
    }

   private:
    size_t capacity_ = 0;  ///< Maximum candidates retained by this heap.
    std::priority_queue<SearchCandidate<IDType, DistanceType>,
                        std::vector<SearchCandidate<IDType, DistanceType>>,
                        CandidateWorseFirst>
        heap_;  ///< Worst retained candidate is removed first.
  };

  /** @brief Reject malformed vector search parameters. */
  static void validate_search_info(const SearchInfo &search_info, const char *search_name) {
    if (search_info.topk_ == 0) {
      throw std::invalid_argument(std::string(search_name) + ": topk must be > 0");
    }
    if (search_info.ef_ < search_info.topk_) {
      throw std::invalid_argument(std::string(search_name) + ": ef must be >= topk");
    }
  }

  /** @brief Run unfiltered vector search and normalize approximate backend distances if needed. */
  [[nodiscard]] auto execute_plain_search(const DataType *query,
                                          const SearchInfo &search_info) const
      -> std::vector<SearchCandidate<IDType, DistanceType>> {
    auto batch = backend_->search(VectorSearchRequest<DataType, IDType>{
        .query_ = query,
        .topk_ = search_info.topk_,
        .candidate_budget_ = search_info.ef_,
    });
    return normalize_candidates(query, std::move(batch.candidates_), search_info.topk_, nullptr);
  }

  /**
   * @brief Expand unmasked ANN candidates, then batch-evaluate a high-pass scalar predicate.
   *
   * Each retry starts a larger one-shot request because this strategy must also work with backends
   * that do not expose a continuation cursor. Backend-reported distances are retained unless the
   * backend explicitly declares them approximate; this preserves RaBitQ implicit reranking.
   */
  [[nodiscard]] auto execute_post_filter(const DataType *query,
                                         const SearchInfo &search_info,
                                         const MetadataFilterExecutor<IDType> &filter_executor,
                                         const char *search_name,
                                         PlanStats &plan_stats) const
      -> std::vector<SearchCandidate<IDType, DistanceType>> {
    auto required = std::min<size_t>(search_info.topk_, filter_executor.indexed_count());
    if (required == 0) {
      return {};
    }

    auto max_candidates =
        std::min<size_t>(backend_->universe_size(), std::numeric_limits<uint32_t>::max());
    auto expected = static_cast<size_t>(std::ceil(
        (static_cast<double>(required) / plan_stats.pass_rate_) * kPostFilterOversamplingFactor));
    auto candidate_count = std::min(max_candidates, std::max<size_t>(search_info.topk_, expected));
    auto candidate_budget =
        std::min(max_candidates, std::max<size_t>(search_info.ef_, candidate_count));

    for (uint32_t attempt = 0; attempt < kPostFilterMaxAttempts; ++attempt) {
      plan_stats.effective_ef_ = static_cast<uint32_t>(candidate_budget);
      auto batch = backend_->search(VectorSearchRequest<DataType, IDType>{
          .query_ = query,
          .topk_ = static_cast<uint32_t>(candidate_count),
          .candidate_budget_ = static_cast<uint32_t>(candidate_budget),
      });
      plan_stats.post_filter_candidates_examined_ += batch.candidates_.size();

      std::vector<IDType> candidate_ids;
      candidate_ids.reserve(batch.candidates_.size());
      for (const auto &candidate : batch.candidates_) {
        validate_candidate_id(candidate.id_);
        candidate_ids.push_back(candidate.id_);
      }

      std::vector<uint8_t> matches;
      filter_executor.eval_offsets(candidate_ids, matches);
      BoundedCandidates result_pool(search_info.topk_);
      for (size_t i = 0; i < batch.candidates_.size(); ++i) {
        if (matches[i] == 0) {
          continue;
        }
        auto candidate = batch.candidates_[i];
        if (backend_->capabilities().returns_approx_distance_) {
          candidate.distance_ = backend_->exact_distance(query, candidate.id_);
        }
        result_pool.insert(std::move(candidate));
      }
      if (result_pool.size() >= required) {
        LOG_DEBUG("{}: post_filter candidates={}, retries={}, results={}",
                  search_name,
                  plan_stats.post_filter_candidates_examined_,
                  plan_stats.post_filter_retry_count_,
                  result_pool.size());
        return result_pool.take_ordered();
      }
      if (attempt + 1 >= kPostFilterMaxAttempts) {
        break;
      }

      auto next_candidate_count =
          std::min(max_candidates, std::max(candidate_count + 1, candidate_count * 2));
      auto next_candidate_budget =
          std::min(max_candidates, std::max(next_candidate_count, candidate_budget * 2));
      if (next_candidate_count == candidate_count && next_candidate_budget == candidate_budget) {
        break;
      }
      candidate_count = next_candidate_count;
      candidate_budget = next_candidate_budget;
      ++plan_stats.post_filter_retry_count_;
    }

    LOG_DEBUG("{}: post_filter underfilled after {} candidates and {} retries; use mask fallback",
              search_name,
              plan_stats.post_filter_candidates_examined_,
              plan_stats.post_filter_retry_count_);
    mark_fallback(plan_stats, Mode::kBitsetPrefilter, "post_filter_underfill");
    return execute_bitset_prefilter(query, search_info, filter_executor, search_name, plan_stats);
  }

  /** @brief Continue ANN candidate generation and evaluate scalar predicates per emitted batch. */
  [[nodiscard]] auto execute_iterative_filter(const DataType *query,
                                              const SearchInfo &search_info,
                                              const MetadataFilterExecutor<IDType> &filter_executor,
                                              const char *search_name,
                                              PlanStats &plan_stats) const
      -> std::vector<SearchCandidate<IDType, DistanceType>> {
    if (!backend_->capabilities().supports_candidate_cursor_) {
      mark_fallback(plan_stats, Mode::kIndexedExact, "candidate_cursor_unsupported");
      return execute_brute_force_filter(query, search_info.topk_, filter_executor, search_name);
    }

    auto cursor = backend_->open_cursor(VectorSearchRequest<DataType, IDType>{
        .query_ = query,
        .topk_ = search_info.topk_,
        .candidate_budget_ = search_info.ef_,
    });
    if (cursor == nullptr) {
      throw std::runtime_error("VectorSearchBackend returned a null candidate cursor");
    }

    BoundedCandidates result_pool(search_info.topk_);
    std::vector<IDType> candidate_ids;
    std::vector<uint8_t> matches;
    bool exhausted = false;
    while (result_pool.size() < search_info.topk_ && !exhausted) {
      auto batch_size = static_cast<size_t>(search_info.topk_ - result_pool.size());
      auto batch = cursor->next_batch(batch_size);
      exhausted = batch.exhausted_;
      if (batch.candidates_.empty()) {
        break;
      }

      candidate_ids.clear();
      candidate_ids.reserve(batch.candidates_.size());
      for (const auto &candidate : batch.candidates_) {
        candidate_ids.push_back(candidate.id_);
      }
      filter_executor.eval_offsets(candidate_ids, matches);
      for (size_t i = 0; i < batch.candidates_.size(); ++i) {
        if (matches[i] == 0) {
          continue;
        }
        auto candidate = batch.candidates_[i];
        validate_candidate_id(candidate.id_);
        if (backend_->capabilities().returns_approx_distance_) {
          candidate.distance_ = backend_->exact_distance(query, candidate.id_);
        }
        result_pool.insert(std::move(candidate));
      }
    }
    plan_stats.effective_ef_ = search_info.ef_;
    LOG_DEBUG("{}: iterative_filter results={}, requested={}",
              search_name,
              result_pool.size(),
              search_info.topk_);
    return result_pool.take_ordered();
  }

  /** @brief Materialize one scalar mask and push it into ANN result admission. */
  [[nodiscard]] auto execute_bitset_prefilter(const DataType *query,
                                              const SearchInfo &search_info,
                                              const MetadataFilterExecutor<IDType> &filter_executor,
                                              const char *search_name,
                                              PlanStats &plan_stats) const
      -> std::vector<SearchCandidate<IDType, DistanceType>> {
    auto bitset_result = filter_executor.build_blocked_bitset();
    plan_stats.matched_count_ = bitset_result.matched_count_;
    plan_stats.matched_count_known_ = true;
    update_pass_rate(plan_stats);
    if (bitset_result.matched_count_ == 0) {
      LOG_DEBUG("{}: bitset_prefilter matched zero rows", search_name);
      return {};
    }

    if (bitset_result.matched_count_ == backend_->universe_size()) {
      mark_fallback(plan_stats, Mode::kPlainSearch, "filter_matches_all");
      return execute_plain_search(query, search_info);
    }
    if (should_use_brute_force_search(search_info, bitset_result.matched_count_)) {
      mark_fallback(plan_stats, Mode::kIndexedExact, "brute_force_cost_threshold");
      return execute_brute_force_filter(query, search_info.topk_, filter_executor, search_name);
    }
    if (!backend_->capabilities().supports_accept_mask_) {
      mark_fallback(plan_stats, Mode::kIndexedExact, "accept_mask_unsupported");
      return execute_brute_force_filter(query, search_info.topk_, filter_executor, search_name);
    }

    auto adjusted = adjust_search_info(search_info, bitset_result.matched_count_, search_name);
    plan_stats.effective_ef_ = adjusted.ef_;
    auto blocked = std::make_shared<DynamicBitset>(std::move(bitset_result.blocked_));
    BlockedBitsetIdMask<IDType> accept_mask(std::move(blocked));
    auto batch = backend_->search(VectorSearchRequest<DataType, IDType>{
        .query_ = query,
        .topk_ = adjusted.topk_,
        .candidate_budget_ = adjusted.ef_,
        .accept_mask_ = &accept_mask,
    });
    auto candidates =
        normalize_candidates(query, std::move(batch.candidates_), search_info.topk_, &accept_mask);
    auto required = std::min<size_t>(search_info.topk_, bitset_result.matched_count_);
    if (candidates.size() < required) {
      mark_fallback(plan_stats, Mode::kIndexedExact, "ann_underfill");
      return execute_brute_force_filter(query, search_info.topk_, filter_executor, search_name);
    }
    return candidates;
  }

  /** @brief Exact-rerank, validate and bound one backend candidate list. */
  [[nodiscard]] auto normalize_candidates(
      const DataType *query,
      std::vector<SearchCandidate<IDType, DistanceType>> candidates,
      size_t topk,
      const IdMask<IDType> *accept_mask) const
      -> std::vector<SearchCandidate<IDType, DistanceType>> {
    BoundedCandidates result(topk);
    for (auto &candidate : candidates) {
      validate_candidate_id(candidate.id_);
      if (accept_mask != nullptr && !accept_mask->accepts(candidate.id_)) {
        continue;
      }
      if (backend_->capabilities().returns_approx_distance_) {
        candidate.distance_ = backend_->exact_distance(query, candidate.id_);
      }
      result.insert(std::move(candidate));
    }
    return result.take_ordered();
  }

  /** @brief Reject backend results outside its declared internal-ID universe. */
  void validate_candidate_id(IDType id) const {
    if (static_cast<size_t>(id) >= backend_->universe_size()) {
      throw std::runtime_error("VectorSearchBackend returned an out-of-universe ID");
    }
  }

  /** @brief Inflate the ANN budget inversely with known filter selectivity. */
  [[nodiscard]] auto adjust_search_info(const SearchInfo &search_info,
                                        size_t matched_count,
                                        const char *search_name) const -> SearchInfo {
    auto data_count = backend_->universe_size();
    if (matched_count == 0 || matched_count >= data_count) {
      return search_info;
    }
    auto expected_ef = (static_cast<double>(search_info.topk_) * static_cast<double>(data_count)) /
                       static_cast<double>(matched_count);
    expected_ef *= 1.5;
    auto bounded_ef = std::min<double>(data_count, expected_ef);
    SearchInfo adjusted = search_info;
    adjusted.ef_ = static_cast<uint32_t>(
        std::max<double>(search_info.ef_,
                         std::min<double>(bounded_ef, std::numeric_limits<uint32_t>::max())));
    if (adjusted.ef_ != search_info.ef_) {
      LOG_DEBUG("{}: inflate candidate budget from {} to {} for sparse mask pushdown",
                search_name,
                search_info.ef_,
                adjusted.ef_);
    }
    return adjusted;
  }

  /** @brief Estimate when exact distance evaluation is cheaper than ANN mask pushdown. */
  [[nodiscard]] auto should_use_brute_force_search(const SearchInfo &search_info,
                                                   size_t matched_count) const -> bool {
    if (matched_count == 0) {
      return false;
    }
    auto total_count = backend_->universe_size();
    auto topk = static_cast<size_t>(search_info.topk_);
    if (topk >= static_cast<size_t>(static_cast<double>(total_count) * kBFTopkThreshold)) {
      return true;
    }
    auto filtered_out = total_count - matched_count;
    if (filtered_out >=
        static_cast<size_t>(static_cast<double>(total_count) * kKnnBFFilterThreshold)) {
      return true;
    }
    return topk >= static_cast<size_t>(static_cast<double>(matched_count) * kBFTopkThreshold);
  }

  /** @brief Return whether known selectivity and backend behavior favor post-filter execution. */
  [[nodiscard]] auto should_use_post_filter(size_t matched_count) const -> bool {
    auto data_count = backend_->universe_size();
    if (!backend_->capabilities().supports_candidate_expansion_ || data_count == 0 ||
        matched_count >= data_count) {
      return false;
    }
    auto pass_rate =
        static_cast<double>(matched_count) / static_cast<double>(backend_->universe_size());
    return pass_rate >= kPostFilterPassRateThreshold;
  }

  /** @brief Record one runtime strategy transition. */
  static void mark_fallback(PlanStats &stats, Mode executed, const char *reason) {
    stats.executed_mode_ = executed;
    stats.fallback_ = true;
    stats.fallback_reason_ = reason;
  }

  /** @brief Recompute pass rate after an exact match count becomes available. */
  static void update_pass_rate(PlanStats &stats) {
    if (!stats.matched_count_known_) {
      return;
    }
    stats.pass_rate_ = stats.data_count_ == 0 ? 0.0
                                              : static_cast<double>(stats.matched_count_) /
                                                    static_cast<double>(stats.data_count_);
  }

  std::shared_ptr<Backend> backend_;  ///< Vector-only execution contract used by every strategy.
};

}  // namespace alaya
