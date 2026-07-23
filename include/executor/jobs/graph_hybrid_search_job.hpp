// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "executor/jobs/graph_search_job.hpp"
#include "executor/search_info.hpp"
#include "index/graph/graph.hpp"
#include "scalar/scalar_query_provider.hpp"
#include "search/hybrid_search_planner.hpp"
#include "search/legacy_graph_search_backend.hpp"
#include "space/rabitq_space.hpp"
#include "space/space_concepts.hpp"
#include "utils/log.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/metadata_filter_matcher.hpp"

#if defined(__linux__)
  #include "coro/task.hpp"
#endif

namespace alaya {

/**
 * @brief Compatibility facade that binds legacy graph/Space objects to hybrid query services.
 *
 * Strategy selection and execution live in HybridSearchPlanner. This facade retains the existing
 * Python and C++ API while owning only graph-backend adaptation, scalar-view acquisition, external
 * item-ID materialization and plan telemetry.
 */
template <typename DistanceSpaceType,
          typename BuildSpaceType = DistanceSpaceType,
          typename DataType = typename DistanceSpaceType::DataTypeAlias,
          typename DistanceType = typename DistanceSpaceType::DistanceTypeAlias,
          typename IDType = typename DistanceSpaceType::IDTypeAlias>
  requires Space<DistanceSpaceType> && Space<BuildSpaceType>
struct GraphHybridSearchJob {
  using VectorBackend = VectorSearchBackend<DataType, IDType, DistanceType>;
  using Planner = HybridSearchPlanner<DataType, IDType, DistanceType>;
  using Mode = typename Planner::Mode;
  using HybridPlanStats = typename Planner::PlanStats;
  using PlanStatsHook = std::function<void(const HybridPlanStats &)>;

  static constexpr float kHybridSearchKnnBFFilterThreshold =
      Planner::kKnnBFFilterThreshold;  ///< Compatibility alias for the exact-search threshold.
  static constexpr float kHybridSearchBFTopkThreshold =
      Planner::kBFTopkThreshold;  ///< Compatibility alias for the top-k cost threshold.

  std::shared_ptr<DistanceSpaceType> space_ = nullptr;  ///< Legacy query-hot vector/metadata owner.
  std::shared_ptr<BuildSpaceType> build_space_ = nullptr;     ///< Legacy exact-vector rerank owner.
  std::shared_ptr<Graph<DataType, IDType>> graph_ = nullptr;  ///< Legacy graph implementation.
  std::shared_ptr<const ScalarQueryProvider<IDType>> scalar_query_provider_ =
      nullptr;  ///< Optional scalar owner independent from Space.
  std::shared_ptr<VectorBackend> vector_backend_ = nullptr;  ///< Backend-neutral vector contract.

  /**
   * @brief Construct the compatibility facade and, by default, adapt the supplied graph objects.
   * @param space Legacy search Space used by the default backend and scalar compatibility path.
   * @param graph Legacy graph; required for non-RaBitQ default backends.
   * @param build_space Exact/raw Space used for reranking by the default backend.
   * @param scalar_query_provider Optional generation-stable scalar owner.
   * @param vector_backend Optional backend injection that bypasses GraphSearchJob construction.
   */
  explicit GraphHybridSearchJob(
      std::shared_ptr<DistanceSpaceType> space,
      std::shared_ptr<Graph<DataType, IDType>> graph = nullptr,
      std::shared_ptr<BuildSpaceType> build_space = nullptr,
      std::shared_ptr<const ScalarQueryProvider<IDType>> scalar_query_provider = nullptr,
      std::shared_ptr<VectorBackend> vector_backend = nullptr)
      : space_(std::move(space)),
        build_space_(std::move(build_space)),
        graph_(std::move(graph)),
        scalar_query_provider_(std::move(scalar_query_provider)),
        vector_backend_(std::move(vector_backend)) {
    if (space_ == nullptr) {
      throw std::invalid_argument("space is required for graph hybrid search compatibility");
    }
    if (vector_backend_ == nullptr) {
      validate_legacy_components();
      auto graph_job =
          std::make_shared<GraphSearchJob<DistanceSpaceType, BuildSpaceType>>(space_,
                                                                              graph_,
                                                                              nullptr,
                                                                              build_space_);
      vector_backend_ =
          std::make_shared<LegacyGraphSearchBackend<DistanceSpaceType, BuildSpaceType>>(
              std::move(graph_job));
    }
    planner_ = std::make_unique<Planner>(vector_backend_);
  }

  /**
   * @brief Install the observer invoked after each hybrid query.
   * @param hook Observer to install, or an empty function to disable observation.
   *
   * Configure the hook before starting concurrent searches. Replacing it concurrently with a query
   * is unsupported; the callback itself must tolerate concurrent invocations.
   */
  void set_plan_stats_hook(PlanStatsHook hook) { plan_stats_hook_ = std::move(hook); }

  /**
   * @brief Resolve internal result IDs to external item IDs at the scalar query generation.
   * @param ids Internal IDs produced by vector search.
   * @param count Number of valid entries in ids and res.
   * @param res Output slots receiving external item IDs.
   * @param scalar_view Optional provider-owned stable view; null selects legacy Space storage.
   */
  void materialize_item_ids(const IDType *ids,
                            uint32_t count,
                            std::string *res,
                            const ScalarQueryView<IDType> *scalar_view = nullptr) {
    std::vector<std::string> item_ids;
    auto result_ids = std::vector<IDType>(ids, ids + count);
    if (scalar_view != nullptr) {
      item_ids = scalar_view->batch_get_item_ids(result_ids);
    } else if constexpr (DistanceSpaceType::has_scalar_data) {
      item_ids = space_->get_scalar_storage()->batch_get_item_id_only(result_ids);
    } else {
      throw std::runtime_error("hybrid search requires a ScalarQueryProvider");
    }
    if (item_ids.size() != count) {
      throw std::runtime_error("scalar query provider returned an invalid item-ID batch size");
    }
    for (uint32_t i = 0; i < count; ++i) {
      res[i] = std::move(item_ids[i]);
    }
  }

  /** @brief Initialize every output slot before executing a hybrid-search strategy. */
  void initialize_results(IDType *ids, std::string *res, uint32_t topk) const {
    std::fill(ids, ids + topk, std::numeric_limits<IDType>::max());
    std::fill(res, res + topk, std::string{});
  }

  /** @brief Bind filter execution to a stable scalar view or legacy Space storage. */
  [[nodiscard]] auto make_filter_executor(const MetadataFilter &filter,
                                          const ScalarQueryView<IDType> *scalar_view =
                                              nullptr) const -> MetadataFilterExecutor<IDType> {
    if (scalar_view != nullptr) {
      return MetadataFilterExecutor<IDType>(filter,
                                            &scalar_view->scalar_index(),
                                            &scalar_view->record_store(),
                                            scalar_view->universe_size(),
                                            &scalar_view->live_mask(),
                                            scalar_view->live_count());
    }
    if constexpr (DistanceSpaceType::has_scalar_data) {
      return MetadataFilterExecutor<IDType>(filter,
                                            space_->get_scalar_storage(),
                                            vector_backend_->universe_size());
    }
    throw std::runtime_error("hybrid search requires a ScalarQueryProvider");
  }

  /** @brief Pin one scalar generation and validate it against the vector backend's ID universe. */
  [[nodiscard]] auto acquire_scalar_query_view() const -> std::unique_ptr<ScalarQueryView<IDType>> {
    if (scalar_query_provider_ == nullptr) {
      return nullptr;
    }
    auto view = scalar_query_provider_->acquire();
    if (view == nullptr) {
      throw std::runtime_error("ScalarQueryProvider returned a null query view");
    }
    if (view->universe_size() != vector_backend_->universe_size()) {
      throw std::runtime_error("scalar and vector internal-ID universes do not match");
    }
    return view;
  }

  /** @brief Reject zero top-k and candidate budgets narrower than top-k. */
  static void validate_search_info(const SearchInfo &search_info, const char *search_name) {
    if (search_info.topk_ == 0) {
      throw std::invalid_argument(std::string(search_name) + ": topk must be > 0");
    }
    if (search_info.ef_ < search_info.topk_) {
      throw std::invalid_argument(std::string(search_name) + ": ef must be >= topk");
    }
  }

  /** @brief Return the stable telemetry name for a planner strategy. */
  [[nodiscard]] static auto mode_name(Mode mode) -> const char * {
    return Planner::mode_name(mode);
  }

  /** @brief Delegate initial plan-stat construction to the backend-neutral planner. */
  [[nodiscard]] auto make_plan_stats(Mode mode,
                                     const SearchInfo &search_info,
                                     const MetadataFilterExecutor<IDType> &filter_executor) const
      -> HybridPlanStats {
    return planner_->make_plan_stats(mode, search_info, filter_executor);
  }

  /** @brief Delegate strategy selection to the backend-neutral planner. */
  [[nodiscard]] auto build_search_mode(const MetadataFilterExecutor<IDType> &filter_executor,
                                       const SearchInfo &search_info) const -> Mode {
    return planner_->build_search_mode(filter_executor, search_info);
  }

  /** @brief Execute a hybrid query through the backend-neutral planner. */
  void hybrid_search_solo(DataType *query,
                          IDType *ids,
                          const SearchInfo &search_info,
                          const MetadataFilter &filter,
                          std::string *res) {
    execute_search(query, ids, search_info, filter, res, "hybrid_search");
  }

  /** @brief Compatibility overload accepting top-k and ef separately. */
  void hybrid_search_solo(DataType *query,
                          IDType *ids,
                          uint32_t topk,
                          uint32_t ef,
                          const MetadataFilter &filter,
                          std::string *res) {
    hybrid_search_solo(query, ids, SearchInfo{.topk_ = topk, .ef_ = ef}, filter, res);
  }

  /** @brief Execute exact filtered top-k through the vector backend's distance service. */
  void hybrid_search_brute_force_solo(const DataType *query,
                                      IDType *ids,
                                      uint32_t topk,
                                      const MetadataFilter &filter,
                                      std::string *res) {
    if (topk == 0) {
      throw std::invalid_argument("hybrid_search_brute_force: topk must be > 0");
    }
    initialize_results(ids, res, topk);
    auto scalar_view = acquire_scalar_query_view();
    auto filter_executor = make_filter_executor(filter, scalar_view.get());
    auto candidates = planner_->execute_brute_force_filter(query,
                                                           topk,
                                                           filter_executor,
                                                           "hybrid_search_brute_force");
    auto result_count = copy_candidate_ids(candidates, ids);
    materialize_item_ids(ids, result_count, res, scalar_view.get());
    if (result_count < topk) {
      LOG_DEBUG("hybrid_search_brute_force: only found {} results, requested {}",
                result_count,
                topk);
    }
  }

  /** @brief Execute the RaBitQ compatibility entry point through the same planner. */
  void rabitq_hybrid_search_solo(const DataType *query,
                                 const SearchInfo &search_info,
                                 IDType *ids,
                                 const MetadataFilter &filter,
                                 std::string *res) {
    if constexpr (!is_rabitq_space_v<DistanceSpaceType>) {
      throw std::invalid_argument("Only support RaBitQSpace instance!");
    }
    execute_search(query, ids, search_info, filter, res, "rabitq_hybrid_search");
  }

  /** @brief Compatibility RaBitQ overload accepting top-k and ef separately. */
  void rabitq_hybrid_search_solo(const DataType *query,
                                 uint32_t topk,
                                 IDType *ids,
                                 uint32_t ef,
                                 const MetadataFilter &filter,
                                 std::string *res) {
    rabitq_hybrid_search_solo(query, SearchInfo{.topk_ = topk, .ef_ = ef}, ids, filter, res);
  }

#if defined(__linux__)
  /** @brief Coroutine wrapper for hybrid_search_solo. */
  auto hybrid_search(DataType *query,
                     IDType *ids,
                     SearchInfo search_info,
                     const MetadataFilter &filter,
                     std::string *res) -> coro::task<> {
    hybrid_search_solo(query, ids, search_info, filter, res);
    co_return;
  }

  /** @brief Coroutine compatibility overload accepting top-k and ef separately. */
  auto hybrid_search(DataType *query,
                     IDType *ids,
                     uint32_t topk,
                     uint32_t ef,
                     const MetadataFilter &filter,
                     std::string *res) -> coro::task<> {
    hybrid_search_solo(query, ids, topk, ef, filter, res);
    co_return;
  }

  /** @brief Coroutine wrapper for exact filtered search. */
  auto hybrid_search_brute_force(const DataType *query,
                                 IDType *ids,
                                 uint32_t topk,
                                 const MetadataFilter &filter,
                                 std::string *res) -> coro::task<> {
    hybrid_search_brute_force_solo(query, ids, topk, filter, res);
    co_return;
  }

  /** @brief Coroutine wrapper for the RaBitQ compatibility entry point. */
  auto rabitq_hybrid_search(const DataType *query,
                            SearchInfo search_info,
                            IDType *ids,
                            const MetadataFilter &filter,
                            std::string *res) -> coro::task<> {
    rabitq_hybrid_search_solo(query, search_info, ids, filter, res);
    co_return;
  }

  /** @brief Coroutine RaBitQ overload accepting top-k and ef separately. */
  auto rabitq_hybrid_search(const DataType *query,
                            uint32_t topk,
                            IDType *ids,
                            uint32_t ef,
                            const MetadataFilter &filter,
                            std::string *res) -> coro::task<> {
    rabitq_hybrid_search_solo(query, topk, ids, ef, filter, res);
    co_return;
  }
#endif

 private:
  /** @brief Validate objects required when constructing the legacy graph backend adapter. */
  void validate_legacy_components() const {
    if constexpr (!is_rabitq_space_v<DistanceSpaceType>) {
      if (graph_ == nullptr) {
        throw std::invalid_argument("graph is required for graph hybrid search");
      }
      if (build_space_ == nullptr) {
        throw std::invalid_argument("build_space is required for graph hybrid search");
      }
    }
  }

  /** @brief Execute one query and perform facade-only ID materialization and observation. */
  void execute_search(const DataType *query,
                      IDType *ids,
                      const SearchInfo &search_info,
                      const MetadataFilter &filter,
                      std::string *res,
                      const char *search_name) {
    validate_search_info(search_info, search_name);
    initialize_results(ids, res, search_info.topk_);
    auto scalar_view = acquire_scalar_query_view();
    auto filter_executor = make_filter_executor(filter, scalar_view.get());
    auto execution = planner_->execute(query, search_info, filter_executor, search_name);
    auto result_count = copy_candidate_ids(execution.candidates_, ids);
    emit_plan_stats(search_name, execution.stats_);
    materialize_item_ids(ids, result_count, res, scalar_view.get());
    if (result_count < search_info.topk_) {
      LOG_DEBUG("{}: only found {} results, requested {}",
                search_name,
                result_count,
                search_info.topk_);
    }
  }

  /** @brief Copy final ordered candidates into the legacy ID-only output API. */
  [[nodiscard]] static auto copy_candidate_ids(
      const std::vector<SearchCandidate<IDType, DistanceType>> &candidates,
      IDType *ids) -> uint32_t {
    for (size_t i = 0; i < candidates.size(); ++i) {
      ids[i] = candidates[i].id_;
    }
    return static_cast<uint32_t>(candidates.size());
  }

  /** @brief Emit plan telemetry and invoke the optional query observer. */
  void emit_plan_stats(const char *search_name, const HybridPlanStats &stats) const {
    LOG_DEBUG(
        "{}: plan_stats initial={}, executed={}, matched_count={}, matched_known={}, "
        "pass_rate={}, requested_ef={}, effective_ef={}, fanout={}, results={}, fallback={}, "
        "fallback_reason={}",
        search_name,
        mode_name(stats.initial_mode_),
        mode_name(stats.executed_mode_),
        stats.matched_count_,
        stats.matched_count_known_,
        stats.pass_rate_,
        stats.requested_ef_,
        stats.effective_ef_,
        stats.fanout_,
        stats.result_count_,
        stats.fallback_,
        stats.fallback_reason_);
    if (plan_stats_hook_) {
      plan_stats_hook_(stats);
    }
  }

  std::unique_ptr<Planner> planner_;  ///< Backend-neutral strategy selector and executor.
  PlanStatsHook plan_stats_hook_;     ///< Optional observer configured before concurrent queries.
};

}  // namespace alaya
