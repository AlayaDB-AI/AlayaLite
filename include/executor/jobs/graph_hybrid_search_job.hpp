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

#include "../../index/graph/graph.hpp"
#include "../../space/space_concepts.hpp"
#include "../../utils/query_utils.hpp"
#include "executor/search_info.hpp"
#include "graph_search_job.hpp"
#include "scalar/scalar_query_provider.hpp"
#include "space/rabitq_space.hpp"
#include "utils/log.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/metadata_filter_matcher.hpp"
#include "utils/rabitq_utils/search_utils/buffer.hpp"

#if defined(__linux__)
  #include "coro/task.hpp"
#endif

namespace alaya {

template <typename DistanceSpaceType,
          typename BuildSpaceType = DistanceSpaceType,
          typename DataType = typename DistanceSpaceType::DataTypeAlias,
          typename DistanceType = typename DistanceSpaceType::DistanceTypeAlias,
          typename IDType = typename DistanceSpaceType::IDTypeAlias>
  requires Space<DistanceSpaceType> && Space<BuildSpaceType>
struct GraphHybridSearchJob {
  // TODO(P2): Make these thresholds configurable via SearchInfo or constructor
  // parameter instead of hardcoding. Different workloads may benefit from
  // different cutoff points for switching between graph and brute-force search.
  static constexpr float kHybridSearchKnnBFFilterThreshold = 0.93f;
  static constexpr float kHybridSearchBFTopkThreshold = 0.5f;

  std::shared_ptr<DistanceSpaceType> space_ = nullptr;
  std::shared_ptr<BuildSpaceType> build_space_ = nullptr;
  std::shared_ptr<Graph<DataType, IDType>> graph_ = nullptr;
  std::shared_ptr<const ScalarQueryProvider<IDType>> scalar_query_provider_ =
      nullptr;  ///< Optional scalar owner independent from Space.

  // `kPlainSearch` is best when there is no scalar filter.
  // `kBitsetPrefilter` is the default hybrid path and works well when the filter can be pushed
  // down as a blocked bitset into ANN traversal.
  // `kIterativeFilter` keeps ANN generation and scalar evaluation separate and is useful when the
  // caller explicitly prefers iterator-style execution.
  // `kIndexedExact` skips bitset construction and computes exact top-k directly on indexed ids.
  enum class Mode : uint8_t { kPlainSearch, kBitsetPrefilter, kIterativeFilter, kIndexedExact };

  struct HybridPlanStats {
    Mode initial_mode_ = Mode::kPlainSearch;   ///< Mode selected before runtime fallback.
    Mode executed_mode_ = Mode::kPlainSearch;  ///< Mode that produced the final results.
    size_t matched_count_ = 0;                 ///< Rows accepted by the scalar filter, when known.
    size_t data_count_ = 0;                    ///< Total vector slots visible to the search job.
    double pass_rate_ = 0.0;                   ///< matched_count / data_count, when known.
    uint32_t requested_ef_ = 0;                ///< Caller-provided graph-search breadth.
    uint32_t effective_ef_ = 0;                ///< Graph-search breadth after legacy adjustment.
    uint32_t fanout_ = 0;                      ///< Partitions searched; zero in legacy execution.
    uint32_t result_count_ = 0;                ///< Valid results produced by the query.
    bool matched_count_known_ = false;         ///< Whether matched_count and pass_rate are valid.
    bool fallback_ = false;                    ///< Whether execution switched modes at runtime.
    std::string fallback_reason_;              ///< Stable reason code; empty without fallback.
  };

  using PlanStatsHook = std::function<void(const HybridPlanStats &)>;

  explicit GraphHybridSearchJob(
      std::shared_ptr<DistanceSpaceType> space,
      std::shared_ptr<Graph<DataType, IDType>> graph = nullptr,
      std::shared_ptr<BuildSpaceType> build_space = nullptr,
      std::shared_ptr<const ScalarQueryProvider<IDType>> scalar_query_provider = nullptr)
      : space_(std::move(space)),
        build_space_(std::move(build_space)),
        graph_(std::move(graph)),
        scalar_query_provider_(std::move(scalar_query_provider)) {
    if constexpr (!is_rabitq_space_v<DistanceSpaceType>) {
      if (graph_ == nullptr) {
        throw std::invalid_argument("graph is required for graph hybrid search");
      }
      if (build_space_ == nullptr) {
        throw std::invalid_argument("build_space is required for graph hybrid search");
      }
    }
  }

  /**
   * @brief Installs the observer invoked after each legacy hybrid query.
   * @param hook Observer to install, or an empty function to disable observation.
   *
   * Configure the hook before starting concurrent searches; replacing it concurrently with a
   * query is unsupported. The callback itself must tolerate concurrent invocations from searches.
   */
  void set_plan_stats_hook(PlanStatsHook hook) { plan_stats_hook_ = std::move(hook); }

  /**
   * @brief Resolve internal result IDs to external item IDs at the scalar query generation.
   * @param ids Internal IDs produced by vector search.
   * @param count Number of valid entries in ids and res.
   * @param res Output slots receiving external item IDs.
   * @param scalar_view Optional provider-owned stable view; null selects the legacy Space adapter.
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

  /**
   * @brief Bind filter execution to the supplied stable scalar view or the legacy Space storage.
   */
  auto make_filter_executor(const MetadataFilter &filter,
                            const ScalarQueryView<IDType> *scalar_view = nullptr) const
      -> MetadataFilterExecutor<IDType> {
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
                                            space_->get_data_num());
    }
    throw std::runtime_error("hybrid search requires a ScalarQueryProvider");
  }

  /** @brief Pin one scalar generation and validate its ID universe against vector storage. */
  [[nodiscard]] auto acquire_scalar_query_view() const -> std::unique_ptr<ScalarQueryView<IDType>> {
    if (scalar_query_provider_ == nullptr) {
      return nullptr;
    }
    auto view = scalar_query_provider_->acquire();
    if (view == nullptr) {
      throw std::runtime_error("ScalarQueryProvider returned a null query view");
    }
    if (view->universe_size() != static_cast<size_t>(space_->get_data_num())) {
      throw std::runtime_error("scalar and vector internal-ID universes do not match");
    }
    return view;
  }

  static void validate_search_info(const SearchInfo &search_info, const char *search_name) {
    if (search_info.topk_ == 0) {
      throw std::invalid_argument(std::string(search_name) + ": topk must be > 0");
    }
    if (search_info.ef_ < search_info.topk_) {
      throw std::invalid_argument(std::string(search_name) + ": ef must be >= topk");
    }
  }

  [[nodiscard]] static auto mode_name(Mode mode) -> const char * {
    switch (mode) {
      case Mode::kPlainSearch:
        return "plain_search";
      case Mode::kBitsetPrefilter:
        return "bitset_prefilter";
      case Mode::kIterativeFilter:
        return "iterative_filter";
      case Mode::kIndexedExact:
        return "indexed_exact";
    }
    return "unknown";
  }

  /**
   * @brief Creates the initial observation without changing the selected legacy search strategy.
   * @param mode Mode selected by the existing strategy logic.
   * @param search_info Caller-provided search parameters.
   * @param filter_executor Executor that may already know the exact indexed match count.
   * @return Initial statistics; match-derived fields remain invalid when counting would require a
   * full scalar scan.
   */
  [[nodiscard]] auto make_plan_stats(Mode mode,
                                     const SearchInfo &search_info,
                                     const MetadataFilterExecutor<IDType> &filter_executor) const
      -> HybridPlanStats {
    HybridPlanStats stats;
    stats.initial_mode_ = mode;
    stats.executed_mode_ = mode;
    stats.data_count_ = static_cast<size_t>(space_->get_data_num());
    stats.requested_ef_ = search_info.ef_;
    stats.effective_ef_ = search_info.ef_;

    if (filter_executor.is_trivially_true()) {
      stats.matched_count_ = stats.data_count_;
      stats.matched_count_known_ = true;
    } else if (filter_executor.has_index_fast_path()) {
      stats.matched_count_ = filter_executor.indexed_count();
      stats.matched_count_known_ = true;
    }
    update_pass_rate(stats);
    return stats;
  }

  [[nodiscard]] auto build_search_mode(const MetadataFilterExecutor<IDType> &filter_executor,
                                       const SearchInfo &search_info) const -> Mode {
    if (filter_executor.is_trivially_true()) {
      return Mode::kPlainSearch;
    }

    switch (search_info.filter_exec_hint_) {
      case FilterExecHint::kAuto:
        if (filter_executor.has_index_fast_path() &&
            should_use_brute_force_search(search_info, filter_executor.indexed_count())) {
          return Mode::kIndexedExact;
        }
        return Mode::kBitsetPrefilter;
      case FilterExecHint::kIterativeFilter:
        return Mode::kIterativeFilter;
      case FilterExecHint::kDisableIterative:
        return Mode::kBitsetPrefilter;
    }
    return Mode::kBitsetPrefilter;
  }

  static auto materialize_result_ids(const SearchBuffer<DistanceType> &pool,
                                     IDType *ids,
                                     uint32_t topk) -> uint32_t {
    auto result_count = static_cast<uint32_t>(std::min<size_t>(pool.size(), topk));
    pool.copy_results_to(reinterpret_cast<uint32_t *>(ids), result_count);
    return result_count;
  }

  static auto count_materialized_results(const IDType *ids, uint32_t topk) -> uint32_t {
    uint32_t count = 0;
    while (count < topk && ids[count] != std::numeric_limits<IDType>::max()) {
      ++count;
    }
    return count;
  }

  [[nodiscard]] auto adjust_ef_in_search_info(const SearchInfo &search_info,
                                              size_t matched_count,
                                              const char *search_name) const -> SearchInfo {
    if (matched_count == 0 || matched_count >= space_->get_data_num()) {
      return search_info;
    }

    // selectivity = matched_count / total_count,
    // expected_ef = topk / selectivity
    auto expected_ef = static_cast<size_t>(
        (static_cast<double>(search_info.topk_) * static_cast<double>(space_->get_data_num())) /
        static_cast<double>(matched_count));
    // TODO(P2): The 1.5x ef inflation factor is fixed. Consider making it
    // adaptive based on historical query performance or filter selectivity.
    expected_ef += expected_ef / 2;  // 1.5x on default

    SearchInfo adjusted = search_info;
    adjusted.ef_ = static_cast<uint32_t>(
        std::min<size_t>(space_->get_data_num(), std::max<size_t>(search_info.ef_, expected_ef)));
    if (adjusted.ef_ != search_info.ef_) {
      LOG_DEBUG("{}: inflate ef from {} to {} for sparse prefilter pushdown",
                search_name,
                search_info.ef_,
                adjusted.ef_);
    }
    return adjusted;
  }

  [[nodiscard]] auto should_use_brute_force_search(const SearchInfo &search_info,
                                                   size_t matched_count) const -> bool {
    if (matched_count == 0) {
      return false;
    }

    auto total_count = static_cast<size_t>(space_->get_data_num());
    auto topk = static_cast<size_t>(search_info.topk_);
    if (topk >=
        static_cast<size_t>(static_cast<double>(total_count) * kHybridSearchBFTopkThreshold)) {
      return true;
    }

    auto filtered_out = total_count - matched_count;
    if (filtered_out >=
        static_cast<size_t>(static_cast<double>(total_count) * kHybridSearchKnnBFFilterThreshold)) {
      return true;
    }

    return topk >=
           static_cast<size_t>(static_cast<double>(matched_count) * kHybridSearchBFTopkThreshold);
  }

  auto execute_brute_force_filter(const DataType *query,
                                  IDType *ids,
                                  uint32_t topk,
                                  const MetadataFilterExecutor<IDType> &filter_executor,
                                  const char *search_name) -> uint32_t {
    SearchBuffer<DistanceType> result_pool(topk);
    std::vector<IDType> batch_ids;
    std::vector<uint8_t> matches;
    constexpr size_t kBatchSize = 1024;
    batch_ids.reserve(kBatchSize);
    auto run_indexed_candidates = [&](const auto &exact_distance) {
      filter_executor.visit_index_fast_path_ids([&](IDType id) {
        if (filter_executor.match(id)) {
          result_pool.insert(id, exact_distance(id));
        }
      });
    };
    auto run_full_scan = [&](const auto &exact_distance, size_t data_num) {
      for (size_t begin = 0; begin < data_num; begin += kBatchSize) {
        batch_ids.clear();
        auto end = std::min<size_t>(data_num, begin + kBatchSize);
        for (size_t id = begin; id < end; ++id) {
          batch_ids.push_back(static_cast<IDType>(id));
        }
        filter_executor.eval_offsets(batch_ids, matches);
        for (size_t i = 0; i < batch_ids.size(); ++i) {
          if (matches[i] != 0) {
            result_pool.insert(batch_ids[i], exact_distance(batch_ids[i]));
          }
        }
      }
    };
    auto use_indexed_candidates = filter_executor.has_index_fast_path();

    if constexpr (is_rabitq_space_v<DistanceSpaceType>) {
      auto dist_func = space_->get_dist_func();
      auto dim = space_->get_dim();
      auto exact_distance = [&](IDType id) -> DistanceType {
        return dist_func(query, space_->get_data_by_id(id), dim);
      };
      if (use_indexed_candidates) {
        run_indexed_candidates(exact_distance);
      } else {
        run_full_scan(exact_distance, space_->get_data_num());
      }
    } else if constexpr (std::is_same_v<DistanceSpaceType, BuildSpaceType>) {
      auto exact_qc = space_->get_query_computer(query);
      auto exact_distance = [&](IDType id) -> DistanceType {
        return exact_qc(id);
      };
      if (use_indexed_candidates) {
        run_indexed_candidates(exact_distance);
      } else {
        run_full_scan(exact_distance, space_->get_data_num());
      }
    } else {
      auto exact_qc = build_space_->get_query_computer(query);
      auto exact_distance = [&](IDType id) -> DistanceType {
        return exact_qc(id);
      };
      if (use_indexed_candidates) {
        run_indexed_candidates(exact_distance);
      } else {
        run_full_scan(exact_distance, space_->get_data_num());
      }
    }

    auto res_size = materialize_result_ids(result_pool, ids, topk);
    if (use_indexed_candidates) {
      LOG_DEBUG("{}: brute_force_filter indexed_candidates={}, results={}, requested={}",
                search_name,
                filter_executor.indexed_count(),
                res_size,
                topk);
    } else {
      LOG_DEBUG("{}: brute_force_filter results={}, requested={}", search_name, res_size, topk);
    }
    return res_size;
  }

  auto execute_iterative_filter(const DataType *query,
                                IDType *ids,
                                const SearchInfo &search_info,
                                const MetadataFilterExecutor<IDType> &filter_executor,
                                const char *search_name,
                                HybridPlanStats *plan_stats = nullptr) -> uint32_t {
    GraphSearchJob<DistanceSpaceType, BuildSpaceType> base_job(space_,
                                                               graph_,
                                                               nullptr,
                                                               build_space_);
    auto iterator = base_job.make_vector_iterator(query, search_info);

    SearchBuffer<DistanceType> result_pool(search_info.topk_);
    std::vector<IDType> candidate_ids;
    std::vector<DistanceType> candidate_distances;
    std::vector<uint8_t> matches;

    while (result_pool.size() < search_info.topk_ && iterator->has_next()) {
      auto batch_size = static_cast<size_t>(search_info.topk_ - result_pool.size());
      iterator->next_batch(batch_size, candidate_ids, candidate_distances);
      if (candidate_ids.empty()) {
        break;
      }

      filter_executor.eval_offsets(candidate_ids, matches);
      for (size_t i = 0; i < candidate_ids.size(); ++i) {
        if (matches[i] == 0) {
          continue;
        }
        result_pool.insert(candidate_ids[i], candidate_distances[i]);
        if (result_pool.size() == search_info.topk_) {
          break;
        }
      }
    }

    auto res_size = materialize_result_ids(result_pool, ids, search_info.topk_);
    if (plan_stats != nullptr) {
      plan_stats->effective_ef_ = search_info.ef_;
    }
    LOG_DEBUG("{}: iterative_filter results={}, requested={}",
              search_name,
              res_size,
              search_info.topk_);
    return res_size;
  }

  auto execute_bitset_prefilter(const DataType *query,
                                IDType *ids,
                                const SearchInfo &search_info,
                                const MetadataFilterExecutor<IDType> &filter_executor,
                                const char *search_name,
                                HybridPlanStats *plan_stats = nullptr) -> uint32_t {
    auto bitset_result = filter_executor.build_blocked_bitset();
    if (plan_stats != nullptr) {
      plan_stats->matched_count_ = bitset_result.matched_count_;
      plan_stats->matched_count_known_ = true;
      update_pass_rate(*plan_stats);
    }
    if (bitset_result.matched_count_ == 0) {
      LOG_DEBUG("{}: bitset_prefilter matched zero rows", search_name);
      return 0;
    }

    GraphSearchJob<DistanceSpaceType, BuildSpaceType> base_job(space_,
                                                               graph_,
                                                               nullptr,
                                                               build_space_);
    if (bitset_result.matched_count_ == space_->get_data_num()) {
      if (plan_stats != nullptr) {
        plan_stats->executed_mode_ = Mode::kPlainSearch;
        plan_stats->fallback_ = true;
        plan_stats->fallback_reason_ = "filter_matches_all";
      }
      LOG_DEBUG("{}: bitset_prefilter matched all rows, fallback to plain search", search_name);
      if constexpr (is_rabitq_space_v<DistanceSpaceType>) {
        base_job.rabitq_search_solo(query, search_info.topk_, ids, search_info.ef_);
      } else {
        base_job.search_solo(const_cast<DataType *>(query),
                             ids,
                             search_info.topk_,
                             search_info.ef_);
      }
      return std::min<uint32_t>(search_info.topk_, space_->get_data_num());
    }

    LOG_DEBUG("{}: bitset_prefilter matched_rows={}, topk={}, ef={}",
              search_name,
              bitset_result.matched_count_,
              search_info.topk_,
              search_info.ef_);

    if (should_use_brute_force_search(search_info, bitset_result.matched_count_)) {
      if (plan_stats != nullptr) {
        plan_stats->executed_mode_ = Mode::kIndexedExact;
        plan_stats->fallback_ = true;
        plan_stats->fallback_reason_ = "brute_force_cost_threshold";
      }
      LOG_DEBUG("{}: bitset_prefilter switching to brute force, matched_rows={}, topk={}, ef={}",
                search_name,
                bitset_result.matched_count_,
                search_info.topk_,
                search_info.ef_);
      return execute_brute_force_filter(query,
                                        ids,
                                        search_info.topk_,
                                        filter_executor,
                                        search_name);
    }

    auto adjusted_search_info =
        adjust_ef_in_search_info(search_info, bitset_result.matched_count_, search_name);
    if (plan_stats != nullptr) {
      plan_stats->effective_ef_ = adjusted_search_info.ef_;
    }
    if constexpr (is_rabitq_space_v<DistanceSpaceType>) {
      base_job.rabitq_search_solo(query,
                                  adjusted_search_info.topk_,
                                  ids,
                                  adjusted_search_info,
                                  &bitset_result.blocked_);
    } else {
      base_job.search_solo(const_cast<DataType *>(query),
                           ids,
                           adjusted_search_info,
                           &bitset_result.blocked_);
    }

    auto res_size = count_materialized_results(ids, search_info.topk_);
    auto required_results =
        static_cast<uint32_t>(std::min<size_t>(search_info.topk_, bitset_result.matched_count_));
    if (res_size < required_results) {
      if (plan_stats != nullptr) {
        plan_stats->executed_mode_ = Mode::kIndexedExact;
        plan_stats->fallback_ = true;
        plan_stats->fallback_reason_ = "ann_underfill";
      }
      LOG_DEBUG("{}: bitset_prefilter underfilled results={}, expected={}, fallback to brute force",
                search_name,
                res_size,
                required_results);
      return execute_brute_force_filter(query,
                                        ids,
                                        search_info.topk_,
                                        filter_executor,
                                        search_name);
    }
    return res_size;
  }

  void hybrid_search_solo(DataType *query,
                          IDType *ids,
                          const SearchInfo &search_info,
                          const MetadataFilter &filter,
                          std::string *res) {
    validate_search_info(search_info, "hybrid_search");
    initialize_results(ids, res, search_info.topk_);

    auto scalar_view = acquire_scalar_query_view();
    auto filter_executor = make_filter_executor(filter, scalar_view.get());
    auto mode = build_search_mode(filter_executor, search_info);
    auto plan_stats = make_plan_stats(mode, search_info, filter_executor);
    LOG_DEBUG("hybrid_search: plan={}, topk={}, ef={}, hint={}",
              mode_name(mode),
              search_info.topk_,
              search_info.ef_,
              static_cast<int>(search_info.filter_exec_hint_));

    uint32_t res_size = 0;
    if (mode == Mode::kPlainSearch) {
      GraphSearchJob<DistanceSpaceType, BuildSpaceType> base_job(space_,
                                                                 graph_,
                                                                 nullptr,
                                                                 build_space_);
      base_job.search_solo(query, ids, search_info.topk_, search_info.ef_);
      res_size = std::min<uint32_t>(search_info.topk_, space_->get_data_num());
    } else if (mode == Mode::kIndexedExact) {
      res_size = execute_brute_force_filter(query,
                                            ids,
                                            search_info.topk_,
                                            filter_executor,
                                            "hybrid_search");
    } else if (mode == Mode::kBitsetPrefilter) {
      res_size = execute_bitset_prefilter(query,
                                          ids,
                                          search_info,
                                          filter_executor,
                                          "hybrid_search",
                                          &plan_stats);
    } else {
      res_size = execute_iterative_filter(query,
                                          ids,
                                          search_info,
                                          filter_executor,
                                          "hybrid_search",
                                          &plan_stats);
    }

    plan_stats.result_count_ = res_size;
    emit_plan_stats("hybrid_search", plan_stats);
    materialize_item_ids(ids, res_size, res, scalar_view.get());
    if (res_size < search_info.topk_) {
      LOG_DEBUG("hybrid_search: only found {} results, requested {}", res_size, search_info.topk_);
    }
  }

  void hybrid_search_solo(DataType *query,
                          IDType *ids,
                          uint32_t topk,
                          uint32_t ef,
                          const MetadataFilter &filter,
                          std::string *res) {
    hybrid_search_solo(query, ids, SearchInfo{.topk_ = topk, .ef_ = ef}, filter, res);
  }

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
    auto res_size =
        execute_brute_force_filter(query, ids, topk, filter_executor, "hybrid_search_brute_force");
    materialize_item_ids(ids, res_size, res, scalar_view.get());
    if (res_size < topk) {
      LOG_DEBUG("hybrid_search_brute_force: only found {} results, requested {}", res_size, topk);
    }
  }

  void rabitq_hybrid_search_solo(const DataType *query,
                                 const SearchInfo &search_info,
                                 IDType *ids,
                                 const MetadataFilter &filter,
                                 std::string *res) {
    if constexpr (!is_rabitq_space_v<DistanceSpaceType>) {
      throw std::invalid_argument("Only support RaBitQSpace instance!");
    }

    validate_search_info(search_info, "rabitq_hybrid_search");
    initialize_results(ids, res, search_info.topk_);

    auto scalar_view = acquire_scalar_query_view();
    auto filter_executor = make_filter_executor(filter, scalar_view.get());
    auto mode = build_search_mode(filter_executor, search_info);
    auto plan_stats = make_plan_stats(mode, search_info, filter_executor);
    LOG_DEBUG("rabitq_hybrid_search: plan={}, topk={}, ef={}, hint={}",
              mode_name(mode),
              search_info.topk_,
              search_info.ef_,
              static_cast<int>(search_info.filter_exec_hint_));

    uint32_t res_size = 0;
    if (mode == Mode::kPlainSearch) {
      GraphSearchJob<DistanceSpaceType, BuildSpaceType> base_job(space_,
                                                                 graph_,
                                                                 nullptr,
                                                                 build_space_);
      base_job.rabitq_search_solo(query, search_info.topk_, ids, search_info.ef_);
      res_size = std::min<uint32_t>(search_info.topk_, space_->get_data_num());
    } else if (mode == Mode::kIndexedExact) {
      res_size = execute_brute_force_filter(query,
                                            ids,
                                            search_info.topk_,
                                            filter_executor,
                                            "rabitq_hybrid_search");
    } else if (mode == Mode::kBitsetPrefilter) {
      res_size = execute_bitset_prefilter(query,
                                          ids,
                                          search_info,
                                          filter_executor,
                                          "rabitq_hybrid_search",
                                          &plan_stats);
    } else {
      res_size = execute_iterative_filter(query,
                                          ids,
                                          search_info,
                                          filter_executor,
                                          "rabitq_hybrid_search",
                                          &plan_stats);
    }

    plan_stats.result_count_ = res_size;
    emit_plan_stats("rabitq_hybrid_search", plan_stats);
    materialize_item_ids(ids, res_size, res, scalar_view.get());
    if (res_size < search_info.topk_) {
      LOG_DEBUG("rabitq_hybrid_search: only found {} results, requested {}",
                res_size,
                search_info.topk_);
    }
  }

  void rabitq_hybrid_search_solo(const DataType *query,
                                 uint32_t topk,
                                 IDType *ids,
                                 uint32_t ef,
                                 const MetadataFilter &filter,
                                 std::string *res) {
    rabitq_hybrid_search_solo(query, SearchInfo{.topk_ = topk, .ef_ = ef}, ids, filter, res);
  }

#if defined(__linux__)
  auto hybrid_search(DataType *query,
                     IDType *ids,
                     SearchInfo search_info,
                     const MetadataFilter &filter,
                     std::string *res) -> coro::task<> {
    hybrid_search_solo(query, ids, search_info, filter, res);
    co_return;
  }

  auto hybrid_search(DataType *query,
                     IDType *ids,
                     uint32_t topk,
                     uint32_t ef,
                     const MetadataFilter &filter,
                     std::string *res) -> coro::task<> {
    hybrid_search_solo(query, ids, topk, ef, filter, res);
    co_return;
  }

  auto hybrid_search_brute_force(const DataType *query,
                                 IDType *ids,
                                 uint32_t topk,
                                 const MetadataFilter &filter,
                                 std::string *res) -> coro::task<> {
    hybrid_search_brute_force_solo(query, ids, topk, filter, res);
    co_return;
  }

  auto rabitq_hybrid_search(const DataType *query,
                            SearchInfo search_info,
                            IDType *ids,
                            const MetadataFilter &filter,
                            std::string *res) -> coro::task<> {
    rabitq_hybrid_search_solo(query, search_info, ids, filter, res);
    co_return;
  }

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
  /** Recomputes pass_rate_ when the executor supplied an exact match count. */
  static void update_pass_rate(HybridPlanStats &stats) {
    if (!stats.matched_count_known_) {
      return;
    }
    stats.pass_rate_ = stats.data_count_ == 0 ? 0.0
                                              : static_cast<double>(stats.matched_count_) /
                                                    static_cast<double>(stats.data_count_);
  }

  /** Emits debug telemetry and invokes the optional observer after query execution. */
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

  PlanStatsHook plan_stats_hook_;
};

}  // namespace alaya
