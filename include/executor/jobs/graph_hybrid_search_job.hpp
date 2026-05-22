// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../index/graph/graph.hpp"
#include "../../space/space_concepts.hpp"
#include "../../utils/query_utils.hpp"
#include "executor/search_info.hpp"
#include "graph_search_job.hpp"
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

  // `kPlainSearch` is best when there is no scalar filter.
  // `kBitsetPrefilter` is the default hybrid path and works well when the filter can be pushed
  // down as a blocked bitset into ANN traversal.
  // `kIterativeFilter` keeps ANN generation and scalar evaluation separate and is useful when the
  // caller explicitly prefers iterator-style execution.
  // `kIndexedExact` skips bitset construction and computes exact top-k directly on indexed ids.
  enum class Mode : uint8_t { kPlainSearch, kBitsetPrefilter, kIterativeFilter, kIndexedExact };

  using FilterExecutor = MetadataFilterExecutor<IDType>;
  using BlockedBitsetResult = typename FilterExecutor::BlockedBitsetResult;

  struct PreparedHybridSearchPlan {
    FilterExecutor filter_executor_;
    Mode mode_ = Mode::kBitsetPrefilter;
    std::optional<BlockedBitsetResult> prefilter_result_;

    PreparedHybridSearchPlan(FilterExecutor filter_executor, Mode mode)
        : filter_executor_(std::move(filter_executor)), mode_(mode) {}
  };

  explicit GraphHybridSearchJob(std::shared_ptr<DistanceSpaceType> space,
                                std::shared_ptr<Graph<DataType, IDType>> graph = nullptr,
                                std::shared_ptr<BuildSpaceType> build_space = nullptr)
      : space_(std::move(space)), build_space_(std::move(build_space)), graph_(std::move(graph)) {
    if constexpr (!is_rabitq_space_v<DistanceSpaceType>) {
      if (graph_ == nullptr) {
        throw std::invalid_argument("graph is required for graph hybrid search");
      }
      if (build_space_ == nullptr) {
        throw std::invalid_argument("build_space is required for graph hybrid search");
      }
    }
  }

  // clang-format off
  void materialize_item_ids(const IDType *ids,
                            uint32_t count,
                            std::string *res) requires(DistanceSpaceType::has_scalar_data) {
    auto *storage = space_->get_scalar_storage();
    auto item_ids = storage->batch_get_item_id_only(std::vector<IDType>(ids, ids + count));
    for (uint32_t i = 0; i < count; ++i) {
      res[i] = std::move(item_ids[i]);
    }
  }

  void initialize_results(IDType *ids,
                          std::string *res,
                          uint32_t topk) const requires(DistanceSpaceType::has_scalar_data) {
    std::fill(ids, ids + topk, std::numeric_limits<IDType>::max());
    std::fill(res, res + topk, std::string{});
  }

  auto make_filter_executor(const MetadataFilter &filter) const
      -> FilterExecutor requires(DistanceSpaceType::has_scalar_data) {
    return FilterExecutor(filter, space_->get_scalar_storage(), space_->get_data_num());
  }

  auto make_filter_executor(const MetadataFilter &filter,
                            typename FilterExecutor::IndexBuildMode index_build_mode) const
      -> FilterExecutor requires(DistanceSpaceType::has_scalar_data) {
    return FilterExecutor(filter,
                          space_->get_scalar_storage(),
                          space_->get_data_num(),
                          index_build_mode);
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

  [[nodiscard]] auto build_search_mode(const MetadataFilterExecutor<IDType> &filter_executor,
                                       const SearchInfo &search_info) const
      -> Mode requires(DistanceSpaceType::has_scalar_data) {
    return build_search_mode(filter_executor.is_trivially_true(),
                             filter_executor.has_index_fast_path(),
                             filter_executor.indexed_count(),
                             search_info);
  }

  [[nodiscard]] auto build_search_mode(bool is_trivially_true,
                                       bool has_index_fast_path,
                                       size_t indexed_count,
                                       const SearchInfo &search_info) const
      -> Mode requires(DistanceSpaceType::has_scalar_data) {
    if (is_trivially_true) {
      return Mode::kPlainSearch;
    }

    switch (search_info.filter_exec_hint_) {
      case FilterExecHint::kAuto:
        if (has_index_fast_path && should_use_brute_force_search(search_info, indexed_count)) {
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

  [[nodiscard]] static auto estimate_prefilter_ef(const SearchInfo &search_info,
                                                  size_t matched_count,
                                                  size_t total_count) -> size_t {
    if (matched_count == 0 || matched_count >= total_count) {
      return search_info.ef_;
    }

    auto expected_ef = static_cast<size_t>(
        (static_cast<double>(search_info.topk_) * static_cast<double>(total_count)) /
        static_cast<double>(matched_count));
    expected_ef += expected_ef / 2;  // 1.5x on default
    return std::min<size_t>(total_count, std::max<size_t>(search_info.ef_, expected_ef));
  }

  [[nodiscard]] auto adjust_ef_in_search_info(const SearchInfo &search_info,
                                              size_t matched_count,
                                              const char *search_name) const -> SearchInfo {
    if (matched_count == 0 || matched_count >= space_->get_data_num()) {
      return search_info;
    }

    SearchInfo adjusted = search_info;
    adjusted.ef_ = static_cast<uint32_t>(
        estimate_prefilter_ef(search_info, matched_count, space_->get_data_num()));
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
    if (matched_count >= total_count) {
      return false;
    }

    auto topk = static_cast<size_t>(search_info.topk_);
    auto ef = static_cast<size_t>(search_info.ef_);
    if (matched_count <= ef) {
      return true;
    }

    if (estimate_prefilter_ef(search_info, matched_count, total_count) >= matched_count) {
      return true;
    }

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

  auto execute_brute_force_bitset(const DataType *query,
                                  IDType *ids,
                                  uint32_t topk,
                                  const BlockedBitsetResult &bitset_result,
                                  const char *search_name)
      -> uint32_t requires(DistanceSpaceType::has_scalar_data) {
    SearchBuffer<DistanceType> result_pool(topk);
    auto run_candidates = [&](const auto &exact_distance) {
      for (size_t raw_id = 0; raw_id < bitset_result.blocked_.size(); ++raw_id) {
        if (!bitset_result.blocked_.get(raw_id)) {
          auto id = static_cast<IDType>(raw_id);
          result_pool.insert(id, exact_distance(id));
        }
      }
    };

    if constexpr (is_rabitq_space_v<DistanceSpaceType>) {
      auto dist_func = space_->get_dist_func();
      auto dim = space_->get_dim();
      auto exact_distance = [&](IDType id) -> DistanceType {
        return dist_func(query, space_->get_data_by_id(id), dim);
      };
      run_candidates(exact_distance);
    } else if constexpr (std::is_same_v<DistanceSpaceType, BuildSpaceType>) {
      auto exact_qc = space_->get_query_computer(query);
      auto exact_distance = [&](IDType id) -> DistanceType {
        return exact_qc(id);
      };
      run_candidates(exact_distance);
    } else {
      auto exact_qc = build_space_->get_query_computer(query);
      auto exact_distance = [&](IDType id) -> DistanceType {
        return exact_qc(id);
      };
      run_candidates(exact_distance);
    }

    auto res_size = materialize_result_ids(result_pool, ids, topk);
    LOG_DEBUG("{}: brute_force_bitset matched_rows={}, results={}, requested={}",
              search_name,
              bitset_result.matched_count_,
              res_size,
              topk);
    return res_size;
  }

  auto execute_brute_force_filter(const DataType *query,
                                  IDType *ids,
                                  uint32_t topk,
                                  const FilterExecutor &filter_executor,
                                  const char *search_name)
      -> uint32_t requires(DistanceSpaceType::has_scalar_data) {
    SearchBuffer<DistanceType> result_pool(topk);
    std::vector<IDType> batch_ids;
    std::vector<uint8_t> matches;
    constexpr size_t kBatchSize = 1024;
    batch_ids.reserve(kBatchSize);
    auto run_indexed_candidates = [&](const auto &exact_distance) {
      for (auto id : filter_executor.indexed_ids()) {
        if (filter_executor.match(id)) {
          result_pool.insert(id, exact_distance(id));
        }
      }
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
                                const FilterExecutor &filter_executor,
                                const char *search_name)
      -> uint32_t requires(DistanceSpaceType::has_scalar_data) {
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
    LOG_DEBUG("{}: iterative_filter results={}, requested={}",
              search_name,
              res_size,
              search_info.topk_);
    return res_size;
  }

  auto execute_prebuilt_bitset_prefilter(const DataType *query,
                                         IDType *ids,
                                         const SearchInfo &search_info,
                                         const FilterExecutor &filter_executor,
                                         const BlockedBitsetResult &bitset_result,
                                         const char *search_name)
      -> uint32_t requires(DistanceSpaceType::has_scalar_data) {
    (void)filter_executor;
    if (bitset_result.matched_count_ == 0) {
      LOG_DEBUG("{}: bitset_prefilter matched zero rows", search_name);
      return 0;
    }

    GraphSearchJob<DistanceSpaceType, BuildSpaceType> base_job(space_,
                                                               graph_,
                                                               nullptr,
                                                               build_space_);
    if (bitset_result.matched_count_ == space_->get_data_num()) {
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
      LOG_DEBUG("{}: bitset_prefilter switching to brute force, matched_rows={}, topk={}, ef={}",
                search_name,
                bitset_result.matched_count_,
                search_info.topk_,
                search_info.ef_);
      return execute_brute_force_bitset(query, ids, search_info.topk_, bitset_result, search_name);
    }

    auto adjusted_search_info =
        adjust_ef_in_search_info(search_info, bitset_result.matched_count_, search_name);
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
      LOG_DEBUG("{}: bitset_prefilter underfilled results={}, expected={}, fallback to brute force",
                search_name,
                res_size,
                required_results);
      return execute_brute_force_bitset(query, ids, search_info.topk_, bitset_result, search_name);
    }
    return res_size;
  }

  auto execute_bitset_prefilter(const DataType *query,
                                IDType *ids,
                                const SearchInfo &search_info,
                                const FilterExecutor &filter_executor,
                                const char *search_name)
      -> uint32_t requires(DistanceSpaceType::has_scalar_data) {
    auto bitset_result = filter_executor.build_blocked_bitset();
    return execute_prebuilt_bitset_prefilter(query,
                                             ids,
                                             search_info,
                                             filter_executor,
                                             bitset_result,
                                             search_name);
  }

  [[nodiscard]] auto prepare_hybrid_search_plan(const MetadataFilter &filter,
                                                const SearchInfo &search_info) const
      -> PreparedHybridSearchPlan requires(DistanceSpaceType::has_scalar_data) {
    validate_search_info(search_info, "hybrid_search");

    auto filter_executor =
        make_filter_executor(filter, FilterExecutor::IndexBuildMode::kSkip);
    auto direct_bitset_result = filter_executor.build_direct_indexed_blocked_bitset();
    if (direct_bitset_result.has_value()) {
      auto mode = build_search_mode(filter_executor.is_trivially_true(),
                                    true,
                                    direct_bitset_result->matched_count_,
                                    search_info);
      if (mode != Mode::kBitsetPrefilter) {
        filter_executor.materialize_index_fast_path();
      }
      PreparedHybridSearchPlan plan(std::move(filter_executor), mode);
      if (mode == Mode::kBitsetPrefilter) {
        plan.prefilter_result_ = std::move(direct_bitset_result);
      }
      return plan;
    }

    filter_executor.materialize_index_fast_path();
    auto mode = build_search_mode(filter_executor, search_info);
    PreparedHybridSearchPlan plan(std::move(filter_executor), mode);
    if (mode == Mode::kBitsetPrefilter) {
      plan.prefilter_result_ = plan.filter_executor_.build_blocked_bitset();
    }
    return plan;
  }

  void execute_prepared_hybrid_search(const DataType *query,
                                      IDType *ids,
                                      const SearchInfo &search_info,
                                      const PreparedHybridSearchPlan &plan,
                                      std::string *res)
      requires(DistanceSpaceType::has_scalar_data) {
    const char *search_name = "hybrid_search";
    if constexpr (is_rabitq_space_v<DistanceSpaceType>) {
      search_name = "rabitq_hybrid_search";
    }

    initialize_results(ids, res, search_info.topk_);
    LOG_DEBUG("{}: plan={}, topk={}, ef={}, hint={}",
              search_name,
              mode_name(plan.mode_),
              search_info.topk_,
              search_info.ef_,
              static_cast<int>(search_info.filter_exec_hint_));

    uint32_t res_size = 0;
    if (plan.mode_ == Mode::kPlainSearch) {
      GraphSearchJob<DistanceSpaceType, BuildSpaceType> base_job(space_,
                                                                 graph_,
                                                                 nullptr,
                                                                 build_space_);
      if constexpr (is_rabitq_space_v<DistanceSpaceType>) {
        base_job.rabitq_search_solo(query, search_info.topk_, ids, search_info.ef_);
      } else {
        base_job.search_solo(const_cast<DataType *>(query),
                             ids,
                             search_info.topk_,
                             search_info.ef_);
      }
      res_size = std::min<uint32_t>(search_info.topk_, space_->get_data_num());
    } else if (plan.mode_ == Mode::kIndexedExact) {
      res_size = execute_brute_force_filter(query,
                                            ids,
                                            search_info.topk_,
                                            plan.filter_executor_,
                                            search_name);
    } else if (plan.mode_ == Mode::kBitsetPrefilter) {
      if (plan.prefilter_result_.has_value()) {
        res_size = execute_prebuilt_bitset_prefilter(query,
                                                     ids,
                                                     search_info,
                                                     plan.filter_executor_,
                                                     *plan.prefilter_result_,
                                                     search_name);
      } else {
        res_size = execute_bitset_prefilter(query,
                                            ids,
                                            search_info,
                                            plan.filter_executor_,
                                            search_name);
      }
    } else {
      res_size =
          execute_iterative_filter(query, ids, search_info, plan.filter_executor_, search_name);
    }

    materialize_item_ids(ids, res_size, res);
    if (res_size < search_info.topk_) {
      LOG_DEBUG("{}: only found {} results, requested {}",
                search_name,
                res_size,
                search_info.topk_);
    }
  }

  void hybrid_search_solo(DataType *query,
                          IDType *ids,
                          const SearchInfo &search_info,
                          const MetadataFilter &filter,
                          std::string *res) requires(DistanceSpaceType::has_scalar_data) {
    auto plan = prepare_hybrid_search_plan(filter, search_info);
    execute_prepared_hybrid_search(query, ids, search_info, plan, res);
  }

  void hybrid_search_solo(DataType *query,
                          IDType *ids,
                          uint32_t topk,
                          uint32_t ef,
                          const MetadataFilter &filter,
                          std::string *res) requires(DistanceSpaceType::has_scalar_data) {
    hybrid_search_solo(query, ids, SearchInfo{.topk_ = topk, .ef_ = ef}, filter, res);
  }

  void hybrid_search_brute_force_solo(const DataType *query,
                                      IDType *ids,
                                      uint32_t topk,
                                      const MetadataFilter &filter,
                                      std::string *res)
      requires(DistanceSpaceType::has_scalar_data) {
    if (topk == 0) {
      throw std::invalid_argument("hybrid_search_brute_force: topk must be > 0");
    }

    initialize_results(ids, res, topk);
    auto filter_executor = make_filter_executor(filter);
    auto res_size =
        execute_brute_force_filter(query, ids, topk, filter_executor, "hybrid_search_brute_force");
    materialize_item_ids(ids, res_size, res);
    if (res_size < topk) {
      LOG_DEBUG("hybrid_search_brute_force: only found {} results, requested {}", res_size, topk);
    }
  }

  void rabitq_hybrid_search_solo(const DataType *query,
                                 const SearchInfo &search_info,
                                 IDType *ids,
                                 const MetadataFilter &filter,
                                 std::string *res) requires(DistanceSpaceType::has_scalar_data) {
    if constexpr (!is_rabitq_space_v<DistanceSpaceType>) {
      throw std::invalid_argument("Only support RaBitQSpace instance!");
    }

    validate_search_info(search_info, "rabitq_hybrid_search");
    auto plan = prepare_hybrid_search_plan(filter, search_info);
    execute_prepared_hybrid_search(query, ids, search_info, plan, res);
  }

  void rabitq_hybrid_search_solo(const DataType *query,
                                 uint32_t topk,
                                 IDType *ids,
                                 uint32_t ef,
                                 const MetadataFilter &filter,
                                 std::string *res) requires(DistanceSpaceType::has_scalar_data) {
    rabitq_hybrid_search_solo(query, SearchInfo{.topk_ = topk, .ef_ = ef}, ids, filter, res);
  }

#if defined(__linux__)
  auto hybrid_search(DataType *query,
                     IDType *ids,
                     SearchInfo search_info,
                     const MetadataFilter &filter,
                     std::string *res)
      -> coro::task<> requires(DistanceSpaceType::has_scalar_data) {
    hybrid_search_solo(query, ids, search_info, filter, res);
    co_return;
  }

  auto hybrid_search(DataType *query,
                     IDType *ids,
                     uint32_t topk,
                     uint32_t ef,
                     const MetadataFilter &filter,
                     std::string *res)
      -> coro::task<> requires(DistanceSpaceType::has_scalar_data) {
    hybrid_search_solo(query, ids, topk, ef, filter, res);
    co_return;
  }

  auto hybrid_search_brute_force(const DataType *query,
                                 IDType *ids,
                                 uint32_t topk,
                                 const MetadataFilter &filter,
                                 std::string *res)
      -> coro::task<> requires(DistanceSpaceType::has_scalar_data) {
    hybrid_search_brute_force_solo(query, ids, topk, filter, res);
    co_return;
  }

  auto rabitq_hybrid_search(const DataType *query,
                            SearchInfo search_info,
                            IDType *ids,
                            const MetadataFilter &filter,
                            std::string *res)
      -> coro::task<> requires(DistanceSpaceType::has_scalar_data) {
    rabitq_hybrid_search_solo(query, search_info, ids, filter, res);
    co_return;
  }

  auto rabitq_hybrid_search(const DataType *query,
                            uint32_t topk,
                            IDType *ids,
                            uint32_t ef,
                            const MetadataFilter &filter,
                            std::string *res)
      -> coro::task<> requires(DistanceSpaceType::has_scalar_data) {
    rabitq_hybrid_search_solo(query, topk, ids, ef, filter, res);
    co_return;
  }
#endif
  // clang-format on
};

}  // namespace alaya
