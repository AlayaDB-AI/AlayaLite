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
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <sys/types.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include "dispatch.hpp"
#include "executor/jobs/graph_hybrid_search_job.hpp"
#include "executor/jobs/graph_search_job.hpp"
#include "executor/jobs/graph_update_job.hpp"
#include "executor/scheduler.hpp"
#include "index/graph/fusion_graph.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "index/graph/nsg/nsg_builder.hpp"
#include "index/graph/qg/qg_builder.hpp"
#include "materialized_view.hpp"
#include "params.hpp"
#include "parse.hpp"
#include "space/rabitq_space.hpp"
#include "space/raw_space.hpp"
#include "space/sq4_space.hpp"
#include "space/sq8_space.hpp"
#include "storage/rocksdb_storage.hpp"
#include "utils/index_encoding.hpp"
#include "utils/log.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/metric_type.hpp"
#include "utils/scalar_data.hpp"
#include "utils/thread_pool.hpp"
#include "utils/types.hpp"

namespace py = pybind11;

namespace alaya {

class BasePyIndex {
 public:
  uint32_t data_dim_{0};
  BasePyIndex() = default;
  ~BasePyIndex() = default;
};

template <typename GraphBuilderType, typename SearchSpaceType>
class PyIndex : public BasePyIndex {
 public:
  using IDType = typename SearchSpaceType::IDTypeAlias;
  using DataType = typename SearchSpaceType::DataTypeAlias;
  using DistanceType = typename SearchSpaceType::DistanceTypeAlias;
  using BuildSpaceType = typename GraphBuilderType::DistanceSpaceTypeAlias;
  using MaterializedViewSearchSpaceType = StripScalarDataT<SearchSpaceType>;
  using MaterializedViewBuildSpaceType = StripScalarDataT<BuildSpaceType>;

  using MaterializedViewCandidate = VectorCandidate<IDType, DistanceType>;

  struct MaterializedViewPartition {
    MetadataValue value_;
    std::string
        encoded_value_;  // encode metadata value into string for efficient RocksDB key prefix scan
    std::vector<IDType>
        local_to_global_ids_;  // mapping from local ID in partition to global ID in the whole index
    std::shared_ptr<MaterializedViewSearchSpaceType> search_space_{
        nullptr};  // space without scalar data for search
    std::shared_ptr<MaterializedViewBuildSpaceType> build_space_{
        nullptr};  // space without scalar data for building graph, can be the same as search_space_
                   // if scalar data is not used in build process
    std::shared_ptr<Graph<DataType, IDType>> graph_{nullptr};
    std::shared_ptr<
        alaya::GraphSearchJob<MaterializedViewSearchSpaceType, MaterializedViewBuildSpaceType>>
        search_job_{nullptr};
  };

  struct MaterializedViewPartitionSeed {
    MetadataValue value_;
    std::string encoded_value_;
    std::vector<IDType> global_ids_;
  };

  PyIndex() = delete;
  explicit PyIndex(IndexParams params) : params_(std::move(params)) {}

  auto to_string() const -> std::string { return "PyIndex"; }
  auto get_materialized_view_partition_count() const -> uint32_t {
    return static_cast<uint32_t>(materialized_view_partitions_.size());
  }

 private:
  static constexpr size_t kMaterializedViewMaxPartitions =
      128;  // to avoid creating too many small partitions which can lead to high search overhead
  static constexpr float kMaterializedViewKnnBFFilterThreshold =
      0.93F;  // if the filter would exclude more than 93% of the data, it's better to do
              // brute-force search on the whole partition instead of building materialized view
  static constexpr float kMaterializedViewBFTopkThreshold =
      0.5F;  // if topk is more than 50% of the data, it's better to do brute-force search on the
             // whole partition instead of building materialized view
  static constexpr size_t kMaterializedViewRaBitQExactPartitionThreshold =
      128;  // small RaBitQ MV partitions are cheaper and more stable to scan exactly than to build
            // a per-partition QG index

  static auto count_result_ids(const IDType *ids, uint32_t topk) -> uint32_t {
    uint32_t count = 0;
    while (count < topk && ids[count] != std::numeric_limits<IDType>::max()) {
      ++count;
    }
    return count;
  }

  static void insert_materialized_view_candidate(std::vector<MaterializedViewCandidate> &results,
                                                 IDType id,
                                                 DistanceType distance,
                                                 size_t limit) {
    // todo: switch this merge buffer to SearchBuffer once it supports generic ID types cleanly.
    if (limit == 0) {
      return;
    }
    if (results.size() == limit && distance >= results.back().distance_) {
      return;
    }

    auto it = std::upper_bound(results.begin(),
                               results.end(),
                               distance,
                               [](const DistanceType &lhs, const MaterializedViewCandidate &rhs) {
                                 return lhs < rhs.distance_;
                               });
    results.insert(it, MaterializedViewCandidate{id, distance});
    if (results.size() > limit) {
      results.pop_back();
    }
  }

  static auto should_use_materialized_view_brute_force(const SearchInfo &search_info,
                                                       size_t total_count,
                                                       size_t matched_count) -> bool {
    if (matched_count == 0) {
      return false;
    }

    // 1. demanding too much results
    auto topk = static_cast<size_t>(search_info.topk_);
    if (topk >=
        static_cast<size_t>(static_cast<double>(total_count) * kMaterializedViewBFTopkThreshold)) {
      return true;
    }

    // 2. filter is too selective
    auto filtered_out_num = total_count - matched_count;
    if (filtered_out_num >= static_cast<size_t>(static_cast<double>(total_count) *
                                                kMaterializedViewKnnBFFilterThreshold)) {
      return true;
    }

    // 3. topk is too large compared to matched count
    return topk >= static_cast<size_t>(static_cast<double>(matched_count) *
                                       kMaterializedViewBFTopkThreshold);
  }

  /**
   * @brief Adjust the search info for materialized view based on partition characteristics
   * @param search_info Original search info
   * @param partition_size Number of items in this particular partition
   * @param matched_count Number of matched items that are not filtered out by the entire filter in
   * this particular partition, usually matched_count <= partition_size
   * @return Adjusted search info for materialized view search
   */
  static auto adjust_materialized_view_search_info(const SearchInfo &search_info,
                                                   size_t partition_size,
                                                   size_t matched_count) -> SearchInfo {
    SearchInfo adjusted = search_info;
    // make sure topk and ef are not larger than partition size
    adjusted.topk_ = static_cast<uint32_t>(std::min<size_t>(search_info.topk_, partition_size));
    adjusted.ef_ = static_cast<uint32_t>(
        std::min<size_t>(partition_size, std::max<uint32_t>(search_info.ef_, adjusted.topk_)));

    if (matched_count == 0 || matched_count == partition_size) {
      return adjusted;
    }

    // expected scenario: topk / ef = matched_count / partition_size => ef = topk * partition_size /
    // matched_count
    auto expected_ef = static_cast<size_t>(
        (static_cast<double>(adjusted.topk_) * static_cast<double>(partition_size)) /
        static_cast<double>(matched_count));

    expected_ef += expected_ef / 2;

    // bigger ef means more candidates to rerank, but we need to still make sure it is not larger
    // than partition size
    adjusted.ef_ = static_cast<uint32_t>(
        std::min<size_t>(partition_size, std::max<size_t>(adjusted.ef_, expected_ef)));
    return adjusted;
  }

  void reset_materialized_view() {
    materialized_view_ready_ = false;
    materialized_view_field_.clear();
    materialized_view_partition_lookup_.clear();
    materialized_view_partitions_.clear();
  }

  void invalidate_materialized_view(std::string_view reason) {
    if (materialized_view_ready_) {
      LOG_DEBUG("materialized_view: invalidate cached partitions, reason={}", reason);
    }
    reset_materialized_view();
  }

  auto is_materialized_view_supported() const -> bool {
    if constexpr (!SearchSpaceType::has_scalar_data) {
      return false;
    }

    if (search_space_ == nullptr || params_.indexed_fields_.empty()) {
      return false;
    }

    if constexpr (!is_rabitq_space_v<SearchSpaceType>) {
      if (build_space_ == nullptr || params_.index_type_ != IndexType::HNSW) {
        return false;
      }
    }

    return true;
  }

  void copy_materialized_view_source_vector(IDType global_id, DataType *dst) const {
    // todo: for rabitq space, assign build_space_ = search_space_ to avoid duplicate data storage.
    if constexpr (is_rabitq_space_v<SearchSpaceType>) {
      auto *src = search_space_->get_data_by_id(global_id);
      std::memcpy(dst, src, static_cast<size_t>(data_dim_) * sizeof(DataType));
    } else {
      auto *src = build_space_->get_data_by_id(global_id);
      std::memcpy(dst, src, static_cast<size_t>(data_dim_) * sizeof(DataType));
    }
  }

  auto collect_materialized_view_partition_seeds(const RocksDBStorage<IDType> &storage) const
      -> std::optional<std::vector<MaterializedViewPartitionSeed>> {
    std::vector<MaterializedViewPartitionSeed> partitions;
    std::unordered_map<std::string, size_t> partition_lookup;

    // todo: build materialized views asynchronously or on demand for large datasets.
    for (IDType id = 0; id < search_space_->get_data_num(); ++id) {
      std::string raw_value;
      if (!storage.get_raw_value(id, raw_value)) {
        continue;
      }

      auto field_value = ScalarData::deserialize_single_metadata_value(raw_value.data(),
                                                                       raw_value.size(),
                                                                       materialized_view_field_);
      if (!field_value.has_value()) {
        // todo: missing partition field may deserve an explicit error instead of being skipped.
        continue;
      }

      auto encoded_value = index_encoding::encode_value(*field_value);
      auto lookup_it = partition_lookup.find(encoded_value);
      if (lookup_it == partition_lookup.end()) {
        lookup_it = partition_lookup.emplace(encoded_value, partitions.size()).first;
        partitions.push_back(MaterializedViewPartitionSeed{*field_value, encoded_value, {}});
        if (partitions.size() > kMaterializedViewMaxPartitions) {
          return std::nullopt;
        }
      }
      partitions[lookup_it->second].global_ids_.push_back(id);
    }

    return partitions;
  }

  template <typename ExactDistanceEvaluator>
  void emit_materialized_view_brute_force_results(
      const MaterializedViewPartition &partition,
      const SearchInfo &search_info,
      const DynamicBitset *residual_blocked,
      ExactDistanceEvaluator &&exact_distance,
      std::vector<MaterializedViewCandidate> &results) const {
    auto partition_size = partition.local_to_global_ids_.size();
    for (size_t local_id = 0; local_id < partition_size; ++local_id) {
      if (residual_blocked != nullptr && residual_blocked->get(local_id)) {
        continue;
      }
      insert_materialized_view_candidate(results,
                                         partition.local_to_global_ids_[local_id],
                                         exact_distance(static_cast<IDType>(local_id)),
                                         search_info.topk_);
    }
  }

  template <typename ExactDistanceEvaluator>
  void search_materialized_view_partition_with_exact_distance(
      const MaterializedViewPartition &partition,
      const DataType *query,
      const SearchInfo &search_info,
      const SearchInfo &local_search_info,
      size_t matched_count,
      const DynamicBitset *residual_blocked,
      bool filter_covered,
      bool brute_force_requested,
      ExactDistanceEvaluator &&exact_distance,
      std::vector<MaterializedViewCandidate> &results) const {
    auto partition_size = partition.local_to_global_ids_.size();

    // 1. demanded brute_force search
    if (brute_force_requested || partition.search_job_ == nullptr ||
        should_use_materialized_view_brute_force(local_search_info,
                                                 partition_size,
                                                 matched_count)) {
      emit_materialized_view_brute_force_results(partition,
                                                 local_search_info,
                                                 residual_blocked,
                                                 std::forward<ExactDistanceEvaluator>(
                                                     exact_distance),
                                                 results);
      return;
    }

    auto required_results =
        static_cast<uint32_t>(std::min<size_t>(local_search_info.topk_, matched_count));
    if (required_results == 0) {
      return;
    }

    // 2. partition satisfies search condition, do simple ann search
    if (filter_covered) {
      std::vector<IDType> local_ids(local_search_info.topk_, std::numeric_limits<IDType>::max());
      if constexpr (is_rabitq_space_v<MaterializedViewSearchSpaceType>) {
        partition.search_job_->rabitq_search_solo(query,
                                                  local_search_info.topk_,
                                                  local_ids.data(),
                                                  local_search_info);
      } else {
        partition.search_job_->search_solo(const_cast<DataType *>(query),
                                           local_ids.data(),
                                           local_search_info);
      }

      auto found_results = count_result_ids(local_ids.data(), local_search_info.topk_);
      if (found_results < required_results) {
        emit_materialized_view_brute_force_results(partition,
                                                   local_search_info,
                                                   residual_blocked,
                                                   std::forward<ExactDistanceEvaluator>(
                                                       exact_distance),
                                                   results);
        return;
      }

      for (uint32_t i = 0; i < found_results; ++i) {
        auto local_id = local_ids[i];
        insert_materialized_view_candidate(results,
                                           partition.local_to_global_ids_[local_id],
                                           exact_distance(local_id),
                                           local_search_info.topk_);
      }
      return;
    }

    // 3. iterative post-filtering
    if (search_info.filter_exec_hint_ == FilterExecHint::kIterativeFilter) {
      // todo: inflate ef iteratively when post-filtering cannot collect enough results.
      auto iterator = partition.search_job_->make_vector_iterator(query, local_search_info);
      while (results.size() < local_search_info.topk_ && iterator->has_next()) {
        auto candidate = iterator->next();
        if (!candidate.has_value()) {
          break;
        }
        if (residual_blocked != nullptr && residual_blocked->get(candidate->id_)) {
          continue;
        }
        insert_materialized_view_candidate(results,
                                           partition.local_to_global_ids_[candidate->id_],
                                           exact_distance(candidate->id_),
                                           local_search_info.topk_);
      }
      return;
    }

    // 4. blocked bitmap pre-filtering
    auto adjusted_search_info =
        adjust_materialized_view_search_info(local_search_info, partition_size, matched_count);
    std::vector<IDType> local_ids(adjusted_search_info.topk_, std::numeric_limits<IDType>::max());
    if constexpr (is_rabitq_space_v<MaterializedViewSearchSpaceType>) {
      partition.search_job_->rabitq_search_solo(query,
                                                adjusted_search_info.topk_,
                                                local_ids.data(),
                                                adjusted_search_info,
                                                residual_blocked);
    } else {
      partition.search_job_->search_solo(const_cast<DataType *>(query),
                                         local_ids.data(),
                                         adjusted_search_info,
                                         residual_blocked);
    }

    // 5. can not find enough result, fall back to brute-force search
    auto found_results = count_result_ids(local_ids.data(), adjusted_search_info.topk_);
    if (found_results < required_results) {
      emit_materialized_view_brute_force_results(partition,
                                                 local_search_info,
                                                 residual_blocked,
                                                 std::forward<ExactDistanceEvaluator>(
                                                     exact_distance),
                                                 results);
      return;
    }

    for (uint32_t i = 0; i < found_results; ++i) {
      auto local_id = local_ids[i];
      insert_materialized_view_candidate(results,
                                         partition.local_to_global_ids_[local_id],
                                         exact_distance(local_id),
                                         local_search_info.topk_);
    }
  }

  auto build_materialized_view_partition(const MetadataValue &value,
                                         std::string encoded_value,
                                         std::vector<IDType> global_ids)
      -> MaterializedViewPartition {
    MaterializedViewPartition partition;
    partition.value_ = value;
    partition.encoded_value_ = std::move(encoded_value);
    partition.local_to_global_ids_ = std::move(global_ids);

    auto partition_size = partition.local_to_global_ids_.size();
    auto partition_capacity = static_cast<IDType>(partition_size);
    std::vector<DataType> partition_vectors(partition_size * static_cast<size_t>(data_dim_));
    for (size_t i = 0; i < partition_size; ++i) {
      copy_materialized_view_source_vector(partition.local_to_global_ids_[i],
                                           partition_vectors.data() +
                                               (i * static_cast<size_t>(data_dim_)));
    }

    partition.build_space_ = std::make_shared<MaterializedViewBuildSpaceType>(partition_capacity,
                                                                              data_dim_,
                                                                              params_.metric_);
    partition.build_space_->fit(partition_vectors.data(), partition_capacity);

    if constexpr (std::is_same_v<MaterializedViewBuildSpaceType, MaterializedViewSearchSpaceType>) {
      partition.search_space_ = partition.build_space_;
    } else {
      partition.search_space_ =
          std::make_shared<MaterializedViewSearchSpaceType>(partition_capacity,
                                                            data_dim_,
                                                            params_.metric_);
      partition.search_space_->fit(partition_vectors.data(), partition_capacity);
    }

    // todo: small partitions may not need a child index; choose a threshold with benchmarks.
    if constexpr (is_rabitq_space_v<MaterializedViewSearchSpaceType>) {
      if (partition_size > kMaterializedViewRaBitQExactPartitionThreshold) {
        QGBuilder<MaterializedViewSearchSpaceType> graph_builder(partition.search_space_);
        graph_builder.build_graph();
        partition.search_job_ = std::make_shared<
            alaya::GraphSearchJob<MaterializedViewSearchSpaceType,
                                  MaterializedViewBuildSpaceType>>(partition.search_space_,
                                                                   nullptr,
                                                                   nullptr,
                                                                   partition.build_space_);
      }
    } else {
      if (partition_size == 1) {
        partition.graph_ =
            std::make_shared<Graph<DataType, IDType>>(partition_capacity, params_.max_nbrs_);
        partition.graph_->eps_.push_back(0);
      } else {
        HNSWBuilder<MaterializedViewBuildSpaceType>
            graph_builder(partition.build_space_,
                          params_.max_nbrs_,
                          materialized_view_ef_construction_);
        partition.graph_ = std::shared_ptr<Graph<DataType, IDType>>(
            graph_builder.build_graph(materialized_view_build_threads_).release());
      }
      partition.search_job_ = std::make_shared<
          alaya::GraphSearchJob<MaterializedViewSearchSpaceType,
                                MaterializedViewBuildSpaceType>>(partition.search_space_,
                                                                 partition.graph_,
                                                                 nullptr,
                                                                 partition.build_space_);
    }

    return partition;
  }

  void try_build_materialized_view() {
    reset_materialized_view();
    if (!is_materialized_view_supported()) {
      return;
    }

    auto *storage = search_space_->get_scalar_storage();
    if (storage == nullptr) {
      return;
    }

    materialized_view_field_ = params_.indexed_fields_.front();
    if (params_.indexed_fields_.size() > 1) {
      LOG_INFO("materialized_view: only the first indexed field is partitioned, field={}",
               materialized_view_field_);
    }

    try {
      auto partitions = collect_materialized_view_partition_seeds(*storage);
      if (!partitions.has_value()) {
        LOG_INFO("materialized_view: skip build, field={} has too many partitions (>{})",
                 materialized_view_field_,
                 kMaterializedViewMaxPartitions);
        reset_materialized_view();
        return;
      }

      if (partitions->size() <= 1) {
        LOG_INFO("materialized_view: skip build, field={} has {} partition(s)",
                 materialized_view_field_,
                 partitions->size());
        reset_materialized_view();
        return;
      }

      materialized_view_partitions_.reserve(partitions->size());
      materialized_view_partition_lookup_.reserve(partitions->size());
      for (auto &partition_seed : *partitions) {
        auto partition = build_materialized_view_partition(partition_seed.value_,
                                                           std::move(partition_seed.encoded_value_),
                                                           std::move(partition_seed.global_ids_));
        materialized_view_partition_lookup_.emplace(partition.encoded_value_,
                                                    materialized_view_partitions_.size());
        materialized_view_partitions_.push_back(std::move(partition));
      }
      materialized_view_ready_ = !materialized_view_partitions_.empty();
      LOG_INFO("materialized_view: built field={}, partitions={}",
               materialized_view_field_,
               materialized_view_partitions_.size());
    } catch (const std::exception &e) {
      LOG_ERROR("materialized_view: build failed for field={}, error={}",
                materialized_view_field_,
                e.what());
      reset_materialized_view();
    }
  }

  auto execute_materialized_view_partition_search(
      const MaterializedViewPartition &partition,
      const DataType *query,
      const SearchInfo &search_info,
      const MetadataFilterExecutor<IDType> *filter_executor,
      bool brute_force_requested,
      bool filter_covered) const -> std::vector<MaterializedViewCandidate> {
    std::vector<MaterializedViewCandidate> results;
    auto partition_size = partition.local_to_global_ids_.size();
    if (partition_size == 0) {
      return results;
    }

    // initialize local_search_info
    SearchInfo local_search_info =
        adjust_materialized_view_search_info(search_info, partition_size, partition_size);

    std::optional<typename MetadataFilterExecutor<IDType>::BlockedBitsetResult>
        residual_filter_result;
    const DynamicBitset *residual_blocked = nullptr;
    auto matched_count = partition_size;
    if (!filter_covered) {
      assert(filter_executor != nullptr);
      residual_filter_result =
          filter_executor->build_blocked_bitset(partition.local_to_global_ids_);
      matched_count = residual_filter_result->matched_count_;
      if (matched_count == 0) {
        return results;
      }
      residual_blocked = &residual_filter_result->blocked_;
    }

    auto dist_func = partition.build_space_->get_dist_func();
    auto dim = partition.build_space_->get_dim();
    search_materialized_view_partition_with_exact_distance(
        partition,
        query,
        search_info,
        local_search_info,
        matched_count,
        residual_blocked,
        filter_covered,
        brute_force_requested,
        [&](IDType local_id) -> DistanceType {
          return dist_func(query, partition.build_space_->get_data_by_id(local_id), dim);
        },
        results);
    return results;
  }  // execute_materialized_view_partition_search

  auto try_materialized_view_hybrid_search(const DataType *query,
                                           IDType *ids,
                                           const SearchInfo &search_info,
                                           const MetadataFilter &filter,
                                           bool brute_force_requested,
                                           std::string *item_ids) const -> bool {
    if (!materialized_view_ready_ || filter.is_empty()) {
      return false;
    }
    auto partition_selection = analyze_materialized_view_filter(filter, materialized_view_field_);
    if (!partition_selection.eligible_) {
      return false;
    }
    auto *storage = search_space_->get_scalar_storage();
    if (storage == nullptr) {
      return false;
    }

    std::fill(ids, ids + search_info.topk_, std::numeric_limits<IDType>::max());
    std::fill(item_ids, item_ids + search_info.topk_, std::string{});

    std::unique_ptr<MetadataFilterExecutor<IDType>> filter_executor;
    if (!partition_selection.filter_covered_) {
      filter_executor =
          std::make_unique<MetadataFilterExecutor<IDType>>(filter,
                                                           storage,
                                                           search_space_->get_data_num());
    }

    std::vector<MaterializedViewCandidate> merged_results;
    merged_results.reserve(search_info.topk_);

    size_t selected_partitions = 0;
    for (const auto &value : partition_selection.values_) {
      auto lookup_it =
          materialized_view_partition_lookup_.find(index_encoding::encode_value(value));
      if (lookup_it == materialized_view_partition_lookup_.end()) {
        // filter value does not match any partition, skip this value since it won't contribute to
        // the result
        continue;
      }

      ++selected_partitions;
      auto partition_results =
          execute_materialized_view_partition_search(materialized_view_partitions_[lookup_it
                                                                                       ->second],
                                                     query,
                                                     search_info,
                                                     filter_executor.get(),
                                                     brute_force_requested,
                                                     partition_selection.filter_covered_);
      for (const auto &candidate : partition_results) {
        insert_materialized_view_candidate(merged_results,
                                           candidate.id_,
                                           candidate.distance_,
                                           search_info.topk_);
      }
    }

    LOG_DEBUG("hybrid_search: plan=materialized_view, field={}, partitions={}",
              materialized_view_field_,
              selected_partitions);

    if (merged_results.empty()) {  // no available result
      return true;
    }

    std::vector<IDType> materialized_ids;
    materialized_ids.reserve(merged_results.size());
    for (size_t i = 0; i < merged_results.size(); ++i) {
      ids[i] = merged_results[i].id_;
      materialized_ids.push_back(merged_results[i].id_);
    }

    auto materialized_item_ids = storage->batch_get_item_id_only(materialized_ids);
    for (size_t i = 0; i < materialized_item_ids.size(); ++i) {
      item_ids[i] = std::move(materialized_item_ids[i]);
    }
    return true;
  }

  void execute_hybrid_search_dispatch(const DataType *query,
                                      IDType *ids,
                                      const SearchInfo &search_info,
                                      const MetadataFilter &filter,
                                      bool brute_force_requested,
                                      std::string *item_ids) const {
    if (try_materialized_view_hybrid_search(query,
                                            ids,
                                            search_info,
                                            filter,
                                            brute_force_requested,
                                            item_ids)) {
      return;
    }

    // materialized view optimization is not available, fall back to original hybrid search
    if (brute_force_requested) {
      hybrid_search_job_->hybrid_search_brute_force_solo(query,
                                                         ids,
                                                         search_info.topk_,
                                                         filter,
                                                         item_ids);
    } else if constexpr (is_rabitq_space_v<SearchSpaceType>) {
      hybrid_search_job_->rabitq_hybrid_search_solo(query, search_info, ids, filter, item_ids);
    } else {
      hybrid_search_job_->hybrid_search_solo(const_cast<DataType *>(query),
                                             ids,
                                             search_info,
                                             filter,
                                             item_ids);
    }
  }

  // todo: this cache may become a bottleneck under frequent thread-count changes.
  // Cache a thread pool per requested width to amortize batch-search setup.
  auto get_hybrid_batch_pool(uint32_t requested_threads) -> std::shared_ptr<alaya::ThreadPool> {
    auto effective_threads = std::max<uint32_t>(1, requested_threads);
    std::lock_guard<std::mutex> lock(hybrid_batch_pool_mutex_);
    if (hybrid_batch_pool_ == nullptr || hybrid_batch_pool_threads_ != effective_threads) {
      hybrid_batch_pool_ = std::make_shared<alaya::ThreadPool>(effective_threads);
      hybrid_batch_pool_threads_ = effective_threads;
    }
    return hybrid_batch_pool_;
  }

#if defined(__linux__)
  // add coroutine support
  auto execute_hybrid_search_dispatch_task(const DataType *query,
                                           IDType *ids,
                                           SearchInfo search_info,
                                           const MetadataFilter &filter,
                                           bool brute_force_requested,
                                           std::string *item_ids) const -> coro::task<> {
    execute_hybrid_search_dispatch(query,
                                   ids,
                                   search_info,
                                   filter,
                                   brute_force_requested,
                                   item_ids);
    co_return;
  }
#endif

 public:
  auto get_data_by_id(IDType id) -> py::array_t<DataType> {
    if constexpr (is_rabitq_space_v<SearchSpaceType>) {
      if (search_space_ == nullptr) {
        throw std::runtime_error("space is nullptr");
      }
      if (id >= search_space_->get_data_num()) {
        throw std::runtime_error("id out of range");
      }
      auto data = search_space_->get_data_by_id(id);
      return py::array_t<DataType>({data_dim_}, {sizeof(DataType)}, data);
    } else {
      if (build_space_ == nullptr) {
        throw std::runtime_error("space is nullptr");
      }
      if (id >= build_space_->get_data_num()) {
        throw std::runtime_error("id out of range");
      }
      auto data = build_space_->get_data_by_id(id);
      return py::array_t<DataType>({data_dim_}, {sizeof(DataType)}, data);
    }
  }

  auto get_dim() const -> uint32_t { return data_dim_; }

  auto save(const std::string &index_path,
            const std::string &data_path = std::string(),
            const std::string &quant_path = std::string()) -> void {
    std::string_view index_path_view{index_path};
    std::string_view data_path_view{data_path};
    std::string_view quant_path_view{quant_path};

    if constexpr (!is_rabitq_space_v<SearchSpaceType>) {
      graph_index_->save(index_path_view);
      if (!data_path.empty()) {
        build_space_->save(data_path_view);
      }
    }

    if (!quant_path.empty()) {
      search_space_->save(quant_path_view);
    }
  }

  auto load(const std::string &index_path,
            const std::string &data_path = std::string(),
            const std::string &quant_path = std::string()) -> void {
    // index_path_ = index_path;
    std::string_view index_path_view{index_path};
    std::string_view data_path_view{data_path};
    std::string_view quant_path_view{quant_path};

    if constexpr (is_rabitq_space_v<SearchSpaceType>) {
      search_space_ = std::make_shared<SearchSpaceType>();
      search_space_->load(quant_path_view);
      data_size_ = search_space_->get_data_size();
      data_dim_ = search_space_->get_dim();
      search_job_ =
          std::make_shared<alaya::GraphSearchJob<SearchSpaceType, BuildSpaceType>>(search_space_,
                                                                                   nullptr,
                                                                                   nullptr,
                                                                                   build_space_);
      hybrid_search_job_ = std::make_shared<
          alaya::GraphHybridSearchJob<SearchSpaceType, BuildSpaceType>>(search_space_,
                                                                        nullptr,
                                                                        build_space_);
    } else {
      graph_index_ = std::make_shared<Graph<DataType, IDType>>();
      graph_index_->load(index_path_view);

      if (!data_path.empty()) {
        build_space_ = std::make_shared<BuildSpaceType>();
        build_space_->load(data_path_view);
        build_space_->set_metric_function();
      }

      if constexpr (std::is_same<BuildSpaceType, SearchSpaceType>::value) {
        search_space_ = build_space_;
      } else {
        search_space_ = std::make_shared<SearchSpaceType>();
        search_space_->load(quant_path_view);
        search_space_->set_metric_function();
      }

      data_size_ = build_space_->get_data_size();
      data_dim_ = build_space_->get_dim();

      job_context_ = std::make_shared<JobContext<IDType>>();

      search_job_ =
          std::make_shared<alaya::GraphSearchJob<SearchSpaceType, BuildSpaceType>>(search_space_,
                                                                                   graph_index_,
                                                                                   job_context_,
                                                                                   build_space_);
      hybrid_search_job_ = std::make_shared<
          alaya::GraphHybridSearchJob<SearchSpaceType, BuildSpaceType>>(search_space_,
                                                                        graph_index_,
                                                                        build_space_);
      update_job_ = std::make_shared<GraphUpdateJob<SearchSpaceType, BuildSpaceType>>(search_job_);
    }
    materialized_view_ef_construction_ = std::max<uint32_t>(200, params_.max_nbrs_ * 4);
    materialized_view_build_threads_ = params_.materialized_view_build_threads_ != 0
                                           ? params_.materialized_view_build_threads_
                                       : params_.build_threads_ != 0 ? params_.build_threads_
                                                                     : 1;
    try_build_materialized_view();
    LOG_DEBUG("creator task generator success");
  }

  auto fit(py::array_t<DataType> vectors,
           uint32_t ef_construction,
           uint32_t num_threads,
           const py::object &item_ids = py::none(),
           const py::object &documents = py::none(),
           const py::object &metadata_list = py::none()) -> void {
    LOG_INFO("start fit data");

    if (vectors.ndim() != 2) {
      throw std::runtime_error("Array must be 2D");
    }

    data_size_ = vectors.shape(0);
    data_dim_ = vectors.shape(1);
    vectors_ = static_cast<DataType *>(vectors.request().ptr);
    materialized_view_ef_construction_ = ef_construction;
    materialized_view_build_threads_ = params_.materialized_view_build_threads_ != 0
                                           ? params_.materialized_view_build_threads_
                                           : std::max<uint32_t>(1, num_threads);

    // Build ScalarData array if provided (only for search_space_)
    std::vector<ScalarData> scalar_data_vec;
    bool has_scalar = !item_ids.is_none();

    if (has_scalar) {
      scalar_data_vec =
          build_scalar_data_vec(item_ids.cast<py::list>(), documents, metadata_list, data_size_);
    }
    ScalarData *scalar_ptr = has_scalar ? scalar_data_vec.data() : nullptr;

    // Create RocksDB config with custom path if provided
    RocksDBConfig rocksdb_config = RocksDBConfig::default_config();
    if (!params_.rocksdb_path_.empty()) {
      rocksdb_config.db_path_ = params_.rocksdb_path_;
    }
    // Set indexed fields for fast filtering
    rocksdb_config.indexed_fields_ = params_.indexed_fields_;

    // Keep the RaBitQ branch separate until the graph-builder path is unified.
    if constexpr (is_rabitq_space_v<SearchSpaceType>) {
      if constexpr (SearchSpaceType::has_scalar_data) {
        search_space_ = std::make_shared<SearchSpaceType>(params_.capacity_,
                                                          data_dim_,
                                                          params_.metric_,
                                                          rocksdb_config);
        search_space_->fit(vectors_, data_size_, scalar_ptr);
      } else {
        search_space_ =
            std::make_shared<SearchSpaceType>(params_.capacity_, data_dim_, params_.metric_);
        search_space_->fit(vectors_, data_size_);
      }
      auto graph_builder = std::make_shared<QGBuilder<SearchSpaceType>>(search_space_);
      graph_builder->build_graph();
      search_job_ =
          std::make_shared<alaya::GraphSearchJob<SearchSpaceType, BuildSpaceType>>(search_space_,
                                                                                   nullptr,
                                                                                   nullptr,
                                                                                   build_space_);
      hybrid_search_job_ = std::make_shared<
          alaya::GraphHybridSearchJob<SearchSpaceType, BuildSpaceType>>(search_space_,
                                                                        nullptr,
                                                                        build_space_);
    } else {
      if constexpr (BuildSpaceType::has_scalar_data) {
        build_space_ = std::make_shared<BuildSpaceType>(params_.capacity_,
                                                        data_dim_,
                                                        params_.metric_,
                                                        rocksdb_config);
      } else {
        build_space_ =
            std::make_shared<BuildSpaceType>(params_.capacity_, data_dim_, params_.metric_);
      }

      if constexpr (std::is_same<BuildSpaceType, SearchSpaceType>::value) {
        // When BuildSpaceType == SearchSpaceType, pass scalar data to build_space
        if constexpr (BuildSpaceType::has_scalar_data) {
          build_space_->fit(vectors_, data_size_, scalar_ptr);
        } else {
          build_space_->fit(vectors_, data_size_);
        }
        search_space_ = build_space_;
      } else {
        build_space_->fit(vectors_, data_size_);

        if constexpr (SearchSpaceType::has_scalar_data) {
          search_space_ = std::make_shared<SearchSpaceType>(params_.capacity_,
                                                            data_dim_,
                                                            params_.metric_,
                                                            rocksdb_config);
          search_space_->fit(vectors_, data_size_, scalar_ptr);
        } else {
          search_space_ =
              std::make_shared<SearchSpaceType>(params_.capacity_, data_dim_, params_.metric_);
          search_space_->fit(vectors_, data_size_);
        }
      }

      auto build_start = std::chrono::steady_clock::now();
      auto graph_builder = std::make_shared<HNSWBuilder<BuildSpaceType>>(build_space_,
                                                                         params_.max_nbrs_,
                                                                         ef_construction);
      graph_index_ = graph_builder->build_graph(num_threads);

      LOG_INFO("The time of building hnsw is {}s.",
               static_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() -
                                                          build_start)
                   .count());

      job_context_ = std::make_shared<JobContext<IDType>>();

      search_job_ =
          std::make_shared<alaya::GraphSearchJob<SearchSpaceType, BuildSpaceType>>(search_space_,
                                                                                   graph_index_,
                                                                                   job_context_,
                                                                                   build_space_);
      hybrid_search_job_ = std::make_shared<
          alaya::GraphHybridSearchJob<SearchSpaceType, BuildSpaceType>>(search_space_,
                                                                        graph_index_,
                                                                        build_space_);
      update_job_ = std::make_shared<GraphUpdateJob<SearchSpaceType, BuildSpaceType>>(search_job_);
    }
    try_build_materialized_view();
    LOG_DEBUG("Create task generator successfully!");
  }

  auto insert(py::array_t<DataType> insert_data,
              uint32_t ef,
              const std::string &item_id = "",
              const std::string &document = "",
              const py::dict &metadata = py::dict()) -> IDType {
    if (update_job_ == nullptr) {
      throw std::runtime_error("incremental updates are not supported for the current index type");
    }
    auto insert_data_ptr = static_cast<DataType *>(insert_data.request().ptr);
    MetadataMap meta_map = pydict_to_metadata_map(metadata);
    ScalarData scalar_data{item_id, document, meta_map};
    auto inserted_id = update_job_->insert_and_update(insert_data_ptr, ef, &scalar_data);
    invalidate_materialized_view("insert");
    return inserted_id;
  }

  auto remove(uint32_t id) -> void {
    if (update_job_ == nullptr) {
      throw std::runtime_error("incremental updates are not supported for the current index type");
    }
    update_job_->remove(id);
    invalidate_materialized_view("remove");
  }

  auto remove(const std::string &item_id) -> void {
    if (update_job_ == nullptr) {
      throw std::runtime_error("incremental updates are not supported for the current index type");
    }
    update_job_->remove(item_id);
    invalidate_materialized_view("remove_by_item_id");
  }

  /**
   * @brief Check if item_id exists in the index
   * @param item_id The item_id to check
   * @return true if exists, false otherwise
   */
  auto contains(const std::string &item_id) -> bool {
    if constexpr (SearchSpaceType::has_scalar_data) {
      try {
        search_space_->get_scalar_data(item_id);
        return true;
      } catch (...) {
        return false;
      }
    }
    return false;
  }

  /**
   * @brief Get scalar data by item_id
   * @param item_id The item_id to look up
   * @return Python dict containing internal_id, item_id, document, and metadata
   * @throws std::runtime_error if item_id not found or no scalar data available
   */
  auto get_scalar_data_by_item_id(const std::string &item_id) -> py::dict {
    if constexpr (!SearchSpaceType::has_scalar_data) {
      throw std::runtime_error("get_scalar_data requires a space that supports scalar data");
    } else {
      auto [internal_id, scalar_data] = search_space_->get_scalar_data(item_id);
      py::dict result = scalar_data_to_pydict(scalar_data);
      result["internal_id"] = internal_id;
      return result;
    }
  }

  /**
   * @brief Get scalar data by internal ID
   * @param internal_id The internal ID
   * @return Python dict containing item_id, document, and metadata
   */
  auto get_scalar_data_by_internal_id(IDType internal_id) -> py::dict {
    if constexpr (!SearchSpaceType::has_scalar_data) {
      throw std::runtime_error("get_scalar_data requires a space that supports scalar data");
    } else {
      decltype(search_space_->get_scalar_data(internal_id)) scalar_data;
      {
        py::gil_scoped_release release;
        scalar_data = search_space_->get_scalar_data(internal_id);
      }
      return scalar_data_to_pydict(scalar_data);
    }
  }

  /**
   * @brief Batch get item_ids by internal IDs (lightweight, uses MultiGet)
   * @param internal_ids numpy array of internal IDs
   * @return Python list of item_id strings
   */
  auto batch_get_item_ids_by_internal_ids(py::array_t<IDType> internal_ids) -> py::list {
    if constexpr (!SearchSpaceType::has_scalar_data) {
      throw std::runtime_error("batch_get_item_ids requires a space that supports scalar data");
    } else {
      auto buf = internal_ids.request();
      auto *id_ptr = static_cast<IDType *>(buf.ptr);
      size_t count = static_cast<size_t>(buf.size);
      std::vector<IDType> ids(id_ptr, id_ptr + count);

      std::vector<std::string> item_ids;
      {
        py::gil_scoped_release release;
        auto *storage = search_space_->get_scalar_storage();
        item_ids = storage->batch_get_item_id_only(ids);
      }

      py::list result;
      for (auto &item_id : item_ids) {
        result.append(std::move(item_id));
      }
      return result;
    }
  }

  /**
   * @brief Get the number of vectors in the index
   * @return Number of vectors
   */
  auto get_data_num() -> IDType {
    if (build_space_ != nullptr) {
      return build_space_->get_data_num();
    } else if (search_space_ != nullptr) {
      return search_space_->get_data_num();
    }
    return 0;
  }

  auto search(py::array_t<DataType> query, uint32_t topk, uint32_t ef) -> py::array_t<IDType> {
    auto *query_ptr = static_cast<DataType *>(query.request().ptr);
    std::vector<IDType> result_ids(topk);

    {
      py::gil_scoped_release release;
      if constexpr (is_rabitq_space_v<SearchSpaceType>) {
        search_job_->rabitq_search_solo(query_ptr, topk, result_ids.data(), ef);
      } else {
        search_job_->search_solo(query_ptr, result_ids.data(), topk, ef);
      }
    }

    auto ret = py::array_t<IDType>(static_cast<size_t>(topk));
    auto ret_ptr = static_cast<IDType *>(ret.request().ptr);
    std::copy(result_ids.begin(), result_ids.end(), ret_ptr);
    return ret;
  }

  auto search_with_distance(py::array_t<DataType> query, uint32_t topk, uint32_t ef) -> py::object {
    if constexpr (is_rabitq_space_v<SearchSpaceType>) {
      throw std::runtime_error("search_with_distance is not supported for RaBitQ space");
    }

    auto *query_ptr = static_cast<DataType *>(query.request().ptr);

    auto ret_ids = py::array_t<IDType>(static_cast<size_t>(topk));
    auto ret_id_ptr = static_cast<IDType *>(ret_ids.request().ptr);

    auto ret_dists = py::array_t<DistanceType>(static_cast<size_t>(topk));
    auto ret_dist_ptr = static_cast<DistanceType *>(ret_dists.request().ptr);

    search_job_->search_solo(query_ptr, ret_id_ptr, ret_dist_ptr, topk, ef);

    return py::make_tuple(ret_ids, ret_dists);
  }

  /**
   * @brief Hybrid search with metadata filtering
   * @param query Query vector
   * @param topk Number of results to return
   * @param ef Number of candidates to explore
   * @param filter Metadata filter for filtering results
   * @return Tuple of (ids, item_ids)
   */
  auto hybrid_search(py::array_t<DataType> query,
                     uint32_t topk,
                     uint32_t ef,
                     const MetadataFilter &filter,
                     bool bf = false,
                     const std::string &filter_exec_hint = std::string()) -> py::object {
    if constexpr (!SearchSpaceType::has_scalar_data) {
      throw std::runtime_error("hybrid_search requires a space that supports scalar data");
    } else {
      auto *query_ptr = static_cast<DataType *>(query.request().ptr);
      SearchInfo search_info{topk, ef, parse_filter_exec_hint(filter_exec_hint)};

      std::vector<IDType> result_ids(topk);
      std::vector<std::string> item_ids(topk);
      {
        py::gil_scoped_release release;
        execute_hybrid_search_dispatch(query_ptr,
                                       result_ids.data(),
                                       search_info,
                                       filter,
                                       bf,
                                       item_ids.data());
      }

      auto ret_ids = py::array_t<IDType>(static_cast<size_t>(topk));
      auto ret_id_ptr = static_cast<IDType *>(ret_ids.request().ptr);
      std::copy(result_ids.begin(), result_ids.end(), ret_id_ptr);

      // Convert item_ids to Python list
      py::list item_id_list;
      for (const auto &item_id : item_ids) {
        item_id_list.append(item_id);
      }

      return py::make_tuple(ret_ids, item_id_list);
    }
  }

  /**
   * @brief Batch hybrid search with metadata filtering (coroutine version)
   * @param queries Query vectors
   * @param topk Number of results per query
   * @param ef Number of candidates to explore
   * @param filter Metadata filter for filtering results
   * @param num_threads Number of threads
   * @return Tuple of (ids_array, item_ids_list_of_lists)
   */
  auto batch_hybrid_search(py::array_t<DataType> queries,
                           uint32_t topk,
                           uint32_t ef,
                           const MetadataFilter &filter,
                           uint32_t num_threads,
                           bool bf = false,
                           const std::string &filter_exec_hint = std::string()) -> py::object {
    if constexpr (!SearchSpaceType::has_scalar_data) {
      throw std::runtime_error("batch_hybrid_search requires a space that supports scalar data");
    } else {
      auto shape = queries.shape();
      size_t query_size = shape[0];
      size_t query_dim = shape[1];
      SearchInfo search_info{topk, ef, parse_filter_exec_hint(filter_exec_hint)};

      auto *query_ptr = static_cast<DataType *>(queries.request().ptr);

      std::vector<std::vector<IDType>> id_results(query_size, std::vector<IDType>(topk));
      std::vector<std::vector<std::string>> item_id_results(query_size,
                                                            std::vector<std::string>(topk));
      {
        py::gil_scoped_release release;
        auto batch_pool = get_hybrid_batch_pool(num_threads);
        std::vector<std::future<void>> futures;
        futures.reserve(query_size);
        for (uint32_t i = 0; i < query_size; i++) {
          auto cur_query = query_ptr + i * query_dim;
          futures.emplace_back(batch_pool->enqueue([this,
                                                    cur_query,
                                                    ids = id_results[i].data(),
                                                    search_info,
                                                    filter_ptr = &filter,
                                                    bf,
                                                    item_ids = item_id_results[i].data()]() {
            execute_hybrid_search_dispatch(cur_query, ids, search_info, *filter_ptr, bf, item_ids);
          }));
        }
        for (auto &future : futures) {
          future.get();
        }
      }

      // Build result arrays (GIL re-acquired)
      auto ret_ids = py::array_t<IDType>({query_size, static_cast<size_t>(topk)});
      auto ret_id_ptr = static_cast<IDType *>(ret_ids.request().ptr);
      for (size_t i = 0; i < query_size; i++) {
        std::copy(id_results[i].begin(), id_results[i].end(), ret_id_ptr + i * topk);
      }

      // Convert item_ids to Python list of lists
      py::list all_item_id_lists;
      for (size_t i = 0; i < query_size; i++) {
        py::list item_id_list;
        for (const auto &item_id : item_id_results[i]) {
          item_id_list.append(item_id);
        }
        all_item_id_lists.append(item_id_list);
      }

      return py::make_tuple(ret_ids, all_item_id_lists);
    }
  }

  /**
   * @brief Filter query without vector search
   * @param filter Metadata filter
   * @param limit Maximum number of results
   * @return Tuple of (ids_list, scalar_data_list)
   */
  auto filter_query(const MetadataFilter &filter, uint32_t limit) -> py::object {
    if constexpr (!SearchSpaceType::has_scalar_data) {
      throw std::runtime_error("filter_query requires a space that supports scalar data");
    } else {
      auto results = search_space_->get_scalar_data(filter, limit);

      py::list ids_list;
      py::list scalar_list;

      for (const auto &[internal_id, sd] : results) {
        ids_list.append(internal_id);
        scalar_list.append(scalar_data_to_pydict(sd));
      }

      return py::make_tuple(ids_list, scalar_list);
    }
  }

  auto batch_search(py::array_t<DataType> queries, uint32_t topk, uint32_t ef, uint32_t num_threads)
      -> py::array_t<IDType> {
    auto shape = queries.shape();
    size_t query_size = shape[0];
    size_t query_dim = shape[1];

    auto *query_ptr = static_cast<DataType *>(queries.request().ptr);

#if defined(__linux__)
    std::vector<std::vector<IDType>> res_pool(query_size, std::vector<IDType>(topk));

    {
      py::gil_scoped_release release;
      std::vector<CpuID> worker_cpus;
      std::vector<coro::task<>> coros;

      worker_cpus.reserve(num_threads);
      coros.reserve(query_size);

      for (uint32_t i = 0; i < num_threads; i++) {
        worker_cpus.push_back(i);
      }
      auto scheduler = std::make_shared<alaya::Scheduler>(worker_cpus);
      for (uint32_t i = 0; i < query_size; i++) {
        auto cur_query = query_ptr + i * query_dim;

        if constexpr (is_rabitq_space_v<SearchSpaceType>) {
          coros.emplace_back(search_job_->rabitq_search(cur_query, topk, res_pool[i].data(), ef));
        } else {
          // search now handles rerank internally and returns topk results
          coros.emplace_back(search_job_->search(cur_query, res_pool[i].data(), topk, ef));
        }

        scheduler->schedule(coros.back().handle());
      }
      scheduler->begin();
      scheduler->join();
    }

    auto ret = py::array_t<IDType>({query_size, static_cast<size_t>(topk)});
    auto ret_ptr = static_cast<IDType *>(ret.request().ptr);
    for (size_t i = 0; i < query_size; i++) {
      std::copy(res_pool[i].begin(), res_pool[i].end(), ret_ptr + i * topk);
    }
    return ret;
#else
    std::vector<std::vector<IDType>> res_pool(query_size, std::vector<IDType>(topk));

    {
      py::gil_scoped_release release;
      for (size_t i = 0; i < query_size; i++) {
        auto cur_query = query_ptr + i * query_dim;
        if constexpr (is_rabitq_space_v<SearchSpaceType>) {
          search_job_->rabitq_search_solo(cur_query, topk, res_pool[i].data(), ef);
        } else {
          search_job_->search_solo(cur_query, res_pool[i].data(), topk, ef);
        }
      }
    }

    auto ret = py::array_t<IDType>({query_size, static_cast<size_t>(topk)});
    auto ret_ptr = static_cast<IDType *>(ret.request().ptr);
    for (size_t i = 0; i < query_size; i++) {
      std::copy(res_pool[i].begin(), res_pool[i].end(), ret_ptr + i * topk);
    }
    return ret;

#endif
  }

  auto batch_search_with_distance(py::array_t<DataType> queries,
                                  uint32_t topk,
                                  uint32_t ef,
                                  uint32_t num_threads) -> py::object {
    size_t query_size = queries.shape(0);
    size_t query_dim = queries.shape(1);

    auto *query_ptr = static_cast<DataType *>(queries.request().ptr);

#if defined(__linux__)
    // Arrays to store topk results (search now returns topk directly)
    std::vector<std::vector<IDType>> topk_ids(query_size, std::vector<IDType>(topk));
    std::vector<std::vector<DistanceType>> topk_dists(query_size, std::vector<DistanceType>(topk));

    {
      py::gil_scoped_release release;
      std::vector<CpuID> worker_cpus;
      std::vector<coro::task<>> coros;

      worker_cpus.reserve(num_threads);
      coros.reserve(query_size);

      for (uint32_t i = 0; i < num_threads; i++) {
        worker_cpus.push_back(i);
      }
      auto scheduler = std::make_shared<alaya::Scheduler>(worker_cpus);

      for (uint32_t i = 0; i < query_size; i++) {
        auto cur_query = query_ptr + i * query_dim;
        // search now handles rerank internally and returns topk results with distances
        coros.emplace_back(
            search_job_->search(cur_query, topk_ids[i].data(), topk_dists[i].data(), topk, ef));
        scheduler->schedule(coros.back().handle());
      }

      scheduler->begin();
      scheduler->join();
    }

    auto ret_id = get_topk_array(topk_ids, topk);
    auto ret_dist = get_topk_array(topk_dists, topk);
    return py::make_tuple(ret_id, ret_dist);
#else
    std::vector<std::vector<IDType>> topk_ids(query_size, std::vector<IDType>(topk));
    std::vector<std::vector<DistanceType>> topk_dists(query_size, std::vector<DistanceType>(topk));

    {
      py::gil_scoped_release release;
      for (size_t i = 0; i < query_size; i++) {
        auto cur_query = query_ptr + i * query_dim;
        search_job_->search_solo(cur_query, topk_ids[i].data(), topk_dists[i].data(), topk, ef);
      }
    }

    auto ret_id = get_topk_array(topk_ids, topk);
    auto ret_dist = get_topk_array(topk_dists, topk);
    return py::make_tuple(ret_id, ret_dist);
#endif
  }

  /**
   * @brief Close the RocksDB storage explicitly
   */
  auto close_db() -> void {
    if (search_space_ != nullptr) {
      search_space_->close_db();
    }
  }

 private:
  // MetricType metric_{MetricType::L2};
  // uint32_t capacity_{100000};
  DataType *vectors_{nullptr};
  IDType data_size_{0};

  IndexParams params_;
  std::filesystem::path index_path_;

  std::shared_ptr<Graph<DataType, IDType>> graph_index_{nullptr};
  std::shared_ptr<BuildSpaceType> build_space_{nullptr};
  std::shared_ptr<SearchSpaceType> search_space_{nullptr};

  std::shared_ptr<alaya::GraphSearchJob<SearchSpaceType, BuildSpaceType>> search_job_{nullptr};
  std::shared_ptr<alaya::GraphHybridSearchJob<SearchSpaceType, BuildSpaceType>> hybrid_search_job_{
      nullptr};
  std::shared_ptr<alaya::GraphUpdateJob<SearchSpaceType, BuildSpaceType>> update_job_{nullptr};
  std::shared_ptr<JobContext<IDType>> job_context_{nullptr};
  std::mutex hybrid_batch_pool_mutex_;
  std::shared_ptr<alaya::ThreadPool> hybrid_batch_pool_{nullptr};
  uint32_t hybrid_batch_pool_threads_{0};
  std::string materialized_view_field_;
  std::unordered_map<std::string, size_t>
      materialized_view_partition_lookup_;  // partition key to partition index
  std::vector<MaterializedViewPartition>
      materialized_view_partitions_;  // index partitions for materialized view
  uint32_t materialized_view_ef_construction_{200};
  uint32_t materialized_view_build_threads_{1};
  bool materialized_view_ready_{false};
};

class PyIndexInterface {
 public:
  explicit PyIndexInterface(const IndexParams &params) : params_(params) {  // NOLINT
    DISPATCH_AND_CREATE(params);
  }

  auto to_string() -> std::string { return "PyIndexInterface"; }

  auto fit(py::array &vectors,  // NOLINT
           uint32_t ef_construction,
           uint32_t num_threads,
           const py::object &item_ids = py::none(),
           const py::object &documents = py::none(),
           const py::object &metadata_list = py::none()) -> void {
    DISPATCH_AND_CAST_WITH_ARR(vectors,
                               typed_vectors,
                               index,
                               index->fit(typed_vectors,
                                          ef_construction,
                                          num_threads,
                                          item_ids,
                                          documents,
                                          metadata_list););
  }

  auto search(py::array &query, uint32_t topk, uint32_t ef) -> py::array {  // NOLINT
    DISPATCH_AND_CAST_WITH_ARR(query,
                               typed_query,
                               index,
                               return index->search(typed_query, topk, ef););
  }

  auto get_data_by_id(uint32_t id) -> py::array {  // NOLINT
    DISPATCH_AND_CAST(index, return index->get_data_by_id(id););
  }

  auto insert(py::array &insert_data,
              uint32_t ef,
              const py::object &item_id_obj = py::none(),
              const std::string &document = "",
              const py::dict &metadata = py::dict())
      -> std::variant<uint32_t, uint64_t> {  // NOLINT
    // Convert item_id to string using Python's str() for any type
    std::string item_id = item_id_obj.is_none() ? "" : py::str(item_id_obj).cast<std::string>();
    DISPATCH_AND_CAST_WITH_ARR(insert_data,
                               typed_insert_data,
                               index,
                               return index
                                   ->insert(typed_insert_data, ef, item_id, document, metadata););
  }

  auto remove(uint32_t id) -> void {  // NOLINT
    DISPATCH_AND_CAST(index, index->remove(id););
  }

  auto remove_by_item_id(const py::object &item_id_obj) -> void {  // NOLINT
    std::string item_id = py::str(item_id_obj).cast<std::string>();
    DISPATCH_AND_CAST(index, index->remove(item_id););
  }

  auto get_scalar_data_by_item_id(const py::object &item_id_obj) -> py::dict {  // NOLINT
    std::string item_id = py::str(item_id_obj).cast<std::string>();
    DISPATCH_AND_CAST(index, return index->get_scalar_data_by_item_id(item_id););
  }

  auto contains(const py::object &item_id_obj) -> bool {  // NOLINT
    std::string item_id = py::str(item_id_obj).cast<std::string>();
    DISPATCH_AND_CAST(index, return index->contains(item_id););
  }

  auto get_scalar_data_by_internal_id(uint32_t internal_id) -> py::dict {  // NOLINT
    DISPATCH_AND_CAST(index, return index->get_scalar_data_by_internal_id(internal_id););
  }

  auto batch_get_item_ids_by_internal_ids(py::array internal_ids) -> py::list {  // NOLINT
    DISPATCH_AND_CAST(index, {
      auto typed_ids = internal_ids.cast<py::array_t<uint32_t>>();
      return index->batch_get_item_ids_by_internal_ids(typed_ids);
    });
  }

  auto filter_query(const MetadataFilter &filter, uint32_t limit) -> py::object {  // NOLINT
    DISPATCH_AND_CAST(index, return index->filter_query(filter, limit););
  }

  auto get_data_num() -> std::variant<uint32_t, uint64_t> {  // NOLINT
    DISPATCH_AND_CAST(index, return index->get_data_num(););
  }

  auto get_materialized_view_partition_count() -> uint32_t {  // NOLINT
    DISPATCH_AND_CAST(index, return index->get_materialized_view_partition_count(););
  }

  auto batch_search(py::array &queries,
                    uint32_t topk,
                    uint32_t ef,  // NOLINT
                    uint32_t num_threads) -> py::array {
    DISPATCH_AND_CAST_WITH_ARR(queries,
                               typed_queries,
                               index,
                               return index->batch_search(typed_queries, topk, ef, num_threads););
  }

  auto batch_search_with_distance(py::array &queries,
                                  uint32_t topk,
                                  uint32_t ef,  // NOLINT
                                  uint32_t num_threads) -> py::object {
    DISPATCH_AND_CAST_WITH_ARR(queries,
                               typed_queries,
                               index,
                               return index->batch_search_with_distance(typed_queries,
                                                                        topk,
                                                                        ef,
                                                                        num_threads););
  }

  auto load(const std::string &index_path,  // NOLINT
            const std::string &data_path = std::string(),
            const std::string &quant_path = std::string()) -> void {
    DISPATCH_AND_CAST(index, index->load(index_path, data_path, quant_path););
  }

  auto save(const std::string &index_path,  // NOLINT
            const std::string &data_path = std::string(),
            const std::string &quant_path = std::string()) -> void {
    DISPATCH_AND_CAST(index, index->save(index_path, data_path, quant_path););
  }

  auto get_data_dim() -> uint32_t { return index_->data_dim_; }

  auto hybrid_search(py::array &query,
                     uint32_t topk,
                     uint32_t ef,
                     const MetadataFilter &filter,
                     bool bf = false,
                     const std::string &filter_exec_hint = std::string()) -> py::object {
    DISPATCH_AND_CAST_WITH_ARR(query,
                               typed_query,
                               index,
                               return index->hybrid_search(typed_query,
                                                           topk,
                                                           ef,
                                                           filter,
                                                           bf,
                                                           filter_exec_hint););
  }

  auto batch_hybrid_search(py::array &queries,
                           uint32_t topk,
                           uint32_t ef,
                           const MetadataFilter &filter,
                           uint32_t num_threads,
                           bool bf = false,
                           const std::string &filter_exec_hint = std::string()) -> py::object {
    DISPATCH_AND_CAST_WITH_ARR(queries,
                               typed_queries,
                               index,
                               return index->batch_hybrid_search(typed_queries,
                                                                 topk,
                                                                 ef,
                                                                 filter,
                                                                 num_threads,
                                                                 bf,
                                                                 filter_exec_hint););
  }

  auto close_db() -> void {  // NOLINT
    DISPATCH_AND_CAST(index, index->close_db(););
  }

  auto has_scalar_data() const -> bool { return params_.has_scalar_data_; }

  virtual ~PyIndexInterface() = default;
  IndexParams params_;
  std::shared_ptr<BasePyIndex> index_;
};
}  // namespace alaya
