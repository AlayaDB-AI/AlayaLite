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

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "executor/jobs/graph_search_job.hpp"
#include "executor/search_info.hpp"
#include "executor/vector_iterator.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "index/graph/qg/qg_builder.hpp"
#include "params.hpp"
#include "space/rabitq_space.hpp"
#include "space/raw_space.hpp"
#include "space/sq4_space.hpp"
#include "space/sq8_space.hpp"
#include "storage/rocksdb_storage.hpp"
#include "utils/index_encoding.hpp"
#include "utils/log.hpp"
#include "utils/memory.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/metadata_filter_matcher.hpp"
#include "utils/scalar_data.hpp"

namespace alaya {

// when scalar storage is unnecessary, strip scalar payloads all the way through search space.
template <typename SpaceType>
struct StripScalarData;

template <typename DataType,
          typename DistanceType,
          typename IDType,
          typename DataStorage,
          typename ScalarDataType>
struct StripScalarData<RawSpace<DataType, DistanceType, IDType, DataStorage, ScalarDataType>> {
  using type = RawSpace<DataType, DistanceType, IDType, DataStorage, EmptyScalarData>;
};

template <typename DataType,
          typename DistanceType,
          typename IDType,
          typename DataStorage,
          typename ScalarDataType>
struct StripScalarData<SQ4Space<DataType, DistanceType, IDType, DataStorage, ScalarDataType>> {
  using type = SQ4Space<DataType, DistanceType, IDType, DataStorage, EmptyScalarData>;
};

template <typename DataType,
          typename DistanceType,
          typename IDType,
          typename DataStorage,
          typename ScalarDataType>
struct StripScalarData<SQ8Space<DataType, DistanceType, IDType, DataStorage, ScalarDataType>> {
  using type = SQ8Space<DataType, DistanceType, IDType, DataStorage, EmptyScalarData>;
};

template <typename DataType, typename DistanceType, typename IDType, typename ScalarDataType>
struct StripScalarData<RaBitQSpace<DataType, DistanceType, IDType, ScalarDataType>> {
  using type = RaBitQSpace<DataType, DistanceType, IDType, EmptyScalarData>;
};

template <typename SpaceType>
using StripScalarDataT = typename StripScalarData<SpaceType>::type;

template <typename RawSpaceType>
class RawSubsetSpaceView;

template <typename RawSpaceType>
struct IsRawSubsetSpaceView : std::false_type {};

template <typename RawSpaceType>
struct IsRawSubsetSpaceView<RawSubsetSpaceView<RawSpaceType>> : std::true_type {};

template <typename RawSpaceType>
inline constexpr bool is_raw_subset_space_v = IsRawSubsetSpaceView<RawSpaceType>::value;

template <typename DataType,
          typename DistanceType,
          typename IDType,
          typename DataStorage,
          typename ScalarDataType>
class RawSubsetSpaceView<RawSpace<DataType, DistanceType, IDType, DataStorage, ScalarDataType>> {
 public:
  using BaseSpaceType = RawSpace<DataType, DistanceType, IDType, DataStorage, ScalarDataType>;
  using DataTypeAlias = DataType;
  using IDTypeAlias = IDType;
  using DistanceTypeAlias = DistanceType;
  using DistDataType = DataType;

  static constexpr bool has_scalar_data = false;

  RawSubsetSpaceView() = delete;

  RawSubsetSpaceView(std::shared_ptr<BaseSpaceType> base_space,
                     std::vector<IDType> local_to_global_ids)
      : base_space_(std::move(base_space)), local_to_global_ids_(std::move(local_to_global_ids)) {
    if (base_space_ == nullptr) {
      throw std::invalid_argument("RawSubsetSpaceView requires a backing RawSpace");
    }
  }

  void fit(const DataType *, IDType, const EmptyScalarData * = nullptr) {
    throw std::logic_error("RawSubsetSpaceView does not own vector storage");
  }

  void set_metric_function() { base_space_->set_metric_function(); }

  auto get_dist_func() const { return base_space_->get_dist_func(); }

  auto get_capacity() const -> IDType { return static_cast<IDType>(local_to_global_ids_.size()); }

  auto get_data_num() const -> IDType { return static_cast<IDType>(local_to_global_ids_.size()); }

  auto get_data_size() const -> size_t { return base_space_->get_data_size(); }

  auto get_dim() const -> uint32_t { return base_space_->get_dim(); }

  auto get_distance(IDType lhs, IDType rhs) const -> DistanceType {
    return base_space_->get_distance(map_id(lhs), map_id(rhs));
  }

  auto get_data_by_id(IDType local_id) const -> DataType * {
    return base_space_->get_data_by_id(map_id(local_id));
  }

  auto prefetch_by_id(IDType local_id) -> void { base_space_->prefetch_by_id(map_id(local_id)); }

  auto prefetch_by_address(DataType *address) -> void { base_space_->prefetch_by_address(address); }

  struct QueryComputer {
    const RawSubsetSpaceView &distance_space_;
    std::vector<DataType, AlignedAlloc<DataType>> query_;

    QueryComputer(const RawSubsetSpaceView &distance_space, const DataType *query)
        : distance_space_(distance_space), query_(distance_space.get_dim()) {
      std::memcpy(query_.data(), query, distance_space.get_data_size());
    }

    QueryComputer(const RawSubsetSpaceView &distance_space, const IDType id)
        : distance_space_(distance_space), query_(distance_space.get_dim()) {
      std::memcpy(query_.data(), distance_space.get_data_by_id(id), distance_space.get_data_size());
    }

    auto operator()(IDType local_id) const -> DistanceType {
      return distance_space_.get_dist_func()(query_.data(),
                                             distance_space_.get_data_by_id(local_id),
                                             distance_space_.get_dim());
    }
  };

  auto get_query_computer(const DataType *query) const { return QueryComputer(*this, query); }

  auto get_query_computer(const IDType id) const { return QueryComputer(*this, id); }

 private:
  auto map_id(IDType local_id) const -> IDType {
    return local_to_global_ids_.at(static_cast<size_t>(local_id));
  }

  std::shared_ptr<BaseSpaceType> base_space_;
  std::vector<IDType> local_to_global_ids_;
};

template <typename SpaceType>
struct MaterializedViewSpace {
  using type = StripScalarDataT<SpaceType>;
};

template <typename DataType,
          typename DistanceType,
          typename IDType,
          typename DataStorage,
          typename ScalarDataType>
struct MaterializedViewSpace<
    RawSpace<DataType, DistanceType, IDType, DataStorage, ScalarDataType>> {
  using type =
      RawSubsetSpaceView<RawSpace<DataType, DistanceType, IDType, DataStorage, ScalarDataType>>;
};

template <typename SpaceType>
using MaterializedViewSpaceT = typename MaterializedViewSpace<SpaceType>::type;

struct MaterializedViewPartitionSelection {
  bool eligible_ = false;
  bool filter_covered_ = false;
  std::vector<MetadataValue> values_;
};

inline void append_unique_metadata_value(std::vector<MetadataValue> &values,
                                         const MetadataValue &value) {
  if (std::find(values.begin(), values.end(), value) == values.end()) {
    values.push_back(value);
  }
}

inline auto intersect_metadata_values(const std::vector<MetadataValue> &lhs,
                                      const std::vector<MetadataValue> &rhs)
    -> std::vector<MetadataValue> {
  std::vector<MetadataValue> intersection;
  for (const auto &value : lhs) {
    if (std::find(rhs.begin(), rhs.end(), value) != rhs.end()) {
      append_unique_metadata_value(intersection, value);
    }
  }
  return intersection;
}

inline auto collect_conjunctive_filter_conditions(const MetadataFilter &filter,
                                                  std::vector<const FilterCondition *> &conditions)
    -> bool {
  if (filter.is_empty()) {
    return true;
  }
  if (filter.logic_op != LogicOp::AND) {
    return false;
  }

  for (const auto &condition : filter.conditions) {
    conditions.push_back(&condition);
  }
  for (const auto &sub_filter : filter.sub_filters) {
    if (sub_filter == nullptr || !collect_conjunctive_filter_conditions(*sub_filter, conditions)) {
      return false;
    }
  }
  return true;
}

inline auto analyze_materialized_view_filter(const MetadataFilter &filter,
                                             const std::string &target_field)
    -> MaterializedViewPartitionSelection {
  if (filter.is_empty() || target_field.empty()) {
    return {};
  }

  std::vector<const FilterCondition *> conditions;
  if (!collect_conjunctive_filter_conditions(filter, conditions)) {
    return {};
  }

  std::optional<std::vector<MetadataValue>> allowed_values;
  bool filter_covered = true;
  for (const auto *condition : conditions) {
    if (condition->field != target_field) {
      filter_covered = false;
      continue;
    }

    std::vector<MetadataValue> condition_values;
    switch (condition->op) {
      case FilterOp::EQ:
        append_unique_metadata_value(condition_values, condition->value);
        break;
      case FilterOp::IN:
        for (const auto &value : condition->values) {
          append_unique_metadata_value(condition_values, value);
        }
        break;
      default:
        return {};
    }

    if (!allowed_values.has_value()) {
      allowed_values = std::move(condition_values);
    } else {
      allowed_values = intersect_metadata_values(*allowed_values, condition_values);
    }
  }

  if (!allowed_values.has_value()) {
    return {};
  }

  return MaterializedViewPartitionSelection{true, filter_covered, std::move(*allowed_values)};
}

template <typename SearchSpaceType, typename BuildSpaceType>
class MaterializedViewManager {
 public:
  using IDType = typename SearchSpaceType::IDTypeAlias;
  using DataType = typename SearchSpaceType::DataTypeAlias;
  using DistanceType = typename SearchSpaceType::DistanceTypeAlias;
  using SearchSpacePtr = std::shared_ptr<SearchSpaceType>;
  using BuildSpacePtr = std::shared_ptr<BuildSpaceType>;
  using MaterializedViewSearchSpaceType = MaterializedViewSpaceT<SearchSpaceType>;
  using MaterializedViewBuildSpaceType = MaterializedViewSpaceT<BuildSpaceType>;
  using MaterializedViewCandidate = VectorCandidate<IDType, DistanceType>;

  struct Partition {
    MetadataValue value_;
    std::string encoded_value_;
    std::vector<IDType> local_to_global_ids_;
    std::shared_ptr<MaterializedViewSearchSpaceType> search_space_{nullptr};
    std::shared_ptr<MaterializedViewBuildSpaceType> build_space_{nullptr};
    std::shared_ptr<Graph<DataType, IDType>> graph_{nullptr};
    std::shared_ptr<GraphSearchJob<MaterializedViewSearchSpaceType, MaterializedViewBuildSpaceType>>
        search_job_{nullptr};
  };

  void rebuild(const IndexParams &params,
               uint32_t data_dim,
               const SearchSpacePtr &search_space,
               const BuildSpacePtr &build_space,
               uint32_t ef_construction,
               uint32_t build_threads) {
    bind_sources(params, data_dim, search_space, build_space, ef_construction, build_threads);
    reset();
    if (!is_supported()) {
      return;
    }

    auto *storage = search_space_->get_scalar_storage();
    if (storage == nullptr) {
      return;
    }

    field_ = params_->indexed_fields_.front();
    if (params_->indexed_fields_.size() > 1) {
      LOG_INFO("materialized_view: only the first indexed field is partitioned, field={}", field_);
    }

    try {
      auto partitions = collect_partition_seeds(*storage);
      if (!partitions.has_value()) {
        LOG_INFO("materialized_view: skip build, field={} has too many partitions (>{})",
                 field_,
                 kMaxPartitions);
        reset();
        return;
      }

      if (partitions->size() <= 1) {
        LOG_INFO("materialized_view: skip build, field={} has {} partition(s)",
                 field_,
                 partitions->size());
        reset();
        return;
      }

      partition_lookup_.reserve(partitions->size());
      partitions_.reserve(partitions->size());
      for (auto &partition_seed : *partitions) {
        auto partition = build_partition(partition_seed.value_,
                                         std::move(partition_seed.encoded_value_),
                                         std::move(partition_seed.global_ids_));
        partition_lookup_.emplace(partition.encoded_value_, partitions_.size());
        partitions_.push_back(std::move(partition));
      }

      ready_ = !partitions_.empty();
      LOG_INFO("materialized_view: built field={}, partitions={}", field_, partitions_.size());
    } catch (const std::exception &e) {
      LOG_ERROR("materialized_view: build failed for field={}, error={}", field_, e.what());
      reset();
    }
  }

  void invalidate(std::string_view reason) {
    if (ready_) {
      LOG_DEBUG("materialized_view: invalidate cached partitions, reason={}", reason);
    }
    reset();
  }

  [[nodiscard]] auto get_partition_count() const -> uint32_t {
    return static_cast<uint32_t>(partitions_.size());
  }

  auto try_hybrid_search(const DataType *query,
                         IDType *ids,
                         const SearchInfo &search_info,
                         const MetadataFilter &filter,
                         bool brute_force_requested,
                         std::string *item_ids) const -> bool {
    if (!ready_ || filter.is_empty() || search_space_ == nullptr) {
      return false;
    }

    auto partition_selection = analyze_materialized_view_filter(filter, field_);
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
      auto lookup_it = partition_lookup_.find(index_encoding::encode_value(value));
      if (lookup_it == partition_lookup_.end()) {
        continue;
      }

      ++selected_partitions;
      auto partition_results = execute_partition_search(partitions_[lookup_it->second],
                                                        query,
                                                        search_info,
                                                        filter_executor.get(),
                                                        brute_force_requested,
                                                        partition_selection.filter_covered_);
      for (const auto &candidate : partition_results) {
        insert_candidate(merged_results, candidate.id_, candidate.distance_, search_info.topk_);
      }
    }

    LOG_DEBUG("hybrid_search: plan=materialized_view, field={}, partitions={}",
              field_,
              selected_partitions);

    if (merged_results.empty()) {
      return true;
    }

    sort_candidates(merged_results);

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

 private:
  static constexpr size_t kMaxPartitions =
      128;  // avoid too many tiny partitions that would dominate search overhead
  static constexpr float kKnnBFFilterThreshold =
      0.93F;  // if the filter excludes most of a partition, brute force is usually cheaper
  static constexpr float kBFTopkThreshold =
      0.5F;  // if topk covers too much of the partition, brute force is usually cheaper
  static constexpr size_t kRaBitQExactPartitionThreshold =
      128;  // small RaBitQ partitions are cheaper to scan exactly than to re-index

  struct PartitionSeed {
    MetadataValue value_;
    std::string encoded_value_;
    std::vector<IDType> global_ids_;
  };

  static auto count_result_ids(const IDType *ids, uint32_t topk) -> uint32_t {
    uint32_t count = 0;
    while (count < topk && ids[count] != std::numeric_limits<IDType>::max()) {
      ++count;
    }
    return count;
  }

  static auto candidate_less(const MaterializedViewCandidate &lhs,
                             const MaterializedViewCandidate &rhs) -> bool {
    if (lhs.distance_ != rhs.distance_) {
      return lhs.distance_ < rhs.distance_;
    }
    return lhs.id_ < rhs.id_;
  }

  static void insert_candidate(std::vector<MaterializedViewCandidate> &results,
                               IDType id,
                               DistanceType distance,
                               size_t limit) {
    if (limit == 0) {
      return;
    }

    MaterializedViewCandidate candidate{id, distance};
    if (results.size() < limit) {
      results.push_back(candidate);
      std::push_heap(results.begin(), results.end(), candidate_less);
      return;
    }

    if (!candidate_less(candidate, results.front())) {
      return;
    }

    std::pop_heap(results.begin(), results.end(), candidate_less);
    results.back() = candidate;
    std::push_heap(results.begin(), results.end(), candidate_less);
  }

  static void sort_candidates(std::vector<MaterializedViewCandidate> &results) {
    std::sort(results.begin(), results.end(), candidate_less);
  }

  static auto should_use_brute_force(const SearchInfo &search_info,
                                     size_t total_count,
                                     size_t matched_count) -> bool {
    if (matched_count == 0) {
      return false;
    }

    auto topk = static_cast<size_t>(search_info.topk_);
    if (topk >= static_cast<size_t>(static_cast<double>(total_count) * kBFTopkThreshold)) {
      return true;
    }

    auto filtered_out_num = total_count - matched_count;
    if (filtered_out_num >=
        static_cast<size_t>(static_cast<double>(total_count) * kKnnBFFilterThreshold)) {
      return true;
    }

    return topk >= static_cast<size_t>(static_cast<double>(matched_count) * kBFTopkThreshold);
  }

  static auto adjust_search_info(const SearchInfo &search_info,
                                 size_t partition_size,
                                 size_t matched_count) -> SearchInfo {
    SearchInfo adjusted = search_info;
    adjusted.topk_ = static_cast<uint32_t>(std::min<size_t>(search_info.topk_, partition_size));
    adjusted.ef_ = static_cast<uint32_t>(
        std::min<size_t>(partition_size, std::max<uint32_t>(search_info.ef_, adjusted.topk_)));

    if (matched_count == 0 || matched_count == partition_size) {
      return adjusted;
    }

    auto expected_ef = static_cast<size_t>(
        (static_cast<double>(adjusted.topk_) * static_cast<double>(partition_size)) /
        static_cast<double>(matched_count));
    expected_ef += expected_ef / 2;

    adjusted.ef_ = static_cast<uint32_t>(
        std::min<size_t>(partition_size, std::max<size_t>(adjusted.ef_, expected_ef)));
    return adjusted;
  }

  void bind_sources(const IndexParams &params,
                    uint32_t data_dim,
                    const SearchSpacePtr &search_space,
                    const BuildSpacePtr &build_space,
                    uint32_t ef_construction,
                    uint32_t build_threads) {
    params_ = &params;
    data_dim_ = data_dim;
    search_space_ = search_space;
    build_space_ = build_space;
    ef_construction_ = ef_construction;
    build_threads_ = std::max<uint32_t>(1, build_threads);
  }

  void reset() {
    ready_ = false;
    field_.clear();
    partition_lookup_.clear();
    partitions_.clear();
  }

  [[nodiscard]] auto is_supported() const -> bool {
    if (params_ == nullptr) {
      return false;
    }

    if constexpr (!SearchSpaceType::has_scalar_data) {
      return false;
    }

    if (search_space_ == nullptr || params_->indexed_fields_.empty()) {
      return false;
    }

    if constexpr (!is_rabitq_space_v<SearchSpaceType>) {
      if (build_space_ == nullptr || params_->index_type_ != IndexType::HNSW) {
        return false;
      }
    }

    return true;
  }

  void copy_source_vector(IDType global_id, DataType *dst) const {
    if constexpr (is_rabitq_space_v<SearchSpaceType>) {
      auto *src = search_space_->get_data_by_id(global_id);
      std::memcpy(dst, src, static_cast<size_t>(data_dim_) * sizeof(DataType));
    } else {
      auto *src = build_space_->get_data_by_id(global_id);
      std::memcpy(dst, src, static_cast<size_t>(data_dim_) * sizeof(DataType));
    }
  }

  auto collect_partition_seeds(const RocksDBStorage<IDType> &storage) const
      -> std::optional<std::vector<PartitionSeed>> {
    std::vector<PartitionSeed> partitions;
    std::unordered_map<std::string, size_t> partition_lookup;

    // todo: build materialized views asynchronously or on demand for large datasets.
    for (IDType id = 0; id < search_space_->get_data_num(); ++id) {
      std::string raw_value;
      if (!storage.get_raw_value(id, raw_value)) {
        continue;
      }

      auto field_value =
          ScalarData::deserialize_single_metadata_value(raw_value.data(), raw_value.size(), field_);
      if (!field_value.has_value()) {
        // todo: missing partition field may deserve an explicit error instead of being skipped.
        continue;
      }

      auto encoded_value = index_encoding::encode_value(*field_value);
      auto lookup_it = partition_lookup.find(encoded_value);
      if (lookup_it == partition_lookup.end()) {
        lookup_it = partition_lookup.emplace(encoded_value, partitions.size()).first;
        partitions.push_back(PartitionSeed{*field_value, encoded_value, {}});
        if (partitions.size() > kMaxPartitions) {
          return std::nullopt;
        }
      }
      partitions[lookup_it->second].global_ids_.push_back(id);
    }

    return partitions;
  }

  template <typename ExactDistanceEvaluator>
  void emit_brute_force_results(const Partition &partition,
                                const SearchInfo &search_info,
                                const DynamicBitset *residual_blocked,
                                ExactDistanceEvaluator &&exact_distance,
                                std::vector<MaterializedViewCandidate> &results) const {
    auto partition_size = partition.local_to_global_ids_.size();
    for (size_t local_id = 0; local_id < partition_size; ++local_id) {
      if (residual_blocked != nullptr && residual_blocked->get(local_id)) {
        continue;
      }
      insert_candidate(results,
                       partition.local_to_global_ids_[local_id],
                       exact_distance(static_cast<IDType>(local_id)),
                       search_info.topk_);
    }
  }

  template <typename ExactDistanceEvaluator>
  void search_partition_with_exact_distance(const Partition &partition,
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

    if (brute_force_requested || partition.search_job_ == nullptr ||
        should_use_brute_force(local_search_info, partition_size, matched_count)) {
      emit_brute_force_results(partition,
                               local_search_info,
                               residual_blocked,
                               std::forward<ExactDistanceEvaluator>(exact_distance),
                               results);
      return;
    }

    auto required_results =
        static_cast<uint32_t>(std::min<size_t>(local_search_info.topk_, matched_count));
    if (required_results == 0) {
      return;
    }

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
        emit_brute_force_results(partition,
                                 local_search_info,
                                 residual_blocked,
                                 std::forward<ExactDistanceEvaluator>(exact_distance),
                                 results);
        return;
      }

      for (uint32_t i = 0; i < found_results; ++i) {
        auto local_id = local_ids[i];
        insert_candidate(results,
                         partition.local_to_global_ids_[local_id],
                         exact_distance(local_id),
                         local_search_info.topk_);
      }
      return;
    }

    if (search_info.filter_exec_hint_ == FilterExecHint::kIterativeFilter) {
      auto iterator = partition.search_job_->make_vector_iterator(query, local_search_info);
      while (results.size() < local_search_info.topk_ && iterator->has_next()) {
        auto candidate = iterator->next();
        if (!candidate.has_value()) {
          break;
        }
        if (residual_blocked != nullptr && residual_blocked->get(candidate->id_)) {
          continue;
        }
        insert_candidate(results,
                         partition.local_to_global_ids_[candidate->id_],
                         exact_distance(candidate->id_),
                         local_search_info.topk_);
      }
      return;
    }

    auto adjusted_search_info =
        adjust_search_info(local_search_info, partition_size, matched_count);
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

    auto found_results = count_result_ids(local_ids.data(), adjusted_search_info.topk_);
    if (found_results < required_results) {
      emit_brute_force_results(partition,
                               local_search_info,
                               residual_blocked,
                               std::forward<ExactDistanceEvaluator>(exact_distance),
                               results);
      return;
    }

    for (uint32_t i = 0; i < found_results; ++i) {
      auto local_id = local_ids[i];
      insert_candidate(results,
                       partition.local_to_global_ids_[local_id],
                       exact_distance(local_id),
                       local_search_info.topk_);
    }
  }

  auto build_partition(const MetadataValue &value,
                       std::string encoded_value,
                       std::vector<IDType> global_ids) const -> Partition {
    Partition partition;
    partition.value_ = value;
    partition.encoded_value_ = std::move(encoded_value);
    partition.local_to_global_ids_ = std::move(global_ids);

    auto partition_size = partition.local_to_global_ids_.size();
    auto partition_capacity = static_cast<IDType>(partition_size);

    if constexpr (is_raw_subset_space_v<MaterializedViewBuildSpaceType>) {
      partition.build_space_ =
          std::make_shared<MaterializedViewBuildSpaceType>(build_space_,
                                                           partition.local_to_global_ids_);
    } else {
      partition.build_space_ = std::make_shared<MaterializedViewBuildSpaceType>(partition_capacity,
                                                                                data_dim_,
                                                                                params_->metric_);
    }

    if constexpr (std::is_same_v<MaterializedViewBuildSpaceType, MaterializedViewSearchSpaceType>) {
      partition.search_space_ = partition.build_space_;
    } else if constexpr (is_raw_subset_space_v<MaterializedViewSearchSpaceType>) {
      partition.search_space_ =
          std::make_shared<MaterializedViewSearchSpaceType>(search_space_,
                                                            partition.local_to_global_ids_);
    } else {
      partition.search_space_ =
          std::make_shared<MaterializedViewSearchSpaceType>(partition_capacity,
                                                            data_dim_,
                                                            params_->metric_);
    }

    constexpr bool kBuildSpaceNeedsFit = !is_raw_subset_space_v<MaterializedViewBuildSpaceType>;
    constexpr bool kSearchSpaceNeedsFit = !is_raw_subset_space_v<MaterializedViewSearchSpaceType>;
    if constexpr (kBuildSpaceNeedsFit || (!std::is_same_v<MaterializedViewBuildSpaceType,
                                                          MaterializedViewSearchSpaceType> &&
                                          kSearchSpaceNeedsFit)) {
      std::vector<DataType> partition_vectors(partition_size * static_cast<size_t>(data_dim_));
      for (size_t i = 0; i < partition_size; ++i) {
        copy_source_vector(partition.local_to_global_ids_[i],
                           partition_vectors.data() + (i * static_cast<size_t>(data_dim_)));
      }

      if constexpr (kBuildSpaceNeedsFit) {
        partition.build_space_->fit(partition_vectors.data(), partition_capacity);
      }
      if constexpr (!std::is_same_v<MaterializedViewBuildSpaceType,
                                    MaterializedViewSearchSpaceType> &&
                    kSearchSpaceNeedsFit) {
        partition.search_space_->fit(partition_vectors.data(), partition_capacity);
      }
    }

    // todo: small partitions may not need a child index; choose a threshold with benchmarks.
    if constexpr (is_rabitq_space_v<MaterializedViewSearchSpaceType>) {
      if (partition_size > kRaBitQExactPartitionThreshold) {
        QGBuilder<MaterializedViewSearchSpaceType> graph_builder(partition.search_space_);
        graph_builder.build_graph();
        partition.search_job_ = std::make_shared<
            GraphSearchJob<MaterializedViewSearchSpaceType,
                           MaterializedViewBuildSpaceType>>(partition.search_space_,
                                                            nullptr,
                                                            nullptr,
                                                            partition.build_space_);
      }
    } else {
      if (partition_size == 1) {
        partition.graph_ =
            std::make_shared<Graph<DataType, IDType>>(partition_capacity, params_->max_nbrs_);
        partition.graph_->eps_.push_back(0);
      } else {
        HNSWBuilder<MaterializedViewBuildSpaceType> graph_builder(partition.build_space_,
                                                                  params_->max_nbrs_,
                                                                  ef_construction_);
        partition.graph_ = std::shared_ptr<Graph<DataType, IDType>>(
            graph_builder.build_graph(build_threads_).release());
      }
      partition.search_job_ =
          std::make_shared<GraphSearchJob<MaterializedViewSearchSpaceType,
                                          MaterializedViewBuildSpaceType>>(partition.search_space_,
                                                                           partition.graph_,
                                                                           nullptr,
                                                                           partition.build_space_);
    }

    return partition;
  }

  auto execute_partition_search(const Partition &partition,
                                const DataType *query,
                                const SearchInfo &search_info,
                                const MetadataFilterExecutor<IDType> *filter_executor,
                                bool brute_force_requested,
                                bool filter_covered) const
      -> std::vector<MaterializedViewCandidate> {
    std::vector<MaterializedViewCandidate> results;
    auto partition_size = partition.local_to_global_ids_.size();
    if (partition_size == 0) {
      return results;
    }

    SearchInfo local_search_info = adjust_search_info(search_info, partition_size, partition_size);

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
    search_partition_with_exact_distance(
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
  }

  const IndexParams *params_ = nullptr;
  uint32_t data_dim_ = 0;
  SearchSpacePtr search_space_{nullptr};
  BuildSpacePtr build_space_{nullptr};
  std::string field_;
  std::unordered_map<std::string, size_t> partition_lookup_;
  std::vector<Partition> partitions_;
  uint32_t ef_construction_{200};
  uint32_t build_threads_{1};
  bool ready_{false};
};

}  // namespace alaya
