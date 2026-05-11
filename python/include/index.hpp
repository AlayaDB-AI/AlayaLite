// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

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
#include "recovery/recovery_manager.hpp"
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
  using MaterializedViewManagerType = MaterializedViewManager<SearchSpaceType, BuildSpaceType>;

  PyIndex() = delete;
  explicit PyIndex(IndexParams params) : params_(std::move(params)) { initialize_recovery(); }

  auto to_string() const -> std::string { return "PyIndex"; }
  auto get_materialized_view_partition_count() const -> uint32_t {
    return materialized_view_manager_.get_partition_count();
  }

 private:
  void execute_hybrid_search_dispatch(const DataType *query,
                                      IDType *ids,
                                      const SearchInfo &search_info,
                                      const MetadataFilter &filter,
                                      bool brute_force_requested,
                                      std::string *item_ids) const {
    if (materialized_view_manager_
            .try_hybrid_search(query, ids, search_info, filter, brute_force_requested, item_ids)) {
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

  void initialize_recovery() {
    if (params_.rocksdb_path_.empty()) {
      return;
    }
    auto recovery_root = std::filesystem::path(params_.rocksdb_path_).parent_path() / "recovery";
    recovery_manager_ =
        std::make_unique<alaya::recovery::RecoveryManager>(recovery_root,
                                                           std::filesystem::path(
                                                               params_.rocksdb_path_));

    uint64_t max_seen_op_id = 0;
    recovery_manager_->replayable_records(0, &max_seen_op_id);
    auto manifest = recovery_manager_->current_snapshot();
    if (manifest.has_value()) {
      last_committed_recovery_op_id_ = manifest->applied_through_op_id;
    }
    last_seen_recovery_op_id_ = std::max(max_seen_op_id, last_committed_recovery_op_id_);
    next_recovery_op_id_ = last_seen_recovery_op_id_ + 1;
  }

  [[nodiscard]] auto recovery_scalar_storage() const -> RocksDBStorage<IDType> * {
    if constexpr (SearchSpaceType::has_scalar_data) {
      if (search_space_ != nullptr) {
        return search_space_->get_scalar_storage();
      }
    }
    if constexpr (!std::is_same<BuildSpaceType, SearchSpaceType>::value &&
                  BuildSpaceType::has_scalar_data) {
      if (build_space_ != nullptr) {
        return build_space_->get_scalar_storage();
      }
    }
    return nullptr;
  }

  auto save_state(const std::string &index_path,
                  const std::string &data_path = std::string(),
                  const std::string &quant_path = std::string()) const -> void {
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

  auto load_state(const std::string &index_path,
                  const std::string &data_path = std::string(),
                  const std::string &quant_path = std::string()) -> void {
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
  }

  auto checkpoint_recovery_snapshot(std::string_view reason) -> void {
    if (recovery_manager_ == nullptr || search_space_ == nullptr) {
      return;
    }

    auto snapshot_dir = recovery_manager_->create_snapshot_dir();
    auto snapshot_id = snapshot_dir.filename().string();

    std::string graph_file;
    std::string data_file;
    std::string quant_file;

    if constexpr (!is_rabitq_space_v<SearchSpaceType>) {
      graph_file = "graph.snapshot";
      data_file = "data.snapshot";
    }
    if constexpr (is_rabitq_space_v<SearchSpaceType> ||
                  !std::is_same<BuildSpaceType, SearchSpaceType>::value) {
      quant_file = "quant.snapshot";
    }

    save_state(graph_file.empty() ? std::string() : (snapshot_dir / graph_file).string(),
               data_file.empty() ? std::string() : (snapshot_dir / data_file).string(),
               quant_file.empty() ? std::string() : (snapshot_dir / quant_file).string());

    std::string rocksdb_dir;
    if (auto *storage = recovery_scalar_storage(); storage != nullptr) {
      rocksdb_dir = "rocksdb";
      storage->save((snapshot_dir / rocksdb_dir).string());
    }

    alaya::recovery::SnapshotManifest manifest;
    manifest.snapshot_id = snapshot_id;
    manifest.reason = std::string(reason);
    manifest.applied_through_op_id = last_committed_recovery_op_id_;
    manifest.created_unix_ms = alaya::recovery::SnapshotManifest::current_unix_ms();
    manifest.graph_file = graph_file;
    manifest.data_file = data_file;
    manifest.quant_file = quant_file;
    manifest.rocksdb_dir = rocksdb_dir;

    recovery_manager_->publish_snapshot(manifest, snapshot_dir);
  }

  [[nodiscard]] auto encode_insert_like_payload(const DataType *data,
                                                uint32_t ef,
                                                const ScalarData &scalar_data) const
      -> std::vector<char> {
    alaya::recovery::BinaryWriter writer;
    writer.write_u32(ef);
    writer.write_vector_blob(data, static_cast<size_t>(data_dim_));
    writer.write_blob(scalar_data.serialize());
    return std::move(writer).finish();
  }

  [[nodiscard]] auto encode_remove_item_payload(const std::string &item_id) const
      -> std::vector<char> {
    alaya::recovery::BinaryWriter writer;
    writer.write_string(item_id);
    return std::move(writer).finish();
  }

  [[nodiscard]] auto encode_remove_internal_payload(IDType internal_id) const -> std::vector<char> {
    alaya::recovery::BinaryWriter writer;
    writer.write_u64(static_cast<uint64_t>(internal_id));
    return std::move(writer).finish();
  }

  auto insert_nondurable(DataType *data, uint32_t ef, const ScalarData *scalar_data) -> IDType {
    if (update_job_ == nullptr) {
      throw std::runtime_error("incremental updates are not supported for the current index type");
    }
    auto inserted_id = update_job_->insert_and_update(data, ef, scalar_data);
    materialized_view_manager_.invalidate("insert");
    return inserted_id;
  }

  auto remove_nondurable(IDType id) -> void {
    if (update_job_ == nullptr) {
      throw std::runtime_error("incremental updates are not supported for the current index type");
    }
    update_job_->remove(id);
    materialized_view_manager_.invalidate("remove");
  }

  auto remove_nondurable(const std::string &item_id) -> void {
    if (update_job_ == nullptr) {
      throw std::runtime_error("incremental updates are not supported for the current index type");
    }
    update_job_->remove(item_id);
    materialized_view_manager_.invalidate("remove_by_item_id");
  }

  [[nodiscard]] auto copy_vector_by_internal_id(IDType internal_id) const -> std::vector<DataType> {
    std::vector<DataType> vector(data_dim_);
    const DataType *source = nullptr;

    if constexpr (is_rabitq_space_v<SearchSpaceType>) {
      source = search_space_->get_data_by_id(internal_id);
    } else {
      source = build_space_->get_data_by_id(internal_id);
    }

    std::copy(source, source + data_dim_, vector.begin());
    return vector;
  }

  auto upsert_nondurable(DataType *data, uint32_t ef, const ScalarData &scalar_data) -> IDType {
    if constexpr (!SearchSpaceType::has_scalar_data) {
      throw std::runtime_error("upsert requires scalar data support");
    } else {
      if (!contains(scalar_data.item_id)) {
        return insert_nondurable(data, ef, &scalar_data);
      }

      auto [old_internal_id, old_scalar] = search_space_->get_scalar_data(scalar_data.item_id);
      auto old_vector = copy_vector_by_internal_id(old_internal_id);
      remove_nondurable(scalar_data.item_id);

      try {
        return insert_nondurable(data, ef, &scalar_data);
      } catch (...) {
        LOG_ERROR("recovery: upsert failed after remove, attempting rollback for item_id={}",
                  scalar_data.item_id);
        try {
          insert_nondurable(old_vector.data(), ef, &old_scalar);
        } catch (const std::exception &rollback_error) {
          LOG_CRITICAL("recovery: rollback failed for item_id={} error={}",
                       scalar_data.item_id,
                       rollback_error.what());
        }
        throw;
      }
    }
  }

  auto replay_record(const alaya::recovery::WalRecord &record) -> void {
    alaya::recovery::BinaryReader reader(record.payload.data(), record.payload.size());

    switch (record.mutation_type) {
      case alaya::recovery::MutationType::INSERT:
      case alaya::recovery::MutationType::UPSERT: {
        auto ef = reader.read_u32();
        auto vector_blob = reader.read_blob();
        auto scalar_blob = reader.read_blob();
        if (!ef.has_value() || !vector_blob.has_value() || !scalar_blob.has_value()) {
          throw std::runtime_error("Invalid WAL insert/upsert payload");
        }
        if (vector_blob->size() != static_cast<size_t>(data_dim_) * sizeof(DataType)) {
          throw std::runtime_error("WAL vector payload dimension mismatch");
        }

        auto scalar_data = ScalarData::deserialize(scalar_blob->data(), scalar_blob->size());
        std::vector<DataType> vector(data_dim_);
        std::memcpy(vector.data(), vector_blob->data(), vector_blob->size());

        if (record.mutation_type == alaya::recovery::MutationType::INSERT) {
          insert_nondurable(vector.data(), ef.value(), &scalar_data);
        } else {
          upsert_nondurable(vector.data(), ef.value(), scalar_data);
        }
        break;
      }
      case alaya::recovery::MutationType::REMOVE_BY_ITEM_ID: {
        auto item_id = reader.read_string();
        if (!item_id.has_value()) {
          throw std::runtime_error("Invalid WAL remove-by-item-id payload");
        }
        if (contains(item_id.value())) {
          remove_nondurable(item_id.value());
        } else {
          LOG_WARN("recovery: skip removing missing item_id={} during replay", item_id.value());
        }
        break;
      }
      case alaya::recovery::MutationType::REMOVE_BY_INTERNAL_ID: {
        auto internal_id = reader.read_u64();
        if (!internal_id.has_value()) {
          throw std::runtime_error("Invalid WAL remove-by-id payload");
        }
        if (internal_id.value() < static_cast<uint64_t>(search_space_->get_data_num())) {
          remove_nondurable(static_cast<IDType>(internal_id.value()));
        } else {
          LOG_WARN("recovery: skip removing missing internal_id={} during replay",
                   internal_id.value());
        }
        break;
      }
    }
  }

  auto replay_recovery_log(uint64_t applied_through) -> size_t {
    if (recovery_manager_ == nullptr) {
      return 0;
    }

    auto records =
        recovery_manager_->replayable_records(applied_through, &last_seen_recovery_op_id_);
    for (const auto &record : records) {
      replay_record(record);
      last_committed_recovery_op_id_ = std::max(last_committed_recovery_op_id_, record.op_id);
    }
    if (!records.empty()) {
      LOG_INFO("recovery: replayed {} committed mutations", records.size());
    }
    next_recovery_op_id_ = std::max(next_recovery_op_id_, last_seen_recovery_op_id_ + 1);
    return records.size();
  }

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
    save_state(index_path, data_path, quant_path);
    checkpoint_recovery_snapshot("manual_save");
  }

  auto load(const std::string &index_path,
            const std::string &data_path = std::string(),
            const std::string &quant_path = std::string()) -> void {
    std::string resolved_index_path = index_path;
    std::string resolved_data_path = data_path;
    std::string resolved_quant_path = quant_path;
    uint64_t applied_through = 0;

    if (recovery_manager_ != nullptr) {
      auto manifest = recovery_manager_->current_snapshot();
      auto snapshot_dir = recovery_manager_->current_snapshot_dir();
      if (manifest.has_value() && snapshot_dir.has_value()) {
        recovery_manager_->restore_active_rocksdb_from_snapshot(manifest.value(),
                                                                snapshot_dir.value());
        resolved_index_path = manifest->graph_path(snapshot_dir.value());
        resolved_data_path = manifest->data_path(snapshot_dir.value());
        resolved_quant_path = manifest->quant_path(snapshot_dir.value());
        applied_through = manifest->applied_through_op_id;
        LOG_INFO("recovery: loading snapshot id={} applied_through={}",
                 manifest->snapshot_id,
                 applied_through);
      }
    }

    load_state(resolved_index_path, resolved_data_path, resolved_quant_path);
    auto materialized_view_ef_construction = std::max<uint32_t>(200, params_.max_nbrs_ * 4);
    auto materialized_view_build_threads = params_.materialized_view_build_threads_ != 0
                                               ? params_.materialized_view_build_threads_
                                           : params_.build_threads_ != 0 ? params_.build_threads_
                                                                         : 1;
    materialized_view_manager_.rebuild(params_,
                                       data_dim_,
                                       search_space_,
                                       build_space_,
                                       materialized_view_ef_construction,
                                       materialized_view_build_threads);

    auto replayed = replay_recovery_log(applied_through);
    if (recovery_manager_ != nullptr) {
      auto current_snapshot = recovery_manager_->current_snapshot();
      if (replayed > 0 || !current_snapshot.has_value()) {
        checkpoint_recovery_snapshot(replayed > 0 ? "post_recovery" : "post_load");
      }
    }
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
    auto materialized_view_ef_construction = ef_construction;
    auto materialized_view_build_threads = params_.materialized_view_build_threads_ != 0
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
    materialized_view_manager_.rebuild(params_,
                                       data_dim_,
                                       search_space_,
                                       build_space_,
                                       materialized_view_ef_construction,
                                       materialized_view_build_threads);
    checkpoint_recovery_snapshot("post_fit");
    LOG_DEBUG("Create task generator successfully!");
  }

  auto insert(py::array_t<DataType> insert_data,
              uint32_t ef,
              const std::string &item_id = "",
              const std::string &document = "",
              const py::dict &metadata = py::dict()) -> IDType {
    auto insert_data_ptr = static_cast<DataType *>(insert_data.request().ptr);
    MetadataMap meta_map = pydict_to_metadata_map(metadata);
    ScalarData scalar_data{item_id, document, meta_map};

    // TODO(P2): RocksDB has its own internal WAL and the custom WAL must stay
    // in sync. If the process crashes between insert_nondurable (RocksDB write)
    // and append_commit, replay may cause duplicates. Consider idempotent
    // replay (check if item_id already exists) or a unified WAL.
    if (recovery_manager_ != nullptr) {
      auto op_id = next_recovery_op_id_++;
      recovery_manager_->append_prepare(
          {op_id,
           alaya::recovery::MutationType::INSERT,
           encode_insert_like_payload(insert_data_ptr, ef, scalar_data)});
      auto inserted_id = insert_nondurable(insert_data_ptr, ef, &scalar_data);
      recovery_manager_->append_commit(op_id, alaya::recovery::MutationType::INSERT);
      last_committed_recovery_op_id_ = op_id;
      last_seen_recovery_op_id_ = std::max(last_seen_recovery_op_id_, op_id);
      return inserted_id;
    }

    auto inserted_id = insert_nondurable(insert_data_ptr, ef, &scalar_data);
    return inserted_id;
  }

  auto upsert(py::array_t<DataType> insert_data,
              uint32_t ef,
              const std::string &item_id = "",
              const std::string &document = "",
              const py::dict &metadata = py::dict()) -> IDType {
    auto insert_data_ptr = static_cast<DataType *>(insert_data.request().ptr);
    MetadataMap meta_map = pydict_to_metadata_map(metadata);
    ScalarData scalar_data{item_id, document, meta_map};

    if (recovery_manager_ != nullptr) {
      auto op_id = next_recovery_op_id_++;
      recovery_manager_->append_prepare(
          {op_id,
           alaya::recovery::MutationType::UPSERT,
           encode_insert_like_payload(insert_data_ptr, ef, scalar_data)});
      auto upserted_id = upsert_nondurable(insert_data_ptr, ef, scalar_data);
      recovery_manager_->append_commit(op_id, alaya::recovery::MutationType::UPSERT);
      last_committed_recovery_op_id_ = op_id;
      last_seen_recovery_op_id_ = std::max(last_seen_recovery_op_id_, op_id);
      return upserted_id;
    }

    return upsert_nondurable(insert_data_ptr, ef, scalar_data);
  }

  auto remove(IDType id) -> void {
    if (recovery_manager_ != nullptr) {
      auto op_id = next_recovery_op_id_++;
      recovery_manager_->append_prepare({op_id,
                                         alaya::recovery::MutationType::REMOVE_BY_INTERNAL_ID,
                                         encode_remove_internal_payload(id)});
      remove_nondurable(id);
      recovery_manager_->append_commit(op_id, alaya::recovery::MutationType::REMOVE_BY_INTERNAL_ID);
      last_committed_recovery_op_id_ = op_id;
      last_seen_recovery_op_id_ = std::max(last_seen_recovery_op_id_, op_id);
      return;
    }
    remove_nondurable(id);
  }

  auto remove(const std::string &item_id) -> void {
    if (recovery_manager_ != nullptr) {
      auto op_id = next_recovery_op_id_++;
      recovery_manager_->append_prepare({op_id,
                                         alaya::recovery::MutationType::REMOVE_BY_ITEM_ID,
                                         encode_remove_item_payload(item_id)});
      remove_nondurable(item_id);
      recovery_manager_->append_commit(op_id, alaya::recovery::MutationType::REMOVE_BY_ITEM_ID);
      last_committed_recovery_op_id_ = op_id;
      last_seen_recovery_op_id_ = std::max(last_seen_recovery_op_id_, op_id);
      return;
    }
    remove_nondurable(item_id);
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
   * @brief Batch get scalar data by internal IDs using RocksDB MultiGet.
   * @param internal_ids numpy array of internal IDs
   * @return Python list of scalar-data dicts
   */
  auto batch_get_scalar_data_by_internal_ids(py::array_t<IDType> internal_ids) -> py::list {
    if constexpr (!SearchSpaceType::has_scalar_data) {
      throw std::runtime_error("batch_get_scalar_data requires a space that supports scalar data");
    } else {
      auto buf = internal_ids.request();
      auto *id_ptr = static_cast<IDType *>(buf.ptr);
      size_t count = static_cast<size_t>(buf.size);
      std::vector<IDType> ids(id_ptr, id_ptr + count);

      std::vector<ScalarData> scalar_data;
      {
        py::gil_scoped_release release;
        auto *storage = search_space_->get_scalar_storage();
        scalar_data = storage->batch_get(ids);
      }

      py::list result;
      for (const auto &sd : scalar_data) {
        result.append(scalar_data_to_pydict(sd));
      }
      return result;
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
    if constexpr (is_rabitq_space_v<SearchSpaceType>) {
      throw std::runtime_error("batch_search_with_distance is not supported for RaBitQ space");
    }

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
  MaterializedViewManagerType materialized_view_manager_;
  std::unique_ptr<alaya::recovery::RecoveryManager> recovery_manager_{nullptr};
  uint64_t next_recovery_op_id_{1};
  uint64_t last_committed_recovery_op_id_{0};
  uint64_t last_seen_recovery_op_id_{0};
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

  auto get_data_by_id(const py::object &id_obj) -> py::array {  // NOLINT
    DISPATCH_AND_CAST(index, return index->get_data_by_id(id_obj.cast<IDType>()););
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

  auto upsert(py::array &insert_data,
              uint32_t ef,
              const py::object &item_id_obj = py::none(),
              const std::string &document = "",
              const py::dict &metadata = py::dict())
      -> std::variant<uint32_t, uint64_t> {  // NOLINT
    std::string item_id = item_id_obj.is_none() ? "" : py::str(item_id_obj).cast<std::string>();
    DISPATCH_AND_CAST_WITH_ARR(insert_data,
                               typed_insert_data,
                               index,
                               return index
                                   ->upsert(typed_insert_data, ef, item_id, document, metadata););
  }

  auto remove(const py::object &id_obj) -> void {  // NOLINT
    DISPATCH_AND_CAST(index, index->remove(id_obj.cast<IDType>()););
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

  auto get_scalar_data_by_internal_id(const py::object &internal_id_obj) -> py::dict {  // NOLINT
    DISPATCH_AND_CAST(index,
                      return index->get_scalar_data_by_internal_id(
                          internal_id_obj.cast<IDType>()););
  }

  auto batch_get_scalar_data_by_internal_ids(py::array internal_ids) -> py::list {  // NOLINT
    DISPATCH_AND_CAST(index, {
      auto typed_ids = internal_ids.cast<py::array_t<IDType>>();
      return index->batch_get_scalar_data_by_internal_ids(typed_ids);
    });
  }

  auto batch_get_item_ids_by_internal_ids(py::array internal_ids) -> py::list {  // NOLINT
    DISPATCH_AND_CAST(index, {
      auto typed_ids = internal_ids.cast<py::array_t<IDType>>();
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
