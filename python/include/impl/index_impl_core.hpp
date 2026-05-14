// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "index.hpp"

namespace alaya {

template <typename GraphBuilderType, typename SearchSpaceType>
PyIndex<GraphBuilderType, SearchSpaceType>::PyIndex(IndexParams params)
    : params_(std::move(params)) {
  initialize_recovery();
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::to_string() const -> std::string {
  return "PyIndex";
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::get_materialized_view_partition_count() const
    -> uint32_t {
  return materialized_view_manager_.get_partition_count();
}

template <typename GraphBuilderType, typename SearchSpaceType>
void PyIndex<GraphBuilderType, SearchSpaceType>::initialize_recovery() {
  if (params_.rocksdb_path_.empty()) {
    return;
  }
  auto recovery_root = std::filesystem::path(params_.rocksdb_path_).parent_path() / "recovery";
  recovery_manager_ =
      std::make_unique<alaya::recovery::RecoveryManager>(recovery_root,
                                                         std::filesystem::path(
                                                             params_.rocksdb_path_));

  uint64_t max_seen_op_id = 0;
  (void)recovery_manager_->replayable_records(0, &max_seen_op_id);
  auto manifest = recovery_manager_->current_snapshot();
  if (manifest.has_value()) {
    last_committed_recovery_op_id_ = manifest->applied_through_op_id_;
  }
  last_seen_recovery_op_id_ = std::max(max_seen_op_id, last_committed_recovery_op_id_);
  next_recovery_op_id_ = last_seen_recovery_op_id_ + 1;
}

template <typename GraphBuilderType, typename SearchSpaceType>
[[nodiscard]] auto PyIndex<GraphBuilderType, SearchSpaceType>::recovery_scalar_storage() const
    -> RocksDBStorage<IDType> * {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::save_state(const std::string &index_path,
                                                            const std::string &data_path,
                                                            const std::string &quant_path) const
    -> void {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::load_state(const std::string &index_path,
                                                            const std::string &data_path,
                                                            const std::string &quant_path) -> void {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::checkpoint_recovery_snapshot(
    std::string_view reason) -> void {
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
  manifest.snapshot_id_ = snapshot_id;
  manifest.reason_ = std::string(reason);
  manifest.applied_through_op_id_ = last_committed_recovery_op_id_;
  manifest.created_unix_ms_ = alaya::recovery::SnapshotManifest::current_unix_ms();
  manifest.graph_file_ = graph_file;
  manifest.data_file_ = data_file;
  manifest.quant_file_ = quant_file;
  manifest.rocksdb_dir_ = rocksdb_dir;

  recovery_manager_->publish_snapshot(manifest, snapshot_dir);
}

template <typename GraphBuilderType, typename SearchSpaceType>
[[nodiscard]] auto PyIndex<GraphBuilderType, SearchSpaceType>::encode_insert_like_payload(
    const DataType *data,
    uint32_t ef,
    const ScalarData &scalar_data) const -> std::vector<char> {
  alaya::binary_io::BinaryWriter writer;
  writer.write_u32(ef);
  writer.write_vector_blob(data, static_cast<size_t>(data_dim_));
  writer.write_blob(scalar_data.serialize());
  return std::move(writer).finish();
}

template <typename GraphBuilderType, typename SearchSpaceType>
[[nodiscard]] auto PyIndex<GraphBuilderType, SearchSpaceType>::encode_remove_item_payload(
    const std::string &item_id) const -> std::vector<char> {
  alaya::binary_io::BinaryWriter writer;
  writer.write_string(item_id);
  return std::move(writer).finish();
}

template <typename GraphBuilderType, typename SearchSpaceType>
[[nodiscard]] auto PyIndex<GraphBuilderType, SearchSpaceType>::encode_remove_internal_payload(
    IDType internal_id) const -> std::vector<char> {
  alaya::binary_io::BinaryWriter writer;
  writer.write_u64(static_cast<uint64_t>(internal_id));
  return std::move(writer).finish();
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::insert_nondurable(DataType *data,
                                                                   uint32_t ef,
                                                                   const ScalarData *scalar_data)
    -> IDType {
  if (update_job_ == nullptr) {
    throw std::runtime_error("incremental updates are not supported for the current index type");
  }
  auto inserted_id = update_job_->insert_and_update(data, ef, scalar_data);
  materialized_view_manager_.invalidate("insert");
  return inserted_id;
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::remove_nondurable(IDType id) -> void {
  if (update_job_ == nullptr) {
    throw std::runtime_error("incremental updates are not supported for the current index type");
  }
  update_job_->remove(id);
  materialized_view_manager_.invalidate("remove");
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::remove_nondurable(const std::string &item_id)
    -> void {
  if (update_job_ == nullptr) {
    throw std::runtime_error("incremental updates are not supported for the current index type");
  }
  update_job_->remove(item_id);
  materialized_view_manager_.invalidate("remove_by_item_id");
}

template <typename GraphBuilderType, typename SearchSpaceType>
[[nodiscard]] auto PyIndex<GraphBuilderType, SearchSpaceType>::copy_vector_by_internal_id(
    IDType internal_id) const -> std::vector<DataType> {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::upsert_nondurable(DataType *data,
                                                                   uint32_t ef,
                                                                   const ScalarData &scalar_data)
    -> IDType {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::replay_record(
    const alaya::recovery::WalRecord &record) -> void {
  alaya::binary_io::BinaryReader reader(record.payload_.data(), record.payload_.size());

  switch (record.mutation_type_) {
    case alaya::recovery::MutationType::kInsert:
    case alaya::recovery::MutationType::kUpsert: {
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

      if (record.mutation_type_ == alaya::recovery::MutationType::kInsert) {
        insert_nondurable(vector.data(), ef.value(), &scalar_data);
      } else {
        upsert_nondurable(vector.data(), ef.value(), scalar_data);
      }
      break;
    }
    case alaya::recovery::MutationType::kRemoveByItemId: {
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
    case alaya::recovery::MutationType::kRemoveByInternalId: {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::replay_recovery_log(uint64_t applied_through)
    -> size_t {
  if (recovery_manager_ == nullptr) {
    return 0;
  }

  auto records = recovery_manager_->replayable_records(applied_through, &last_seen_recovery_op_id_);
  for (const auto &record : records) {
    replay_record(record);
    last_committed_recovery_op_id_ = std::max(last_committed_recovery_op_id_, record.op_id_);
  }
  if (!records.empty()) {
    LOG_INFO("recovery: replayed {} committed mutations", records.size());
  }
  next_recovery_op_id_ = std::max(next_recovery_op_id_, last_seen_recovery_op_id_ + 1);
  return records.size();
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::get_data_by_id(IDType id)
    -> py::array_t<DataType> {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::get_dim() const -> uint32_t {
  return data_dim_;
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::save(const std::string &index_path,
                                                      const std::string &data_path,
                                                      const std::string &quant_path) -> void {
  save_state(index_path, data_path, quant_path);
  checkpoint_recovery_snapshot("manual_save");
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::load(const std::string &index_path,
                                                      const std::string &data_path,
                                                      const std::string &quant_path) -> void {
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
      applied_through = manifest->applied_through_op_id_;
      LOG_INFO("recovery: loading snapshot id={} applied_through={}",
               manifest->snapshot_id_,
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::fit(py::array_t<DataType> vectors,
                                                     uint32_t ef_construction,
                                                     uint32_t num_threads,
                                                     const py::object &item_ids,
                                                     const py::object &documents,
                                                     const py::object &metadata_list) -> void {
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
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::insert(py::array_t<DataType> insert_data,
                                                        uint32_t ef,
                                                        const std::string &item_id,
                                                        const std::string &document,
                                                        const py::dict &metadata) -> IDType {
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
         alaya::recovery::MutationType::kInsert,
         encode_insert_like_payload(insert_data_ptr, ef, scalar_data)});
    auto inserted_id = insert_nondurable(insert_data_ptr, ef, &scalar_data);
    recovery_manager_->append_commit(op_id, alaya::recovery::MutationType::kInsert);
    last_committed_recovery_op_id_ = op_id;
    last_seen_recovery_op_id_ = std::max(last_seen_recovery_op_id_, op_id);
    return inserted_id;
  }

  auto inserted_id = insert_nondurable(insert_data_ptr, ef, &scalar_data);
  return inserted_id;
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::upsert(py::array_t<DataType> insert_data,
                                                        uint32_t ef,
                                                        const std::string &item_id,
                                                        const std::string &document,
                                                        const py::dict &metadata) -> IDType {
  auto insert_data_ptr = static_cast<DataType *>(insert_data.request().ptr);
  MetadataMap meta_map = pydict_to_metadata_map(metadata);
  ScalarData scalar_data{item_id, document, meta_map};

  if (recovery_manager_ != nullptr) {
    auto op_id = next_recovery_op_id_++;
    recovery_manager_->append_prepare(
        {op_id,
         alaya::recovery::MutationType::kUpsert,
         encode_insert_like_payload(insert_data_ptr, ef, scalar_data)});
    auto upserted_id = upsert_nondurable(insert_data_ptr, ef, scalar_data);
    recovery_manager_->append_commit(op_id, alaya::recovery::MutationType::kUpsert);
    last_committed_recovery_op_id_ = op_id;
    last_seen_recovery_op_id_ = std::max(last_seen_recovery_op_id_, op_id);
    return upserted_id;
  }

  return upsert_nondurable(insert_data_ptr, ef, scalar_data);
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::remove(IDType id) -> void {
  if (recovery_manager_ != nullptr) {
    auto op_id = next_recovery_op_id_++;
    recovery_manager_->append_prepare({op_id,
                                       alaya::recovery::MutationType::kRemoveByInternalId,
                                       encode_remove_internal_payload(id)});
    remove_nondurable(id);
    recovery_manager_->append_commit(op_id, alaya::recovery::MutationType::kRemoveByInternalId);
    last_committed_recovery_op_id_ = op_id;
    last_seen_recovery_op_id_ = std::max(last_seen_recovery_op_id_, op_id);
    return;
  }
  remove_nondurable(id);
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::remove(const std::string &item_id) -> void {
  if (recovery_manager_ != nullptr) {
    auto op_id = next_recovery_op_id_++;
    recovery_manager_->append_prepare({op_id,
                                       alaya::recovery::MutationType::kRemoveByItemId,
                                       encode_remove_item_payload(item_id)});
    remove_nondurable(item_id);
    recovery_manager_->append_commit(op_id, alaya::recovery::MutationType::kRemoveByItemId);
    last_committed_recovery_op_id_ = op_id;
    last_seen_recovery_op_id_ = std::max(last_seen_recovery_op_id_, op_id);
    return;
  }
  remove_nondurable(item_id);
}

}  // namespace alaya
