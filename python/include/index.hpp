// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
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
#include "executor/jobs/graph_hybrid_search_job.hpp"
#include "executor/jobs/graph_search_job.hpp"
#include "executor/jobs/graph_update_job.hpp"
#include "executor/scheduler.hpp"
#include "helpers/parse.hpp"
#include "index/graph/fusion_graph.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "index/graph/nsg/nsg_builder.hpp"
#include "index/graph/qg/qg_builder.hpp"
#include "materialized_view.hpp"
#include "params.hpp"
#include "recovery/recovery_manager.hpp"
#include "space/rabitq_space.hpp"
#include "space/raw_space.hpp"
#include "space/sq4_space.hpp"
#include "space/sq8_space.hpp"
#include "storage/rocksdb_storage.hpp"
#include "utils/binary_io.hpp"
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
  explicit PyIndex(IndexParams params);
  auto to_string() const -> std::string;
  auto get_materialized_view_partition_count() const -> uint32_t;

 private:
  void execute_hybrid_search_dispatch(const DataType *query,
                                      IDType *ids,
                                      const SearchInfo &search_info,
                                      const MetadataFilter &filter,
                                      bool brute_force_requested,
                                      std::string *item_ids) const;
  // todo: this cache may become a bottleneck under frequent thread-count changes.
  // Cache a thread pool per requested width to amortize batch-search setup.
  auto get_hybrid_batch_pool(uint32_t requested_threads) -> std::shared_ptr<alaya::ThreadPool>;
#if defined(__linux__)
  // add coroutine support
  auto execute_hybrid_search_dispatch_task(const DataType *query,
                                           IDType *ids,
                                           SearchInfo search_info,
                                           const MetadataFilter &filter,
                                           bool brute_force_requested,
                                           std::string *item_ids) const -> coro::task<>;
#endif

  void initialize_recovery();
  [[nodiscard]] auto recovery_scalar_storage() const -> RocksDBStorage<IDType> *;
  auto save_state(const std::string &index_path,
                  const std::string &data_path = std::string(),
                  const std::string &quant_path = std::string()) const -> void;
  auto load_state(const std::string &index_path,
                  const std::string &data_path = std::string(),
                  const std::string &quant_path = std::string()) -> void;
  auto checkpoint_recovery_snapshot(std::string_view reason) -> void;
  [[nodiscard]] auto encode_insert_like_payload(const DataType *data,
                                                uint32_t ef,
                                                const ScalarData &scalar_data) const
      -> std::vector<char>;
  [[nodiscard]] auto encode_remove_item_payload(const std::string &item_id) const
      -> std::vector<char>;
  [[nodiscard]] auto encode_remove_internal_payload(IDType internal_id) const -> std::vector<char>;
  auto insert_nondurable(DataType *data, uint32_t ef, const ScalarData *scalar_data) -> IDType;
  auto remove_nondurable(IDType id) -> void;
  auto remove_nondurable(const std::string &item_id) -> void;
  [[nodiscard]] auto copy_vector_by_internal_id(IDType internal_id) const -> std::vector<DataType>;
  auto upsert_nondurable(DataType *data, uint32_t ef, const ScalarData &scalar_data) -> IDType;
  auto replay_record(const alaya::recovery::WalRecord &record) -> void;
  auto replay_recovery_log(uint64_t applied_through) -> size_t;

 public:
  auto get_data_by_id(IDType id) -> py::array_t<DataType>;
  auto get_dim() const -> uint32_t;
  auto save(const std::string &index_path,
            const std::string &data_path = std::string(),
            const std::string &quant_path = std::string()) -> void;
  auto load(const std::string &index_path,
            const std::string &data_path = std::string(),
            const std::string &quant_path = std::string()) -> void;
  auto fit(py::array_t<DataType> vectors,
           uint32_t ef_construction,
           uint32_t num_threads,
           const py::object &item_ids = py::none(),
           const py::object &documents = py::none(),
           const py::object &metadata_list = py::none()) -> void;
  auto insert(py::array_t<DataType> insert_data,
              uint32_t ef,
              const std::string &item_id = "",
              const std::string &document = "",
              const py::dict &metadata = py::dict()) -> IDType;
  auto upsert(py::array_t<DataType> insert_data,
              uint32_t ef,
              const std::string &item_id = "",
              const std::string &document = "",
              const py::dict &metadata = py::dict()) -> IDType;
  auto remove(IDType id) -> void;
  auto remove(const std::string &item_id) -> void;
  /**
   * @brief Check if item_id exists in the index
   * @param item_id The item_id to check
   * @return true if exists, false otherwise
   */
  auto contains(const std::string &item_id) -> bool;
  /**
   * @brief Get scalar data by item_id
   * @param item_id The item_id to look up
   * @return Python dict containing internal_id, item_id, document, and metadata
   * @throws std::runtime_error if item_id not found or no scalar data available
   */
  auto get_scalar_data_by_item_id(const std::string &item_id) -> py::dict;
  /**
   * @brief Get scalar data by internal ID
   * @param internal_id The internal ID
   * @return Python dict containing item_id, document, and metadata
   */
  auto get_scalar_data_by_internal_id(IDType internal_id) -> py::dict;
  /**
   * @brief Batch get scalar data by internal IDs using RocksDB MultiGet.
   * @param internal_ids numpy array of internal IDs
   * @return Python list of scalar-data dicts
   */
  auto batch_get_scalar_data_by_internal_ids(py::array_t<IDType> internal_ids) -> py::list;
  /**
   * @brief Batch get item_ids by internal IDs (lightweight, uses MultiGet)
   * @param internal_ids numpy array of internal IDs
   * @return Python list of item_id strings
   */
  auto batch_get_item_ids_by_internal_ids(py::array_t<IDType> internal_ids) -> py::list;
  /**
   * @brief Get the number of vectors in the index
   * @return Number of vectors
   */
  auto get_data_num() -> IDType;
  auto search(py::array_t<DataType> query, uint32_t topk, uint32_t ef) -> py::array_t<IDType>;
  auto search_with_distance(py::array_t<DataType> query, uint32_t topk, uint32_t ef) -> py::object;
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
                     const std::string &filter_exec_hint = std::string()) -> py::object;
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
                           const std::string &filter_exec_hint = std::string()) -> py::object;
  /**
   * @brief Filter query without vector search
   * @param filter Metadata filter
   * @param limit Maximum number of results
   * @return Tuple of (ids_list, scalar_data_list)
   */
  auto filter_query(const MetadataFilter &filter, uint32_t limit) -> py::object;
  auto batch_search(py::array_t<DataType> queries, uint32_t topk, uint32_t ef, uint32_t num_threads)
      -> py::array_t<IDType>;
  auto batch_search_with_distance(py::array_t<DataType> queries,
                                  uint32_t topk,
                                  uint32_t ef,
                                  uint32_t num_threads) -> py::object;
  /**
   * @brief Close the RocksDB storage explicitly
   */
  auto close_db() -> void;

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

}  // namespace alaya

#include "instantiations/dispatch.hpp"

namespace alaya {

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
