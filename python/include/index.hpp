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
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <queue>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include "dispatch.hpp"
#include "executor/jobs/graph_search_job.hpp"
#include "executor/jobs/graph_update_job.hpp"
#include "executor/scheduler.hpp"
#include "index/graph/fusion_graph.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "index/graph/nsg/nsg_builder.hpp"
#include "index/graph/qg/qg_builder.hpp"
#include "params.hpp"
#include "space/rabitq_space.hpp"
#include "space/raw_space.hpp"
#include "space/sq4_space.hpp"
#include "space/sq8_space.hpp"
#include "storage/rocksdb_storage.hpp"
#include "utils/log.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/metric_type.hpp"
#include "utils/scalar_data.hpp"
#include "utils/types.hpp"

namespace py = pybind11;

namespace alaya {
// NOLINTBEGIN
template <typename T>
auto get_topk_array(const std::vector<std::vector<T>> &res_pool, size_t topk) -> py::array_t<T> {
  size_t query_size = res_pool.size();
  if (query_size == 0 || topk == 0) {
    return py::array_t<T>({query_size, topk});  // Return empty array if dimensions are zero
  }

  py::array_t<T> ret({query_size, topk});
  T *ret_data = ret.mutable_data();

  size_t output_row_byte_stride = topk * sizeof(T);
  for (size_t i = 0; i < query_size; ++i) {
    // ef must be greater or equal to topk
    std::memcpy(ret_data + (i * topk), res_pool[i].data(), output_row_byte_stride);
  }
  return ret;
}

/**
 * @brief Convert py::dict to MetadataMap
 */
inline auto pydict_to_metadata_map(const py::dict &meta) -> MetadataMap {
  MetadataMap meta_map;
  for (auto item : meta) {
    std::string key = py::str(item.first);
    auto value = item.second;
    if (py::isinstance<py::bool_>(value)) {
      meta_map[key] = value.cast<bool>();
    } else if (py::isinstance<py::int_>(value)) {
      meta_map[key] = value.cast<int64_t>();
    } else if (py::isinstance<py::float_>(value)) {
      meta_map[key] = value.cast<double>();
    } else if (py::isinstance<py::str>(value)) {
      meta_map[key] = value.cast<std::string>();
    }
  }
  return meta_map;
}

/**
 * @brief Convert MetadataMap to py::dict
 */
inline auto metadata_map_to_pydict(const MetadataMap &meta_map) -> py::dict {
  py::dict meta;
  for (const auto &[key, value] : meta_map) {
    std::visit(
        [&meta, &key](auto &&arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, bool>) {
            meta[key.c_str()] = py::bool_(arg);
          } else if constexpr (std::is_same_v<T, int64_t>) {
            meta[key.c_str()] = py::int_(arg);
          } else if constexpr (std::is_same_v<T, double>) {
            meta[key.c_str()] = py::float_(arg);
          } else if constexpr (std::is_same_v<T, std::string>) {
            meta[key.c_str()] = py::str(arg);
          }
        },
        value);
  }
  return meta;
}

/**
 * @brief Convert ScalarData to py::dict
 */
inline auto scalar_data_to_pydict(const ScalarData &scalar_data) -> py::dict {
  py::dict result;
  result["item_id"] = scalar_data.item_id;
  result["document"] = scalar_data.document;
  result["metadata"] = metadata_map_to_pydict(scalar_data.metadata);
  return result;
}

/**
 * @brief Build ScalarData vector from Python lists
 */
inline auto build_scalar_data_vec(const py::list &item_ids,
                                  const py::object &documents,
                                  const py::object &metadata_list,
                                  size_t count) -> std::vector<ScalarData> {
  std::vector<ScalarData> scalar_data_vec;
  scalar_data_vec.reserve(count);

  py::list docs = documents.is_none() ? py::list() : documents.cast<py::list>();
  py::list metas = metadata_list.is_none() ? py::list() : metadata_list.cast<py::list>();

  for (size_t i = 0; i < count; i++) {
    MetadataMap meta_map;
    if (i < metas.size()) {
      meta_map = pydict_to_metadata_map(metas[i].cast<py::dict>());
    }
    std::string doc = (i < docs.size()) ? docs[i].cast<std::string>() : "";
    // Convert item_id to string using Python's str() for any type
    std::string item_id_str = py::str(item_ids[i]).cast<std::string>();
    scalar_data_vec.emplace_back(item_id_str, doc, std::move(meta_map));
  }
  return scalar_data_vec;
}

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

  PyIndex() = delete;
  explicit PyIndex(IndexParams params) : params_(std::move(params)) {};

  auto to_string() const -> std::string { return "PyIndex"; }

  auto get_data_by_id(IDType id) -> py::array_t<DataType> {
    if (build_space_ == nullptr) {
      throw std::runtime_error("space is nullptr");
    }

    if (id >= build_space_->get_data_num()) {
      throw std::runtime_error("id out of range");
    }

    auto data = build_space_->get_data_by_id(id);
    return py::array_t<DataType>({data_dim_}, {sizeof(DataType)}, data);
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
                                                                                   nullptr);
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
      update_job_ = std::make_shared<GraphUpdateJob<SearchSpaceType, BuildSpaceType>>(search_job_);
    }
    LOG_INFO("creator task generator success");
  }

  auto fit(py::array_t<DataType> &vectors,
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

    // TODO: merge
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
                                                                                   nullptr);
    } else {
      build_space_ =
          std::make_shared<BuildSpaceType>(params_.capacity_, data_dim_, params_.metric_);

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
      update_job_ = std::make_shared<GraphUpdateJob<SearchSpaceType, BuildSpaceType>>(search_job_);
    }
    LOG_INFO("Create task generator successfully!");
  }

  auto insert(py::array_t<DataType> &insert_data,
              uint32_t ef,
              const std::string &item_id = "",
              const std::string &document = "",
              const py::dict &metadata = py::dict()) -> IDType {
    auto insert_data_ptr = static_cast<DataType *>(insert_data.request().ptr);
    MetadataMap meta_map = pydict_to_metadata_map(metadata);
    ScalarData scalar_data{item_id, document, meta_map};
    return update_job_->insert_and_update(insert_data_ptr, ef, &scalar_data);
  }

  auto remove(uint32_t id) -> void { update_job_->remove(id); }

  auto remove(const std::string &item_id) -> void { update_job_->remove(item_id); }

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
      auto scalar_data = search_space_->get_scalar_data(internal_id);
      return scalar_data_to_pydict(scalar_data);
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

  auto search(py::array_t<DataType> &query, uint32_t topk, uint32_t ef) -> py::array_t<IDType> {
    auto *query_ptr = static_cast<DataType *>(query.request().ptr);
    auto ret = py::array_t<IDType>(static_cast<size_t>(topk));
    auto ret_ptr = static_cast<IDType *>(ret.request().ptr);

    if constexpr (is_rabitq_space_v<SearchSpaceType>) {
      search_job_->rabitq_search_solo(query_ptr, topk, ret_ptr, ef);
    } else {
      search_job_->search_solo(query_ptr, ret_ptr, topk, ef);
    }

    return ret;
  }

  auto search_with_distance(py::array_t<DataType> &query, uint32_t topk, uint32_t ef)
      -> py::object {
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
  auto hybrid_search(py::array_t<DataType> &query,
                     uint32_t topk,
                     uint32_t ef,
                     const MetadataFilter &filter) -> py::object {
    if constexpr (!SearchSpaceType::has_scalar_data) {
      throw std::runtime_error("hybrid_search requires a space that supports scalar data");
    } else {
      auto *query_ptr = static_cast<DataType *>(query.request().ptr);

      auto ret_ids = py::array_t<IDType>(static_cast<size_t>(topk));
      auto ret_id_ptr = static_cast<IDType *>(ret_ids.request().ptr);

      std::vector<std::string> item_ids(topk);

      if constexpr (is_rabitq_space_v<SearchSpaceType>) {
        search_job_
            ->rabitq_hybrid_search_solo(query_ptr, topk, ret_id_ptr, ef, filter, item_ids.data());
      } else {
        search_job_->hybrid_search_solo(query_ptr, ret_id_ptr, topk, ef, filter, item_ids.data());
      }

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
  auto batch_hybrid_search(py::array_t<DataType> &queries,
                           uint32_t topk,
                           uint32_t ef,
                           const MetadataFilter &filter,
                           uint32_t num_threads) -> py::object {
    if constexpr (!SearchSpaceType::has_scalar_data) {
      throw std::runtime_error("batch_hybrid_search requires a space that supports scalar data");
    } else {
      auto shape = queries.shape();
      size_t query_size = shape[0];
      size_t query_dim = shape[1];

      auto *query_ptr = static_cast<DataType *>(queries.request().ptr);

      std::vector<std::vector<IDType>> id_results(query_size, std::vector<IDType>(topk));
      std::vector<std::vector<std::string>> item_id_results(query_size,
                                                            std::vector<std::string>(topk));

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
          coros.emplace_back(search_job_->rabitq_hybrid_search(cur_query,
                                                               topk,
                                                               id_results[i].data(),
                                                               ef,
                                                               filter,
                                                               item_id_results[i].data()));
        } else {
          coros.emplace_back(search_job_->hybrid_search(cur_query,
                                                        id_results[i].data(),
                                                        topk,
                                                        ef,
                                                        filter,
                                                        item_id_results[i].data()));
        }
        scheduler->schedule(coros.back().handle());
      }
      scheduler->begin();
      scheduler->join();

      // Build result arrays
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

  auto batch_search(py::array_t<DataType> &queries,
                    uint32_t topk,
                    uint32_t ef,
                    uint32_t num_threads) -> py::array_t<IDType> {
    auto shape = queries.shape();
    size_t query_size = shape[0];
    size_t query_dim = shape[1];

    auto *query_ptr = static_cast<DataType *>(queries.request().ptr);

#if defined(__linux__)
    std::vector<std::vector<IDType>> res_pool(query_size, std::vector<IDType>(topk));

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

    auto ret = py::array_t<IDType>({query_size, static_cast<size_t>(topk)});
    auto ret_ptr = static_cast<IDType *>(ret.request().ptr);
    for (size_t i = 0; i < query_size; i++) {
      std::copy(res_pool[i].begin(), res_pool[i].end(), ret_ptr + i * topk);
    }
    return ret;
#else
    auto ret = py::array_t<IDType>({query_size, static_cast<size_t>(topk)});
    auto ret_ptr = static_cast<IDType *>(ret.request().ptr);
    for (size_t i = 0; i < query_size; i++) {
      auto cur_query = query_ptr + i * query_dim;

      if constexpr (is_rabitq_space_v<SearchSpaceType>) {
        search_job_->rabitq_search_solo(cur_query, topk, ret_ptr + i * topk, ef);
      } else {
        // search_solo now handles rerank internally and returns topk results
        search_job_->search_solo(cur_query, ret_ptr + i * topk, topk, ef);
      }
    }
    return ret;

#endif
  }

  auto batch_search_with_distance(py::array_t<DataType> &queries,
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

    auto ret_id = get_topk_array(topk_ids, topk);
    auto ret_dist = get_topk_array(topk_dists, topk);
    return py::make_tuple(ret_id, ret_dist);
#else
    auto ret_id = py::array_t<IDType>({query_size, static_cast<size_t>(topk)});
    auto ret_dist = py::array_t<DistanceType>({query_size, static_cast<size_t>(topk)});

    auto ret_id_ptr = static_cast<IDType *>(ret_id.request().ptr);
    auto ret_dist_ptr = static_cast<DistanceType *>(ret_dist.request().ptr);

    for (size_t i = 0; i < query_size; i++) {
      auto cur_query = query_ptr + i * query_dim;
      // search_solo now handles rerank internally and returns topk results with distances
      search_job_->search_solo(cur_query, ret_id_ptr + i * topk, ret_dist_ptr + i * topk, topk, ef);
    }

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
  std::shared_ptr<alaya::GraphUpdateJob<SearchSpaceType, BuildSpaceType>> update_job_{nullptr};
  std::shared_ptr<JobContext<IDType>> job_context_{nullptr};
};

class PyIndexInterface {
 public:
  explicit PyIndexInterface(const IndexParams &params) : params_(params) {  // NOLINT
    DISPATCH_AND_CREATE(params);
  };

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

  auto filter_query(const MetadataFilter &filter, uint32_t limit) -> py::object {  // NOLINT
    DISPATCH_AND_CAST(index, return index->filter_query(filter, limit););
  }

  auto get_data_num() -> std::variant<uint32_t, uint64_t> {  // NOLINT
    DISPATCH_AND_CAST(index, return index->get_data_num(););
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

  auto hybrid_search(py::array &query, uint32_t topk, uint32_t ef, const MetadataFilter &filter)
      -> py::object {
    DISPATCH_AND_CAST_WITH_ARR(query,
                               typed_query,
                               index,
                               return index->hybrid_search(typed_query, topk, ef, filter););
  }

  auto batch_hybrid_search(py::array &queries,
                           uint32_t topk,
                           uint32_t ef,
                           const MetadataFilter &filter,
                           uint32_t num_threads) -> py::object {
    DISPATCH_AND_CAST_WITH_ARR(queries,
                               typed_queries,
                               index,
                               return index->batch_hybrid_search(typed_queries,
                                                                 topk,
                                                                 ef,
                                                                 filter,
                                                                 num_threads););
  }

  auto close_db() -> void {  // NOLINT
    DISPATCH_AND_CAST(index, index->close_db(););
  }

  virtual ~PyIndexInterface() = default;
  IndexParams params_;
  std::shared_ptr<BasePyIndex> index_;
};
// NOLINTEND
}  // namespace alaya
