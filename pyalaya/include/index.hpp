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
#include <filesystem>
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
#include "params.hpp"
#include "space/raw_space.hpp"
#include "space/sq4_space.hpp"
#include "space/sq8_space.hpp"
#include "utils/log.hpp"
#include "utils/metric_type.hpp"
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

  PyIndex() = delete;
  explicit PyIndex(IndexParams params) : params_(std::move(params)){};

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

  auto save(const std::string &index_path, const std::string &data_path = std::string(),
            const std::string &quant_path = std::string()) -> void {
    std::string_view index_path_view{index_path};
    std::string_view data_path_view{data_path};
    std::string_view quant_path_view{quant_path};

    graph_index_->save(index_path_view);
    if (!data_path.empty()) {
      build_space_->save(data_path_view);
    }
    if (!quant_path.empty()) {
      search_space_->save(quant_path_view);
    }
  }

  auto load(const std::string &index_path, const std::string &data_path = std::string(),
            const std::string &quant_path = std::string()) -> void {
    // index_path_ = index_path;
    std::string_view index_path_view{index_path};
    std::string_view data_path_view{data_path};
    std::string_view quant_path_view{quant_path};

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

    data_size_ = build_space_->data_size_;
    data_dim_ = build_space_->dim_;

    job_context_ = std::make_shared<JobContext<IDType>>();
    search_job_ = std::make_shared<alaya::GraphSearchJob<SearchSpaceType>>(
        search_space_, graph_index_, job_context_);
    update_job_ = std::make_shared<GraphUpdateJob<SearchSpaceType>>(search_job_);
    LOG_INFO("creator task generator success");
  }

  auto fit(py::array_t<DataType> &vectors, uint32_t ef_construction, uint32_t num_threads) -> void {
    LOG_INFO("start fit data");

    if (vectors.ndim() != 2) {
      throw std::runtime_error("Array must be 2D");
    }

    data_size_ = vectors.shape(0);
    data_dim_ = vectors.shape(1);
    vectors_ = static_cast<DataType *>(vectors.request().ptr);

    build_space_ = std::make_shared<BuildSpaceType>(params_.capacity_, data_dim_, params_.metric_);
    build_space_->fit(vectors_, data_size_);
    if constexpr (std::is_same<BuildSpaceType, SearchSpaceType>::value) {
      search_space_ = build_space_;
    } else {
      search_space_ =
          std::make_shared<SearchSpaceType>(params_.capacity_, data_dim_, params_.metric_);
      search_space_->fit(vectors_, data_size_);
    }

    auto build_start = std::chrono::steady_clock::now();
    auto graph_builder = std::make_shared<HNSWBuilder<BuildSpaceType>>(
        build_space_, params_.max_nbrs_, ef_construction);
    graph_index_ = graph_builder->build_graph(num_threads);

    LOG_INFO(
        "The time of building hnsw is {}s.",
        static_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - build_start)
            .count());

    job_context_ = std::make_shared<JobContext<IDType>>();

    search_job_ = std::make_shared<alaya::GraphSearchJob<SearchSpaceType>>(
        search_space_, graph_index_, job_context_);
    update_job_ = std::make_shared<GraphUpdateJob<SearchSpaceType>>(search_job_);
    LOG_INFO("Create task generator successfully!");
  }

  auto insert(py::array_t<DataType> &insert_data, uint32_t ef) -> IDType {
    auto insert_data_ptr = static_cast<DataType *>(insert_data.request().ptr);
    return update_job_->insert_and_update(insert_data_ptr, ef);
  }

  auto remove(uint32_t id) -> void { update_job_->remove(id); }

  auto search(py::array_t<DataType> &query, uint32_t topk, uint32_t ef) -> py::array_t<IDType> {
    auto *query_ptr = static_cast<DataType *>(query.request().ptr);

    std::vector<IDType> res_pool(ef);

    search_job_->search_solo(query_ptr, topk, res_pool.data(), ef);

    auto ret = py::array_t<IDType>(static_cast<size_t>(topk));
    auto ret_ptr = static_cast<IDType *>(ret.request().ptr);

    if constexpr (std::is_same<SearchSpaceType, BuildSpaceType>::value) {
      std::copy(res_pool.begin(), res_pool.begin() + topk, ret_ptr);
    } else {
      rerank(res_pool, ret_ptr, build_space_->get_query_computer(query_ptr), ef, topk);
    }

    return ret;
  }


#if (not defined (__NO_COROUTINE__))
  auto batch_search(py::array_t<DataType> &queries, uint32_t topk, uint32_t ef,
                    uint32_t num_threads) -> py::array_t<IDType> {
    auto shape = queries.shape();
    size_t query_size = shape[0];
    size_t query_dim = shape[1];

    auto *query_ptr = static_cast<DataType *>(queries.request().ptr);

    Timer timer{};
    std::vector<std::vector<IDType>> res_pool(query_size, std::vector<IDType>(ef));

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
      coros.emplace_back(search_job_->search(cur_query, topk, res_pool[i].data(), ef));
      scheduler->schedule(coros.back().handle());
    }
    LOG_INFO("Scheduling {} tasks.", coros.size());
    scheduler->begin();
    scheduler->join();

    LOG_INFO("Total time: {} s.", timer.elapsed() / 1000000.0);

    auto ret = py::array_t<IDType>({query_size, static_cast<size_t>(topk)});
    auto ret_ptr = static_cast<IDType *>(ret.request().ptr);
    if constexpr (std::is_same<SearchSpaceType, BuildSpaceType>::value) {
      for (size_t i = 0; i < query_size; i++) {
        std::copy(res_pool[i].begin(), res_pool[i].begin() + topk, ret_ptr + i * topk);
      }
    } else {
      for (size_t i = 0; i < query_size; i++) {
        rerank(res_pool[i], ret_ptr + i * topk,
               build_space_->get_query_computer(query_ptr + i * query_dim), ef, topk);
      }
    }
    return ret;
  }
#endif

  auto batch_search(py::array_t<DataType> &queries, uint32_t topk, uint32_t ef,
                    uint32_t num_threads) -> py::array_t<IDType> {
    auto shape = queries.shape();
    size_t query_size = shape[0];
    size_t query_dim = shape[1];

    auto *query_ptr = static_cast<DataType *>(queries.request().ptr);

    Timer timer{};
    std::vector<std::vector<IDType>> res_pool(query_size, std::vector<IDType>(ef));

    std::vector<CpuID> worker_cpus;
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < query_size; i++) {
      auto cur_query = query_ptr + (i * query_dim);
      futures.emplace_back(std::async(std::launch::async, [this, cur_query, topk, ef,
                                                           &res_pool, i]() {
        search_job_->search_solo(cur_query, topk, res_pool[i].data(), ef);
      }));
    }
    
    for (auto &f : futures) {
      f.get();
    }

    LOG_INFO("Total time: {} s.", timer.elapsed() / 1000000.0);

    auto ret = py::array_t<IDType>({query_size, static_cast<size_t>(topk)});
    auto ret_ptr = static_cast<IDType *>(ret.request().ptr);
    if constexpr (std::is_same<SearchSpaceType, BuildSpaceType>::value) {
      for (size_t i = 0; i < query_size; i++) {
        std::copy(res_pool[i].begin(), res_pool[i].begin() + topk, ret_ptr + (i * topk));
      }
    } else {
      for (size_t i = 0; i < query_size; i++) {
        rerank(res_pool[i], ret_ptr + (i * topk),
               build_space_->get_query_computer(query_ptr + (i * query_dim)), ef, topk);
      }
    }
    return ret;
  }

  auto batch_search_with_distance(py::array_t<DataType> &queries, uint32_t topk, uint32_t ef,
                                  uint32_t num_threads) -> py::object {
    auto shape = queries.shape();
    size_t query_size = shape[0];
    size_t query_dim = shape[1];

    auto *query_ptr = static_cast<DataType *>(queries.request().ptr);

    Timer timer{};
    std::vector<std::vector<IDType>> res_pool(query_size, std::vector<IDType>(ef));
    std::vector<std::vector<DistanceType>> dist_pool;
    
    for (uint32_t i = 0; i < query_size; i++) {
      auto cur_query = query_ptr + i * query_dim;
      if constexpr (std::is_same<SearchSpaceType, BuildSpaceType>::
                        value) {  // don't need rerank and need to get distance
        dist_pool =
            std::vector<std::vector<DistanceType>>(query_size, std::vector<DistanceType>(ef));
        search_job_->search_solo(cur_query, topk, res_pool[i].data(), dist_pool[i].data(), ef);
      } else {
        search_job_->search_solo(cur_query, topk, res_pool[i].data(), ef);
      }
    }

    LOG_INFO("Total time: {} s.", timer.elapsed() / 1000000.0);

    auto ret = py::array_t<IDType>({query_size, static_cast<size_t>(topk)});
    auto ret_dist = py::array_t<DistanceType>({query_size, static_cast<size_t>(topk)});
    auto ret_ptr = static_cast<IDType *>(ret.request().ptr);
    auto ret_dist_ptr = static_cast<DistanceType *>(ret_dist.request().ptr);
    if constexpr (std::is_same<SearchSpaceType, BuildSpaceType>::value) {
      for (size_t i = 0; i < query_size; i++) {
        std::copy(res_pool[i].begin(), res_pool[i].begin() + topk, ret_ptr + i * topk);
        std::copy(dist_pool[i].begin(), dist_pool[i].begin() + topk, ret_dist_ptr + i * topk);
      }
    } else {
      for (size_t i = 0; i < query_size; i++) {
        rerank(res_pool[i], ret_ptr + i * topk, ret_dist_ptr + i * topk,
               build_space_->get_query_computer(query_ptr + i * query_dim), ef, topk);
      }
    }
    return py::make_tuple(ret, ret_dist);
  }

#if (not defined (__NO_COROUTINE__))
  auto batch_search_with_distance(py::array_t<DataType> &queries, uint32_t topk, uint32_t ef,
                                  uint32_t num_threads) -> py::object {
    auto shape = queries.shape();
    size_t query_size = shape[0];
    size_t query_dim = shape[1];

    auto *query_ptr = static_cast<DataType *>(queries.request().ptr);

    Timer timer{};
    std::vector<std::vector<IDType>> res_pool(query_size, std::vector<IDType>(ef));
    std::vector<std::vector<DistanceType>> dist_pool;

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
      if constexpr (std::is_same<SearchSpaceType, BuildSpaceType>::
                        value) {  // don't need rerank and need to get distance
        dist_pool =
            std::vector<std::vector<DistanceType>>(query_size, std::vector<DistanceType>(ef));
        coros.emplace_back(
            search_job_->search(cur_query, topk, res_pool[i].data(), dist_pool[i].data(), ef));
      } else {
        coros.emplace_back(search_job_->search(cur_query, topk, res_pool[i].data(), ef));
      }

      scheduler->schedule(coros.back().handle());
    }
    LOG_INFO("Scheduling {} tasks.", coros.size());
    scheduler->begin();
    scheduler->join();

    LOG_INFO("Total time: {} s.", timer.elapsed() / 1000000.0);

    auto ret = py::array_t<IDType>({query_size, static_cast<size_t>(topk)});
    auto ret_dist = py::array_t<DistanceType>({query_size, static_cast<size_t>(topk)});
    auto ret_ptr = static_cast<IDType *>(ret.request().ptr);
    auto ret_dist_ptr = static_cast<DistanceType *>(ret_dist.request().ptr);
    if constexpr (std::is_same<SearchSpaceType, BuildSpaceType>::value) {
      for (size_t i = 0; i < query_size; i++) {
        std::copy(res_pool[i].begin(), res_pool[i].begin() + topk, ret_ptr + i * topk);
        std::copy(dist_pool[i].begin(), dist_pool[i].begin() + topk, ret_dist_ptr + i * topk);
      }
    } else {
      for (size_t i = 0; i < query_size; i++) {
        rerank(res_pool[i], ret_ptr + i * topk, ret_dist_ptr + i * topk,
               build_space_->get_query_computer(query_ptr + i * query_dim), ef, topk);
      }
    }
    return py::make_tuple(ret, ret_dist);
  }
#endif

  template <typename DistanceType = float>
  void rerank(std::vector<IDType> &src, IDType *desc, auto dist_compute, uint32_t ef,
              uint32_t topk) {
    std::priority_queue<std::pair<DistanceType, IDType>,
                        std::vector<std::pair<DistanceType, IDType>>, std::greater<>>
        pq;
    for (size_t i = 0; i < ef; i++) {
      pq.push({dist_compute(src[i]), src[i]});
    }
    for (size_t i = 0; i < topk; i++) {
      desc[i] = pq.top().second;
      pq.pop();
    }
  }

  template <typename DistanceType = float>
  void rerank(std::vector<IDType> &src, IDType *desc, DistanceType *distances, auto dist_compute,
              uint32_t ef, uint32_t topk) {
    std::priority_queue<std::pair<DistanceType, IDType>,
                        std::vector<std::pair<DistanceType, IDType>>, std::greater<>>
        pq;
    for (size_t i = 0; i < ef; i++) {
      pq.push({dist_compute(src[i]), src[i]});
    }
    for (size_t i = 0; i < topk; i++) {
      distances[i] = pq.top().first;
      desc[i] = pq.top().second;
      pq.pop();
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

  std::shared_ptr<alaya::GraphSearchJob<SearchSpaceType>> search_job_{nullptr};
  std::shared_ptr<alaya::GraphUpdateJob<SearchSpaceType>> update_job_{nullptr};
  std::shared_ptr<JobContext<IDType>> job_context_{nullptr};
};

class PyIndexInterface {
 public:
  explicit PyIndexInterface(const IndexParams &params) : params_(params) {  // NOLINT
    DISPATCH_AND_CREATE(params);
  };

  auto to_string() -> std::string { return "PyIndexInterface"; }

  auto fit(py::array &vectors,  // NOLINT
           uint32_t ef_construction, uint32_t num_threads) -> void {
    DISPATCH_AND_CAST_WITH_ARR(vectors, typed_vectors, index,
                               index->fit(typed_vectors, ef_construction, num_threads););
  }

  auto search(py::array &query, uint32_t topk, uint32_t ef) -> py::array {  // NOLINT
    DISPATCH_AND_CAST_WITH_ARR(query, typed_query, index,
                               return index->search(typed_query, topk, ef););
  }

  auto get_data_by_id(uint32_t id) -> py::array {  // NOLINT
    DISPATCH_AND_CAST(index, return index->get_data_by_id(id););
  }

  auto insert(py::array &insert_data, uint32_t ef) -> std::variant<uint32_t, uint64_t> {  // NOLINT
    DISPATCH_AND_CAST_WITH_ARR(insert_data, typed_insert_data, index,
                               return index->insert(typed_insert_data, ef););
  }

  auto remove(uint32_t id) -> void {  // NOLINT
    DISPATCH_AND_CAST(index, index->remove(id););
  }

  auto batch_search(py::array &queries, uint32_t topk, uint32_t ef,  // NOLINT
                    uint32_t num_threads) -> py::array {
    DISPATCH_AND_CAST_WITH_ARR(queries, typed_queries, index,
                               return index->batch_search(typed_queries, topk, ef, num_threads););
  }

  auto batch_search_with_distance(py::array &queries, uint32_t topk, uint32_t ef,  // NOLINT
                                  uint32_t num_threads) -> py::object {
    DISPATCH_AND_CAST_WITH_ARR(
        queries, typed_queries, index,
        return index->batch_search_with_distance(typed_queries, topk, ef, num_threads););
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

  virtual ~PyIndexInterface() = default;
  IndexParams params_;
  std::shared_ptr<BasePyIndex> index_;
};

}  // namespace alaya
