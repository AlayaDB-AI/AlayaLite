// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "index.hpp"

namespace alaya {

template <typename GraphBuilderType, typename SearchSpaceType>
void PyIndex<GraphBuilderType, SearchSpaceType>::execute_hybrid_search_dispatch(
    const DataType *query,
    IDType *ids,
    const SearchInfo &search_info,
    const MetadataFilter &filter,
    bool brute_force_requested,
    std::string *item_ids) const {
  if constexpr (!SearchSpaceType::has_scalar_data) {
    (void)query;
    (void)ids;
    (void)search_info;
    (void)filter;
    (void)brute_force_requested;
    (void)item_ids;
    throw std::runtime_error("hybrid_search dispatch requires a space that supports scalar data");
  } else {
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
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::get_hybrid_batch_pool(uint32_t requested_threads)
    -> std::shared_ptr<alaya::ThreadPool> {
  auto effective_threads = std::max<uint32_t>(1, requested_threads);
  std::lock_guard<std::mutex> lock(hybrid_batch_pool_mutex_);
  if (hybrid_batch_pool_ == nullptr || hybrid_batch_pool_threads_ != effective_threads) {
    hybrid_batch_pool_ = std::make_shared<alaya::ThreadPool>(effective_threads);
    hybrid_batch_pool_threads_ = effective_threads;
  }
  return hybrid_batch_pool_;
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::execute_hybrid_search_dispatch_task(
    const DataType *query,
    IDType *ids,
    SearchInfo search_info,
    const MetadataFilter &filter,
    bool brute_force_requested,
    std::string *item_ids) const -> coro::task<> {
  execute_hybrid_search_dispatch(query, ids, search_info, filter, brute_force_requested, item_ids);
  co_return;
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::contains(const std::string &item_id) -> bool {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::get_scalar_data_by_item_id(
    const std::string &item_id) -> py::dict {
  if constexpr (!SearchSpaceType::has_scalar_data) {
    throw std::runtime_error("get_scalar_data requires a space that supports scalar data");
  } else {
    auto [internal_id, scalar_data] = search_space_->get_scalar_data(item_id);
    py::dict result = scalar_data_to_pydict(scalar_data);
    result["internal_id"] = internal_id;
    return result;
  }
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::get_scalar_data_by_internal_id(IDType internal_id)
    -> py::dict {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::batch_get_scalar_data_by_internal_ids(
    py::array_t<IDType> internal_ids) -> py::list {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::batch_get_item_ids_by_internal_ids(
    py::array_t<IDType> internal_ids) -> py::list {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::get_data_num() -> IDType {
  if (build_space_ != nullptr) {
    return build_space_->get_data_num();
  } else if (search_space_ != nullptr) {
    return search_space_->get_data_num();
  }
  return 0;
}

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::search(py::array_t<DataType> query,
                                                        uint32_t topk,
                                                        uint32_t ef) -> py::array_t<IDType> {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::search_with_distance(py::array_t<DataType> query,
                                                                      uint32_t topk,
                                                                      uint32_t ef) -> py::object {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::hybrid_search(py::array_t<DataType> query,
                                                               uint32_t topk,
                                                               uint32_t ef,
                                                               const MetadataFilter &filter,
                                                               bool bf,
                                                               const std::string &filter_exec_hint)
    -> py::object {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::batch_hybrid_search(
    py::array_t<DataType> queries,
    uint32_t topk,
    uint32_t ef,
    const MetadataFilter &filter,
    uint32_t num_threads,
    bool bf,
    const std::string &filter_exec_hint) -> py::object {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::filter_query(const MetadataFilter &filter,
                                                              uint32_t limit) -> py::object {
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::batch_search(py::array_t<DataType> queries,
                                                              uint32_t topk,
                                                              uint32_t ef,
                                                              uint32_t num_threads)
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
    LOG_INFO_ONCE(
        "search fallback: coroutine batch search is unavailable on this platform, using "
        "synchronous search path");
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::batch_search_with_distance(
    py::array_t<DataType> queries,
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
    LOG_INFO_ONCE(
        "search fallback: coroutine distance batch search is unavailable on this platform, using "
        "synchronous search path");
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

template <typename GraphBuilderType, typename SearchSpaceType>
auto PyIndex<GraphBuilderType, SearchSpaceType>::close_db() -> void {
  if (search_space_ != nullptr) {
    search_space_->close_db();
  }
}

}  // namespace alaya
