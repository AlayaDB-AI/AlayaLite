// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#ifndef ALAYA_THIS_SHARD_INSTANTIATIONS
  #error "ALAYA_THIS_SHARD_INSTANTIATIONS must be defined before this header"
#endif

// Search shard TUs instantiate query and hybrid-search method bodies.
#include "impl/index_impl_search.hpp"
#include "instantiations.hpp"

namespace alaya {

#define ALAYA_INSTANTIATE_PYINDEX_SEARCH_METHODS(GraphBuilderT, SearchSpaceT)                     \
  template void PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT, ALAYA_PYINDEX_TYPE SearchSpaceT>::      \
      execute_hybrid_search_dispatch(const ALAYA_PYINDEX_TYPE SearchSpaceT::DataTypeAlias *,      \
                                     ALAYA_PYINDEX_TYPE SearchSpaceT::IDTypeAlias *,              \
                                     const SearchInfo &,                                          \
                                     const MetadataFilter &,                                      \
                                     bool,                                                        \
                                     std::string *) const;                                        \
  template auto PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT,                                         \
                        ALAYA_PYINDEX_TYPE SearchSpaceT>::get_hybrid_batch_pool(uint32_t threads) \
      -> std::shared_ptr<alaya::ThreadPool>;                                                      \
  template auto PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT,                                         \
                        ALAYA_PYINDEX_TYPE SearchSpaceT>::contains(const std::string &) -> bool;  \
  template auto                                                                                   \
  PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT,                                                       \
          ALAYA_PYINDEX_TYPE SearchSpaceT>::get_scalar_data_by_item_id(const std::string &)       \
      -> py::dict;                                                                                \
  template auto PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT, ALAYA_PYINDEX_TYPE SearchSpaceT>::      \
      get_scalar_data_by_internal_id(ALAYA_PYINDEX_TYPE SearchSpaceT::IDTypeAlias) -> py::dict;   \
  template auto PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT, ALAYA_PYINDEX_TYPE SearchSpaceT>::      \
      batch_get_scalar_data_by_internal_ids(                                                      \
          py::array_t<ALAYA_PYINDEX_TYPE SearchSpaceT::IDTypeAlias>)                              \
          ->py::list;                                                                             \
  template auto PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT, ALAYA_PYINDEX_TYPE SearchSpaceT>::      \
      batch_get_item_ids_by_internal_ids(                                                         \
          py::array_t<ALAYA_PYINDEX_TYPE SearchSpaceT::IDTypeAlias>)                              \
          ->py::list;                                                                             \
  template auto                                                                                   \
  PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT, ALAYA_PYINDEX_TYPE SearchSpaceT>::get_data_num()      \
      -> ALAYA_PYINDEX_TYPE SearchSpaceT::IDTypeAlias;                                            \
  template auto PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT, ALAYA_PYINDEX_TYPE SearchSpaceT>::      \
      search(py::array_t<ALAYA_PYINDEX_TYPE SearchSpaceT::DataTypeAlias>, uint32_t, uint32_t)     \
          -> py::array_t<ALAYA_PYINDEX_TYPE SearchSpaceT::IDTypeAlias>;                           \
  template auto PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT, ALAYA_PYINDEX_TYPE SearchSpaceT>::      \
      search_with_distance(py::array_t<ALAYA_PYINDEX_TYPE SearchSpaceT::DataTypeAlias>,           \
                           uint32_t,                                                              \
                           uint32_t) -> py::object;                                               \
  template auto PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT, ALAYA_PYINDEX_TYPE SearchSpaceT>::      \
      hybrid_search(py::array_t<ALAYA_PYINDEX_TYPE SearchSpaceT::DataTypeAlias>,                  \
                    uint32_t,                                                                     \
                    uint32_t,                                                                     \
                    const MetadataFilter &,                                                       \
                    bool,                                                                         \
                    const std::string &) -> py::object;                                           \
  template auto PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT, ALAYA_PYINDEX_TYPE SearchSpaceT>::      \
      batch_hybrid_search(py::array_t<ALAYA_PYINDEX_TYPE SearchSpaceT::DataTypeAlias>,            \
                          uint32_t,                                                               \
                          uint32_t,                                                               \
                          const MetadataFilter &,                                                 \
                          uint32_t,                                                               \
                          bool,                                                                   \
                          const std::string &) -> py::object;                                     \
  template auto PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT,                                         \
                        ALAYA_PYINDEX_TYPE SearchSpaceT>::filter_query(const MetadataFilter &,    \
                                                                       uint32_t) -> py::object;   \
  template auto PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT, ALAYA_PYINDEX_TYPE SearchSpaceT>::      \
      batch_search(py::array_t<ALAYA_PYINDEX_TYPE SearchSpaceT::DataTypeAlias>,                   \
                   uint32_t,                                                                      \
                   uint32_t,                                                                      \
                   uint32_t) -> py::array_t<ALAYA_PYINDEX_TYPE SearchSpaceT::IDTypeAlias>;        \
  template auto PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT, ALAYA_PYINDEX_TYPE SearchSpaceT>::      \
      batch_search_with_distance(py::array_t<ALAYA_PYINDEX_TYPE SearchSpaceT::DataTypeAlias>,     \
                                 uint32_t,                                                        \
                                 uint32_t,                                                        \
                                 uint32_t) -> py::object;                                         \
  template auto                                                                                   \
  PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT, ALAYA_PYINDEX_TYPE SearchSpaceT>::close_db() -> void;

#if defined(__linux__)
  #define ALAYA_INSTANTIATE_PYINDEX_SEARCH_TASK_METHOD(GraphBuilderT, SearchSpaceT)              \
    template auto PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT, ALAYA_PYINDEX_TYPE SearchSpaceT>::   \
        execute_hybrid_search_dispatch_task(const ALAYA_PYINDEX_TYPE SearchSpaceT::DataTypeAlias \
                                                *,                                               \
                                            ALAYA_PYINDEX_TYPE SearchSpaceT::IDTypeAlias *,      \
                                            SearchInfo,                                          \
                                            const MetadataFilter &,                              \
                                            bool,                                                \
                                            std::string *) const -> coro::task<>;
#endif

ALAYA_THIS_SHARD_INSTANTIATIONS(ALAYA_INSTANTIATE_PYINDEX_SEARCH_METHODS)
#if defined(__linux__)
ALAYA_THIS_SHARD_INSTANTIATIONS(ALAYA_INSTANTIATE_PYINDEX_SEARCH_TASK_METHOD)
  #undef ALAYA_INSTANTIATE_PYINDEX_SEARCH_TASK_METHOD
#endif

#undef ALAYA_INSTANTIATE_PYINDEX_SEARCH_METHODS

}  // namespace alaya

#undef ALAYA_THIS_SHARD_INSTANTIATIONS
