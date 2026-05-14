// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#ifndef ALAYA_THIS_SHARD_INSTANTIATIONS
  #error "ALAYA_THIS_SHARD_INSTANTIATIONS must be defined before this header"
#endif

// Single-TU shards instantiate both core and search method bodies together.
#include "impl/index_impl_core.hpp"
#include "impl/index_impl_search.hpp"
#include "instantiations.hpp"

namespace alaya {

#define ALAYA_DEFINE_PYINDEX(GraphBuilderT, SearchSpaceT) \
  template class PyIndex<ALAYA_PYINDEX_TYPE GraphBuilderT, ALAYA_PYINDEX_TYPE SearchSpaceT>;

ALAYA_THIS_SHARD_INSTANTIATIONS(ALAYA_DEFINE_PYINDEX)

#undef ALAYA_DEFINE_PYINDEX

}  // namespace alaya

#undef ALAYA_THIS_SHARD_INSTANTIATIONS
