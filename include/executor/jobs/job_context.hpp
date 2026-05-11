// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace alaya {
template <typename IDType>
struct JobContext {
  std::unordered_map<IDType, std::vector<IDType>> inserted_edges_;
  std::unordered_set<IDType> removed_vertices_;
  std::unordered_map<IDType, std::vector<IDType>> removed_node_nbrs_;
};
}  // namespace alaya
