// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file disk_update_context.hpp
 * @brief Transient bookkeeping for in-place DiskANN updates.
 *
 * Tracks reverse edges for reconnect and cached old neighbors for two-hop
 * bypass through tombstoned nodes. Reused slots must call forget_slot() to
 * evict stale two-hop data.
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace alaya::diskann {

struct DiskUpdateContext {
  /// Reverse edges queued for a node's next reconnect (node id -> new neighbor ids).
  std::unordered_map<uint32_t, std::vector<uint32_t>> inserted_edges_;
  /// Old neighbor list of each deleted node, for two-hop bypass.
  std::unordered_map<uint32_t, std::vector<uint32_t>> removed_node_nbrs_;

  /// Drop all transient state (e.g. after flush, or on teardown).
  void clear() {
    inserted_edges_.clear();
    removed_node_nbrs_.clear();
  }

  /// Forget a slot that has just been reused by an insert: its cached two-hop
  /// data no longer describes the node now living in the slot.
  void forget_slot(uint32_t slot) { removed_node_nbrs_.erase(slot); }

  /// Fraction of @p total slots that are currently tombstoned (0 when total==0).
  [[nodiscard]] double tombstone_ratio(uint64_t total) const {
    if (total == 0) {
      return 0.0;
    }
    return static_cast<double>(removed_node_nbrs_.size()) / static_cast<double>(total);
  }

  /// True when the safety-net proactive reconnect should fire: the tombstone
  /// ratio has reached @p ratio_threshold AND no insert-driven reconnect has run
  /// for at least @p ops_threshold operations (delete-heavy workload).
  [[nodiscard]] bool needs_safety_net_reconnect(double ratio_threshold,
                                                uint64_t total,
                                                uint64_t ops_since_last_insert,
                                                uint64_t ops_threshold) const {
    return ops_since_last_insert >= ops_threshold && tombstone_ratio(total) >= ratio_threshold;
  }
};

}  // namespace alaya::diskann
