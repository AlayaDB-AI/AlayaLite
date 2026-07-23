// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace alaya {

/**
 * @brief Query-facing contract for canonical persisted records.
 *
 * Raw payload access supports selected-field residual evaluation without coupling query code to a
 * storage engine. Mutation and generation publication remain outside this Phase 1 interface.
 */
template <typename IDType>
class RecordStore {
 public:
  /** @brief Destroy the record-store implementation. */
  virtual ~RecordStore() = default;

  /** @brief Read one serialized scalar record. */
  virtual auto get_raw_scalar(IDType id, std::string &value) const -> bool = 0;

  /** @brief Batch-read serialized scalar records; missing rows are represented by empty strings. */
  [[nodiscard]] virtual auto batch_get_raw_scalars(const std::vector<IDType> &ids) const
      -> std::vector<std::string> = 0;

  /** @brief Resolve an external item ID to its internal ID. */
  [[nodiscard]] virtual auto find_by_item_id(const std::string &item_id) const
      -> std::optional<IDType> = 0;

  /** @brief Return the number of persisted scalar records visible to this store. */
  [[nodiscard]] virtual auto size() const -> size_t = 0;

  /** @brief Return the atomic mutation generation visible to record reads. */
  [[nodiscard]] virtual auto generation() const -> uint64_t = 0;
};

}  // namespace alaya
