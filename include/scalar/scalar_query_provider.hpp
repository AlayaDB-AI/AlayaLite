// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "scalar/scalar_index.hpp"
#include "storage/record_store.hpp"
#include "utils/query_utils.hpp"

namespace alaya {

/** @brief Generation-stable scalar providers retained for one hybrid query. */
template <typename IDType>
class ScalarQueryView {
 public:
  /** @brief Destroy the query view and release any storage-engine snapshot. */
  virtual ~ScalarQueryView() = default;

  /** @brief Return immutable query-hot secondary indexes. */
  [[nodiscard]] virtual auto scalar_index() const -> const ScalarIndex<IDType> & = 0;

  /** @brief Return canonical rows used for residual predicate evaluation. */
  [[nodiscard]] virtual auto record_store() const -> const RecordStore<IDType> & = 0;

  /** @brief Return the internal-ID universe shared with the vector backend. */
  [[nodiscard]] virtual auto universe_size() const -> size_t = 0;

  /** @brief Return the number of live records inside the stable ID universe. */
  [[nodiscard]] virtual auto live_count() const -> size_t = 0;

  /** @brief Return set bits for live IDs; holes must be blocked even for an empty filter. */
  [[nodiscard]] virtual auto live_mask() const -> const DynamicBitset & = 0;

  /** @brief Resolve result IDs to external item IDs through the same read generation. */
  [[nodiscard]] virtual auto batch_get_item_ids(const std::vector<IDType> &ids) const
      -> std::vector<std::string> = 0;
};

/** @brief Factory that pins a consistent scalar read generation for each hybrid query. */
template <typename IDType>
class ScalarQueryProvider {
 public:
  /** @brief Destroy the scalar query provider. */
  virtual ~ScalarQueryProvider() = default;

  /** @brief Acquire one scalar index/record store pair at the same generation. */
  [[nodiscard]] virtual auto acquire() const -> std::unique_ptr<ScalarQueryView<IDType>> = 0;
};

}  // namespace alaya
