// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

#include "search/vector_search_backend.hpp"
#include "utils/query_utils.hpp"

namespace alaya {

/**
 * @brief Adapts an owned blocked-ID bitset to the allow-style IdMask contract.
 *
 * Set bits are rejected. The shared ownership keeps the immutable query snapshot alive for every
 * backend call or candidate cursor that references this mask. IDs outside the snapshot universe
 * are rejected because they cannot belong to the scalar generation that produced the mask.
 */
template <typename IDType>
class BlockedBitsetIdMask final : public IdMask<IDType> {
 public:
  /** @brief Bind the mask to an immutable blocked-ID snapshot. */
  explicit BlockedBitsetIdMask(std::shared_ptr<const DynamicBitset> blocked)
      : blocked_(std::move(blocked)) {
    if (blocked_ == nullptr) {
      throw std::invalid_argument("Blocked bitset cannot be null");
    }
  }

  /** @copydoc IdMask::accepts */
  [[nodiscard]] auto accepts(IDType id) const -> bool override {
    auto raw_id = static_cast<size_t>(id);
    return raw_id < blocked_->size() && !blocked_->get(raw_id);
  }

  /** @brief Return the internal-ID universe captured by this mask. */
  [[nodiscard]] auto size() const -> size_t { return blocked_->size(); }

 private:
  std::shared_ptr<const DynamicBitset> blocked_;  ///< Immutable blocked IDs owned by this mask.
};

}  // namespace alaya
