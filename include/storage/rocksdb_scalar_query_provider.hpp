// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "scalar/scalar_query_provider.hpp"
#include "storage/rocksdb_record_store.hpp"
#include "utils/scalar_data.hpp"

namespace alaya {

/** @brief Exposes RocksDBRecordStore query snapshots through ScalarQueryProvider. */
template <typename IDType>
class RocksDBScalarQueryProvider final : public ScalarQueryProvider<IDType> {
 public:
  /** @brief Bind to a v2 store whose lifetime exceeds this provider and its active views. */
  explicit RocksDBScalarQueryProvider(const RocksDBRecordStore<IDType> *store) : store_(store) {
    if (store_ == nullptr) {
      throw std::invalid_argument("RocksDBRecordStore cannot be null");
    }
  }

  /** @copydoc ScalarQueryProvider::acquire */
  [[nodiscard]] auto acquire() const -> std::unique_ptr<ScalarQueryView<IDType>> override {
    return std::make_unique<View>(store_->acquire_query_view());
  }

 private:
  class View final : public ScalarQueryView<IDType> {
   public:
    /** @brief Own one generation-stable RocksDB query view. */
    explicit View(std::unique_ptr<typename RocksDBRecordStore<IDType>::QueryView> view)
        : view_(std::move(view)) {}

    /** @copydoc ScalarQueryView::scalar_index */
    [[nodiscard]] auto scalar_index() const -> const ScalarIndex<IDType> & override {
      return view_->scalar_index();
    }

    /** @copydoc ScalarQueryView::record_store */
    [[nodiscard]] auto record_store() const -> const RecordStore<IDType> & override {
      return *view_;
    }

    /** @copydoc ScalarQueryView::universe_size */
    [[nodiscard]] auto universe_size() const -> size_t override {
      return view_->scalar_index().universe_size();
    }

    /** @copydoc ScalarQueryView::live_count */
    [[nodiscard]] auto live_count() const -> size_t override {
      return view_->scalar_index().live_count();
    }

    /** @copydoc ScalarQueryView::live_mask */
    [[nodiscard]] auto live_mask() const -> const DynamicBitset & override {
      return view_->scalar_index().live_mask();
    }

    /** @copydoc ScalarQueryView::batch_get_item_ids */
    [[nodiscard]] auto batch_get_item_ids(const std::vector<IDType> &ids) const
        -> std::vector<std::string> override {
      auto raw_scalars = view_->batch_get_raw_scalars(ids);
      std::vector<std::string> item_ids;
      item_ids.reserve(raw_scalars.size());
      for (const auto &raw : raw_scalars) {
        if (raw.empty()) {
          item_ids.emplace_back();
        } else {
          item_ids.push_back(ScalarData::deserialize(raw.data(), raw.size()).item_id);
        }
      }
      return item_ids;
    }

   private:
    std::unique_ptr<typename RocksDBRecordStore<IDType>::QueryView>
        view_;  ///< Owns paired RocksDB and scalar-index snapshots for one query.
  };

  const RocksDBRecordStore<IDType> *store_;  ///< Non-owning v2 store backing acquired views.
};

}  // namespace alaya
