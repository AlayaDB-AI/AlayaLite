// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <concepts>
#include <cstdint>

namespace alaya {

struct StorageContent {};

template <typename T>
concept DataStorage = requires(T t, typename T::id_type i, typename T::data_type *data) {
  { t[i] } -> std::same_as<typename T::data_type *>;
  { t.is_valid(i) } -> std::same_as<bool>;
  { t.insert(data) } -> std::same_as<typename T::id_type>;
  { t.remove(i) } -> std::same_as<typename T::id_type>;
  { t.update(i, data) } -> std::same_as<typename T::id_type>;
};  // NOLINT

}  // namespace alaya
