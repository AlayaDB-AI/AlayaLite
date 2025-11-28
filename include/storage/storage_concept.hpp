/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
