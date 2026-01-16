/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <array>
#include <string_view>
#include <tuple>

#if defined(_MSC_VER)
#define ALAYA_UNREACHABLE __assume(0)
#elif defined(__GNUC__) || defined(__clang__)
#define ALAYA_UNREACHABLE __builtin_unreachable()
#else
#define ALAYA_UNREACHABLE
#endif

namespace alaya {

// NOLINTBEGIN
enum class IndexType {
  FLAT = 0,
  HNSW = 1,
  NSG = 2,
  FUSION = 3,
  QG = 4,
};
// NOLINTEND

struct IndexTypeMap {
  static constexpr std::array<std::tuple<std::string_view, IndexType>, 3> kStaticMap = {
      std::make_tuple("FLAT", IndexType::FLAT),
      std::make_tuple("HNSW", IndexType::HNSW),
      std::make_tuple("QG", IndexType::QG),
  };

  constexpr auto operator[](const std::string_view &str) const -> IndexType {
    for (const auto &[key, val] : kStaticMap) {
      if (key == str) {
        return val;
      }
    }
    ALAYA_UNREACHABLE;
  }
};

// index type to string
struct IndexTypeToString {
  static constexpr std::array<std::tuple<IndexType, std::string_view>, 3> kStaticMap = {
      std::make_tuple(IndexType::FLAT, "FLAT"),
      std::make_tuple(IndexType::HNSW, "HNSW"),
      std::make_tuple(IndexType::QG, "QG"),
  };

  constexpr auto operator[](IndexType index_type) const -> std::string_view {
    for (const auto &[key, val] : kStaticMap) {
      if (key == index_type) {
        return val;
      }
    }
    ALAYA_UNREACHABLE;
  }
};

inline constexpr IndexTypeMap kIndexType{};
inline constexpr IndexTypeToString kIndexType2str{};

static_assert(kIndexType["HNSW"] == IndexType::HNSW);
static_assert(kIndexType["QG"] == IndexType::QG);

}  // namespace alaya
