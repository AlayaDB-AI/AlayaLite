// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <array>
#include <string_view>
#include <tuple>
#include "utils/platform.hpp"

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

  constexpr auto operator[](const std::string_view str) const -> IndexType {
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
