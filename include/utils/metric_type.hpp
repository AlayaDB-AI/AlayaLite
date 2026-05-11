// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <array>
#include <string_view>
#include <tuple>

namespace alaya {

// NOLINTBEGIN
enum class MetricType {
  L2,
  IP,
  COS,
  NONE,
};
// NOLINTEND

struct MetricMap {
  static constexpr std::array<std::tuple<std::string_view, MetricType>, 3> kStaticMap = {
      std::make_tuple("L2", MetricType::L2),
      std::make_tuple("IP", MetricType::IP),
      std::make_tuple("COS", MetricType::COS),
  };

  constexpr auto operator[](const std::string_view str) const -> MetricType {
    for (const auto &[key, val] : kStaticMap) {
      if (key == str) {
        return val;
      }
    }
    return MetricType::NONE;
  }
};

inline constexpr MetricMap kMetricMap{};

static_assert(kMetricMap["L2"] == MetricType::L2);

}  // namespace alaya
