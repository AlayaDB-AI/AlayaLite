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
