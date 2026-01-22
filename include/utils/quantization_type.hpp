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
enum class QuantizationType {
  NONE = 0,    // none quantization
  SQ8 = 1,     // 8-bit quantization
  SQ4 = 2,     // 4-bit quantization
  RABITQ = 3,  // 1-bit quantization
};
// NOLINTEND

struct QuantizationTypeMap {
  static constexpr std::array<std::tuple<std::string_view, QuantizationType>, 4> kStaticMap = {
      std::make_tuple("NONE", QuantizationType::NONE),
      std::make_tuple("SQ8", QuantizationType::SQ8),        // 8-bit quantization
      std::make_tuple("SQ4", QuantizationType::SQ4),        // 4-bit quantization
      std::make_tuple("RABITQ", QuantizationType::RABITQ),  // 1-bit quantization
  };

  constexpr auto operator[](const std::string_view str) const -> QuantizationType {
    for (const auto &[key, val] : kStaticMap) {
      if (key == str) {
        return val;
      }
    }
    ALAYA_UNREACHABLE;
  }
};

// quantization type to string
struct QuantizationTypeToString {
  static constexpr std::array<std::tuple<QuantizationType, std::string_view>, 4> kStaticMap = {
      std::make_tuple(QuantizationType::NONE, "NONE"),
      std::make_tuple(QuantizationType::SQ8, "SQ8"),        // 8-bit quantization
      std::make_tuple(QuantizationType::SQ4, "SQ4"),        // 4-bit quantization
      std::make_tuple(QuantizationType::RABITQ, "RABITQ"),  // 1-bit quantization
  };

  constexpr auto operator[](QuantizationType quantization_type) const -> std::string_view {
    for (const auto &[key, val] : kStaticMap) {
      if (key == quantization_type) {
        return val;
      }
    }
    ALAYA_UNREACHABLE;
  }
};

inline constexpr QuantizationTypeMap kQuantizationType{};
inline constexpr QuantizationTypeToString kQuantizationType2str{};

static_assert(kQuantizationType["SQ8"] == QuantizationType::SQ8);
static_assert(kQuantizationType["SQ4"] == QuantizationType::SQ4);
static_assert(kQuantizationType["RABITQ"] == QuantizationType::RABITQ);

}  // namespace alaya
