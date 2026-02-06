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

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "scalar_data.hpp"

namespace alaya {

/**
 * @brief Utilities for encoding metadata values to sortable strings for indexing.
 *
 * These encodings ensure that RocksDB's lexicographic ordering matches the
 * natural ordering of the values (including negative numbers and floats).
 */
namespace index_encoding {

/**
 * @brief Encode int64 to sortable hex string (handles negative numbers)
 *
 * Uses XOR with sign bit to convert signed to unsigned while preserving order.
 * Result: -∞ → "0000..." < 0 → "8000..." < +∞ → "ffff..."
 */
inline auto encode_int64(int64_t value) -> std::string {
  uint64_t encoded = static_cast<uint64_t>(value) ^ (1ULL << 63);
  char buf[17];
  snprintf(buf, sizeof(buf), "%016lx", encoded);
  return {buf};
}

/**
 * @brief Decode sortable hex string back to int64
 */
inline auto decode_int64(const std::string &s) -> int64_t {
  uint64_t encoded = std::stoull(s, nullptr, 16);
  return static_cast<int64_t>(encoded ^ (1ULL << 63));
}

/**
 * @brief Encode double to sortable hex string
 *
 * Uses IEEE 754 bit manipulation to create sortable representation.
 * Positive numbers: flip sign bit
 * Negative numbers: flip all bits
 */
inline auto encode_double(double value) -> std::string {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits >> 63) != 0U) {
    bits = ~bits;  // Negative: flip all bits
  } else {
    bits ^= (1ULL << 63);  // Positive: flip sign bit
  }
  char buf[17];
  snprintf(buf, sizeof(buf), "%016lx", bits);
  return {buf};
}

/**
 * @brief Decode sortable hex string back to double
 */
inline auto decode_double(const std::string &s) -> double {
  uint64_t bits = std::stoull(s, nullptr, 16);
  if ((bits >> 63) != 0U) {
    bits ^= (1ULL << 63);  // Was positive
  } else {
    bits = ~bits;  // Was negative
  }
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

/**
 * @brief Convert MetadataValue to encoded string for index key
 *
 * Format: "{type_prefix}_{encoded_value}"
 * - String: "s_{value}"
 * - Bool:   "b_{0|1}"
 * - Int64:  "i_{hex16}"
 * - Double: "d_{hex16}"
 */
inline auto encode_value(const MetadataValue &value) -> std::string {
  return std::visit(
      [](const auto &v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
          return "s_" + v;
        } else if constexpr (std::is_same_v<T, bool>) {
          return std::string("b_") + (v ? "1" : "0");
        } else if constexpr (std::is_same_v<T, int64_t>) {
          return "i_" + encode_int64(v);
        } else if constexpr (std::is_same_v<T, double>) {
          return "d_" + encode_double(v);
        } else {
          return "u_unknown";
        }
      },
      value);
}

/**
 * @brief Generate field index key: f_{field}_{encoded_value}_{id}
 */
template <typename IDType>
inline auto make_field_index_key(const std::string &field,
                                 const std::string &encoded_value,
                                 IDType id) -> std::string {
  return "f_" + field + "_" + encoded_value + "_" + std::to_string(id);
}

/**
 * @brief Generate field index prefix for exact match: f_{field}_{encoded_value}_
 */
inline auto make_field_index_prefix(const std::string &field, const std::string &encoded_value)
    -> std::string {
  return "f_" + field + "_" + encoded_value + "_";
}

/**
 * @brief Generate field prefix for range scan: f_{field}_
 */
inline auto make_field_prefix(const std::string &field) -> std::string {
  return "f_" + field + "_";
}

/**
 * @brief Extract ID from field index key (last component after underscore)
 */
template <typename IDType>
inline auto extract_id_from_key(const std::string &key) -> IDType {
  auto last_underscore = key.rfind('_');
  if (last_underscore == std::string::npos) {
    return 0;
  }
  return static_cast<IDType>(std::stoul(key.substr(last_underscore + 1)));
}

}  // namespace index_encoding
}  // namespace alaya
