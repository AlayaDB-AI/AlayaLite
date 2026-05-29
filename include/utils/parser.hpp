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
#include <cctype>
#include <charconv>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "utils/scalar_data.hpp"

namespace alaya {

inline auto trim(std::string_view value) -> std::string_view {
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.remove_prefix(1);
  }
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

inline auto parse_u64(std::string_view value) -> uint64_t {
  value = trim(value);
  uint64_t result = 0;
  auto *begin = value.data();
  auto *end = value.data() + value.size();
  auto [ptr, ec] = std::from_chars(begin, end, result);
  if (ec != std::errc() || ptr != end) {
    throw std::runtime_error("invalid unsigned integer: " + std::string(value));
  }
  return result;
}

struct NpyFloatMatrix {
  std::vector<float> data_;
  uint32_t rows_ = 0;
  uint32_t cols_ = 0;
};

inline auto parse_npy_shape(const std::string &header) -> std::pair<uint32_t, uint32_t> {
  auto shape_pos = header.find("'shape'");
  if (shape_pos == std::string::npos) {
    shape_pos = header.find("\"shape\"");
  }
  if (shape_pos == std::string::npos) {
    throw std::runtime_error("NPY header has no shape field");
  }

  auto open = header.find('(', shape_pos);
  auto comma = header.find(',', open);
  auto close = header.find(')', comma);
  if (open == std::string::npos || comma == std::string::npos || close == std::string::npos) {
    throw std::runtime_error("NPY header has unsupported shape");
  }

  auto rows = parse_u64(std::string_view(header).substr(open + 1, comma - open - 1));
  auto cols = parse_u64(std::string_view(header).substr(comma + 1, close - comma - 1));
  if (rows > std::numeric_limits<uint32_t>::max() || cols > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("NPY matrix is too large for uint32_t dimensions");
  }
  return {static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)};
}

inline auto read_little_endian_u16(std::ifstream &reader) -> uint16_t {
  std::array<unsigned char, 2> bytes{};
  reader.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (reader.fail()) {
    throw std::runtime_error("Failed to read NPY header length");
  }
  return static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8);
}

inline auto read_little_endian_u32(std::ifstream &reader) -> uint32_t {
  std::array<unsigned char, 4> bytes{};
  reader.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (reader.fail()) {
    throw std::runtime_error("Failed to read NPY header length");
  }
  return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
}

inline auto load_npy_float_matrix(const std::filesystem::path &filepath, uint32_t max_rows = 0)
    -> NpyFloatMatrix {
  std::ifstream reader(filepath, std::ios::binary);
  if (!reader.is_open()) {
    throw std::runtime_error("Open npy file error: " + filepath.string());
  }

  std::array<char, 6> magic{};
  reader.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (std::string_view(magic.data(), magic.size()) != std::string_view("\x93NUMPY", 6)) {
    throw std::runtime_error("Invalid npy magic: " + filepath.string());
  }

  uint8_t major = 0;
  uint8_t minor = 0;
  reader.read(reinterpret_cast<char *>(&major), sizeof(major));
  reader.read(reinterpret_cast<char *>(&minor), sizeof(minor));
  (void)minor;

  uint32_t header_len = 0;
  if (major == 1) {
    header_len = read_little_endian_u16(reader);
  } else if (major == 2 || major == 3) {
    header_len = read_little_endian_u32(reader);
  } else {
    throw std::runtime_error("Unsupported npy version in " + filepath.string());
  }

  std::string header(header_len, '\0');
  reader.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (header.find("'descr': '<f4'") == std::string::npos &&
      header.find("\"descr\": \"<f4\"") == std::string::npos &&
      header.find("'descr': '|f4'") == std::string::npos &&
      header.find("\"descr\": \"|f4\"") == std::string::npos) {
    throw std::runtime_error("Only float32 npy matrices are supported: " + filepath.string());
  }
  if (header.find("'fortran_order': True") != std::string::npos ||
      header.find("\"fortran_order\": true") != std::string::npos) {
    throw std::runtime_error("Fortran-order npy matrices are not supported: " + filepath.string());
  }

  auto [rows, cols] = parse_npy_shape(header);
  auto rows_to_read = rows;
  if (max_rows > 0 && rows_to_read > max_rows) {
    rows_to_read = max_rows;
  }

  NpyFloatMatrix matrix;
  matrix.rows_ = rows_to_read;
  matrix.cols_ = cols;
  matrix.data_.resize(static_cast<size_t>(matrix.rows_) * matrix.cols_);
  reader.read(reinterpret_cast<char *>(matrix.data_.data()),
              static_cast<std::streamsize>(matrix.data_.size() * sizeof(float)));
  if (reader.fail()) {
    throw std::runtime_error("Failed to read npy matrix data: " + filepath.string());
  }
  return matrix;
}

struct JsonValue {
  using Array = std::vector<JsonValue>;
  using Object = std::unordered_map<std::string, JsonValue>;
  using ArrayPtr = std::shared_ptr<Array>;
  using ObjectPtr = std::shared_ptr<Object>;

  JsonValue() : value_(nullptr) {}
  explicit JsonValue(std::nullptr_t) : value_(nullptr) {}
  explicit JsonValue(bool value) : value_(value) {}
  explicit JsonValue(int64_t value) : value_(value) {}
  explicit JsonValue(double value) : value_(value) {}
  explicit JsonValue(std::string value) : value_(std::move(value)) {}
  explicit JsonValue(Array value) : value_(std::make_shared<Array>(std::move(value))) {}
  explicit JsonValue(Object value) : value_(std::make_shared<Object>(std::move(value))) {}

  std::variant<std::nullptr_t, bool, int64_t, double, std::string, ArrayPtr, ObjectPtr> value_;
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view input) : input_(input) {}

  auto parse() -> JsonValue {
    skip_ws();
    auto result = parse_value();
    skip_ws();
    if (pos_ != input_.size()) {
      throw std::runtime_error("unexpected trailing JSON content");
    }
    return result;
  }

 private:
  [[nodiscard]] auto peek() const -> char {
    if (pos_ >= input_.size()) {
      return '\0';
    }
    return input_[pos_];
  }

  auto consume(char expected) -> bool {
    if (peek() != expected) {
      return false;
    }
    ++pos_;
    return true;
  }

  void expect(char expected) {
    if (!consume(expected)) {
      throw std::runtime_error(std::string("expected JSON char: ") + expected);
    }
  }

  void skip_ws() {
    while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_])) != 0) {
      ++pos_;
    }
  }

  static auto hex_value(char ch) -> uint32_t {
    if (ch >= '0' && ch <= '9') {
      return static_cast<uint32_t>(ch - '0');
    }
    if (ch >= 'a' && ch <= 'f') {
      return static_cast<uint32_t>(ch - 'a' + 10);
    }
    if (ch >= 'A' && ch <= 'F') {
      return static_cast<uint32_t>(ch - 'A' + 10);
    }
    throw std::runtime_error("invalid JSON unicode escape");
  }

  static auto is_high_surrogate(uint32_t code_unit) -> bool {
    return code_unit >= 0xD800 && code_unit <= 0xDBFF;
  }

  static auto is_low_surrogate(uint32_t code_unit) -> bool {
    return code_unit >= 0xDC00 && code_unit <= 0xDFFF;
  }

  static void append_utf8(std::string &output, uint32_t codepoint) {
    if (codepoint > 0x10FFFF || is_high_surrogate(codepoint) || is_low_surrogate(codepoint)) {
      throw std::runtime_error("invalid JSON unicode codepoint");
    }
    if (codepoint <= 0x7F) {
      output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
      output.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
      output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
      output.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
      output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
      output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
      output.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
      output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
      output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
      output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
  }

  auto parse_unicode_escape() -> uint32_t {
    if (pos_ + 4 > input_.size()) {
      throw std::runtime_error("short JSON unicode escape");
    }
    uint32_t code_unit = 0;
    for (uint32_t i = 0; i < 4; ++i) {
      code_unit = (code_unit << 4) | hex_value(input_[pos_++]);
    }
    return code_unit;
  }

  auto parse_string() -> std::string {
    expect('"');
    std::string result;
    while (pos_ < input_.size()) {
      auto ch = input_[pos_++];
      if (ch == '"') {
        return result;
      }
      if (ch != '\\') {
        result.push_back(ch);
        continue;
      }
      if (pos_ >= input_.size()) {
        throw std::runtime_error("unterminated JSON escape");
      }
      auto escaped = input_[pos_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          result.push_back(escaped);
          break;
        case 'b':
          result.push_back('\b');
          break;
        case 'f':
          result.push_back('\f');
          break;
        case 'n':
          result.push_back('\n');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case 't':
          result.push_back('\t');
          break;
        case 'u': {
          auto codepoint = parse_unicode_escape();
          if (is_high_surrogate(codepoint)) {
            if (pos_ + 2 > input_.size() || input_[pos_] != '\\' || input_[pos_ + 1] != 'u') {
              throw std::runtime_error("JSON high surrogate must be followed by low surrogate");
            }
            pos_ += 2;
            auto low_surrogate = parse_unicode_escape();
            if (!is_low_surrogate(low_surrogate)) {
              throw std::runtime_error("JSON high surrogate must be followed by low surrogate");
            }
            codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low_surrogate - 0xDC00);
          } else if (is_low_surrogate(codepoint)) {
            throw std::runtime_error("JSON low surrogate without preceding high surrogate");
          }
          append_utf8(result, codepoint);
          break;
        }
        default:
          throw std::runtime_error("unsupported JSON escape");
      }
    }
    throw std::runtime_error("unterminated JSON string");
  }

  auto parse_number() -> JsonValue {
    auto start = pos_;
    if (peek() == '-') {
      ++pos_;
    }
    while (std::isdigit(static_cast<unsigned char>(peek())) != 0) {
      ++pos_;
    }
    bool is_float = false;
    if (peek() == '.') {
      is_float = true;
      ++pos_;
      while (std::isdigit(static_cast<unsigned char>(peek())) != 0) {
        ++pos_;
      }
    }
    if (peek() == 'e' || peek() == 'E') {
      is_float = true;
      ++pos_;
      if (peek() == '+' || peek() == '-') {
        ++pos_;
      }
      while (std::isdigit(static_cast<unsigned char>(peek())) != 0) {
        ++pos_;
      }
    }

    auto token = input_.substr(start, pos_ - start);
    auto *begin = token.data();
    auto *end = token.data() + token.size();
    if (is_float) {
      double value = 0.0;
      auto [ptr, ec] = std::from_chars(begin, end, value);
      if (ec != std::errc() || ptr != end) {
        throw std::runtime_error("invalid JSON floating-point number");
      }
      return JsonValue{value};
    }

    int64_t value = 0;
    auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc() || ptr != end) {
      throw std::runtime_error("invalid JSON integer");
    }
    return JsonValue{value};
  }

  auto parse_array() -> JsonValue {
    expect('[');
    JsonValue::Array values;
    skip_ws();
    if (consume(']')) {
      return JsonValue{std::move(values)};
    }
    while (true) {
      values.push_back(parse_value());
      skip_ws();
      if (consume(']')) {
        break;
      }
      expect(',');
      skip_ws();
    }
    return JsonValue{std::move(values)};
  }

  auto parse_object() -> JsonValue {
    expect('{');
    JsonValue::Object object;
    skip_ws();
    if (consume('}')) {
      return JsonValue{std::move(object)};
    }
    while (true) {
      skip_ws();
      auto key = parse_string();
      skip_ws();
      expect(':');
      skip_ws();
      object.emplace(std::move(key), parse_value());
      skip_ws();
      if (consume('}')) {
        break;
      }
      expect(',');
    }
    return JsonValue{std::move(object)};
  }

  auto parse_literal(std::string_view literal, JsonValue value) -> JsonValue {
    if (input_.substr(pos_, literal.size()) != literal) {
      throw std::runtime_error("invalid JSON literal");
    }
    pos_ += literal.size();
    return value;
  }

  auto parse_value() -> JsonValue {
    skip_ws();
    switch (peek()) {
      case '{':
        return parse_object();
      case '[':
        return parse_array();
      case '"':
        return JsonValue{parse_string()};
      case 't':
        return parse_literal("true", JsonValue{true});
      case 'f':
        return parse_literal("false", JsonValue{false});
      case 'n':
        return parse_literal("null", JsonValue{nullptr});
      default:
        if (peek() == '-' || std::isdigit(static_cast<unsigned char>(peek())) != 0) {
          return parse_number();
        }
        throw std::runtime_error("invalid JSON value");
    }
  }

  std::string_view input_;
  size_t pos_ = 0;
};

inline auto parse_json_line(std::string_view line) -> JsonValue { return JsonParser(line).parse(); }

inline auto as_object(const JsonValue &value) -> const JsonValue::Object * {
  const auto *object = std::get_if<JsonValue::ObjectPtr>(&value.value_);
  return object == nullptr ? nullptr : object->get();
}

inline auto as_array(const JsonValue &value) -> const JsonValue::Array * {
  const auto *array = std::get_if<JsonValue::ArrayPtr>(&value.value_);
  return array == nullptr ? nullptr : array->get();
}

inline auto as_string(const JsonValue &value) -> const std::string * {
  return std::get_if<std::string>(&value.value_);
}

inline auto json_to_metadata_value(const JsonValue &value) -> std::optional<MetadataValue> {
  if (const auto *bool_value = std::get_if<bool>(&value.value_); bool_value != nullptr) {
    return *bool_value;
  }
  if (const auto *int_value = std::get_if<int64_t>(&value.value_); int_value != nullptr) {
    return *int_value;
  }
  if (const auto *double_value = std::get_if<double>(&value.value_); double_value != nullptr) {
    return *double_value;
  }
  if (const auto *string_value = std::get_if<std::string>(&value.value_); string_value != nullptr) {
    return *string_value;
  }
  return std::nullopt;
}

inline auto json_to_float(const JsonValue &value) -> float {
  if (const auto *int_value = std::get_if<int64_t>(&value.value_); int_value != nullptr) {
    return static_cast<float>(*int_value);
  }
  if (const auto *double_value = std::get_if<double>(&value.value_); double_value != nullptr) {
    return static_cast<float>(*double_value);
  }
  throw std::runtime_error("JSON value is not numeric");
}

inline auto json_to_u32(const JsonValue &value) -> uint32_t {
  int64_t id = 0;
  if (const auto *int_value = std::get_if<int64_t>(&value.value_); int_value != nullptr) {
    id = *int_value;
  } else if (const auto *double_value = std::get_if<double>(&value.value_);
             double_value != nullptr) {
    id = static_cast<int64_t>(*double_value);
  } else {
    throw std::runtime_error("JSON value is not an id");
  }
  if (id < 0 || id > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("JSON id is outside uint32_t range");
  }
  return static_cast<uint32_t>(id);
}

inline auto find_any(const JsonValue::Object &object, std::initializer_list<std::string_view> keys)
    -> const JsonValue * {
  for (auto key : keys) {
    auto it = object.find(std::string(key));
    if (it != object.end()) {
      return &it->second;
    }
  }
  return nullptr;
}

}  // namespace alaya
