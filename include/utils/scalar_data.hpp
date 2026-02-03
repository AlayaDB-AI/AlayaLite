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
#include <cstring>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace alaya {
// NOLINTBEGIN
/// Supported metadata value types
using MetadataValue = std::variant<int64_t, double, std::string, bool>;
using MetadataMap = std::unordered_map<std::string, MetadataValue>;

/**
 * @brief Scalar data structure for storing {item_id, document, metadata}
 *
 * This struct is used as the MetaDataType template parameter for Space classes.
 * It supports serialization/deserialization for RocksDB storage.
 */
struct ScalarData {
  std::string item_id;   ///< User-provided external ID
  std::string document;  ///< Document content
  MetadataMap metadata;  ///< Metadata key-value pairs

  ScalarData() = default;

  ScalarData(std::string id, std::string doc, MetadataMap meta)
      : item_id(std::move(id)), document(std::move(doc)), metadata(std::move(meta)) {}

  /**
   * @brief Serialize to byte stream for RocksDB storage
   * @return Serialized byte vector
   */
  [[nodiscard]] auto serialize() const -> std::vector<char> {
    std::vector<char> buffer;

    // Serialize item_id
    serialize_string(buffer, item_id);
    // Serialize document
    serialize_string(buffer, document);
    // Serialize metadata
    serialize_metadata(buffer, metadata);

    return buffer;
  }

  /**
   * @brief Deserialize from byte stream
   * @param data Pointer to serialized data
   * @param size Size of serialized data
   * @return Deserialized ScalarData
   */
  static auto deserialize(const char *data, size_t size) -> ScalarData {
    ScalarData result;
    size_t offset = 0;

    result.item_id = deserialize_string(data, offset);
    result.document = deserialize_string(data, offset);
    result.metadata = deserialize_metadata(data, offset);

    return result;
  }

  /**
   * @brief Get the serialized size (for estimation)
   * @return Estimated serialized size in bytes
   */
  [[nodiscard]] auto serialized_size() const -> size_t {
    size_t size = 0;
    size += sizeof(uint32_t) + item_id.size();
    size += sizeof(uint32_t) + document.size();
    size += sizeof(uint32_t);  // metadata count
    for (const auto &[key, value] : metadata) {
      size += sizeof(uint32_t) + key.size();
      size += 1;  // type tag
      std::visit(
          [&size](const auto &v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::string>) {
              size += sizeof(uint32_t) + v.size();
            } else {
              size += sizeof(T);
            }
          },
          value);
    }
    return size;
  }

 private:
  static void serialize_string(std::vector<char> &buffer, const std::string &str) {
    uint32_t len = static_cast<uint32_t>(str.size());
    const char *len_ptr = reinterpret_cast<const char *>(&len);
    buffer.insert(buffer.end(), len_ptr, len_ptr + sizeof(len));
    buffer.insert(buffer.end(), str.begin(), str.end());
  }

  static auto deserialize_string(const char *data, size_t &offset) -> std::string {
    uint32_t len;
    std::memcpy(&len, data + offset, sizeof(len));
    offset += sizeof(len);
    std::string result(data + offset, len);
    offset += len;
    return result;
  }

  static void serialize_metadata(std::vector<char> &buffer, const MetadataMap &meta) {
    uint32_t count = static_cast<uint32_t>(meta.size());
    const char *count_ptr = reinterpret_cast<const char *>(&count);
    buffer.insert(buffer.end(), count_ptr, count_ptr + sizeof(count));

    for (const auto &[key, value] : meta) {
      serialize_string(buffer, key);
      serialize_value(buffer, value);
    }
  }

  static void serialize_value(std::vector<char> &buffer, const MetadataValue &value) {
    // Type index: 0=int64_t, 1=double, 2=string, 3=bool
    uint8_t type_index = static_cast<uint8_t>(value.index());
    buffer.push_back(static_cast<char>(type_index));

    std::visit(
        [&buffer](const auto &v) {
          using T = std::decay_t<decltype(v)>;
          if constexpr (std::is_same_v<T, std::string>) {
            serialize_string(buffer, v);
          } else {
            const char *ptr = reinterpret_cast<const char *>(&v);
            buffer.insert(buffer.end(), ptr, ptr + sizeof(T));
          }
        },
        value);
  }

  static auto deserialize_metadata(const char *data, size_t &offset) -> MetadataMap {
    MetadataMap result;
    uint32_t count;
    std::memcpy(&count, data + offset, sizeof(count));
    offset += sizeof(count);

    for (uint32_t i = 0; i < count; ++i) {
      std::string key = deserialize_string(data, offset);
      MetadataValue value = deserialize_value(data, offset);
      result[key] = std::move(value);
    }
    return result;
  }

  static auto deserialize_value(const char *data, size_t &offset) -> MetadataValue {
    uint8_t type_index = static_cast<uint8_t>(data[offset++]);

    switch (type_index) {
      case 0: {  // int64_t
        int64_t v;
        std::memcpy(&v, data + offset, sizeof(v));
        offset += sizeof(v);
        return v;
      }
      case 1: {  // double
        double v;
        std::memcpy(&v, data + offset, sizeof(v));
        offset += sizeof(v);
        return v;
      }
      case 2: {  // string
        return deserialize_string(data, offset);
      }
      case 3: {  // bool
        bool v;
        std::memcpy(&v, data + offset, sizeof(v));
        offset += sizeof(v);
        return v;
      }
      default:
        return int64_t{0};
    }
  }
};
// NOLINTEND
}  // namespace alaya
