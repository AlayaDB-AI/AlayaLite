// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
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
    if (data == nullptr || size == 0) {
      throw std::runtime_error("Invalid ScalarData payload");
    }

    ScalarData result;
    size_t offset = 0;

    result.item_id = deserialize_string(data, size, offset);
    result.document = deserialize_string(data, size, offset);
    result.metadata = deserialize_metadata(data, size, offset);

    if (offset > size) {
      throw std::runtime_error("Corrupted ScalarData payload");
    }

    return result;
  }

  /**
   * @brief Deserialize only selected metadata fields from serialized ScalarData.
   * @param data Pointer to serialized data
   * @param size Size of serialized data
   * @param required_fields Fields to extract from metadata
   * @return Metadata map containing only requested fields
   */
  static auto deserialize_selected_metadata(const char *data,
                                            size_t size,
                                            const std::unordered_set<std::string> &required_fields)
      -> MetadataMap {
    MetadataMap result;
    if (required_fields.empty() || data == nullptr || size == 0) {
      return result;
    }

    size_t offset = 0;
    skip_string(data, size, offset);  // item_id
    skip_string(data, size, offset);  // document

    if (offset + sizeof(uint32_t) > size) {
      return result;
    }

    uint32_t count;
    std::memcpy(&count, data + offset, sizeof(count));
    offset += sizeof(count);

    for (uint32_t i = 0; i < count && offset < size; ++i) {
      std::string key = deserialize_string(data, size, offset);
      if (key.empty() && offset > size) {
        return MetadataMap{};
      }

      if (required_fields.contains(key)) {
        MetadataValue value = deserialize_value(data, size, offset);
        if (offset > size) {
          return MetadataMap{};
        }
        result.emplace(std::move(key), std::move(value));
        if (result.size() == required_fields.size()) {
          break;
        }
      } else {
        skip_value(data, size, offset);
        if (offset > size) {
          return MetadataMap{};
        }
      }
    }
    return result;
  }

  static auto deserialize_single_metadata_value(const char *data,
                                                size_t size,
                                                std::string_view field)
      -> std::optional<MetadataValue> {
    if (field.empty() || data == nullptr || size == 0) {
      return std::nullopt;
    }

    size_t offset = 0;
    skip_string(data, size, offset);  // item_id
    skip_string(data, size, offset);  // document

    if (offset + sizeof(uint32_t) > size) {
      return std::nullopt;
    }

    uint32_t count;
    std::memcpy(&count, data + offset, sizeof(count));
    offset += sizeof(count);

    for (uint32_t i = 0; i < count && offset < size; ++i) {
      std::string key = deserialize_string(data, size, offset);
      if (key.empty() && offset > size) {
        return std::nullopt;
      }

      if (key == field) {
        auto value = deserialize_value(data, size, offset);
        if (offset > size) {
          return std::nullopt;
        }
        return value;
      }

      skip_value(data, size, offset);
      if (offset > size) {
        return std::nullopt;
      }
    }

    return std::nullopt;
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

  static auto deserialize_string(const char *data, size_t size, size_t &offset) -> std::string {
    if (offset + sizeof(uint32_t) > size) {
      offset = size + 1;
      return {};
    }
    uint32_t len;
    std::memcpy(&len, data + offset, sizeof(len));
    offset += sizeof(len);
    if (offset + len > size) {
      offset = size + 1;
      return {};
    }
    std::string result(data + offset, len);
    offset += len;
    return result;
  }

  static void skip_string(const char *data, size_t size, size_t &offset) {
    if (offset + sizeof(uint32_t) > size) {
      offset = size + 1;
      return;
    }
    uint32_t len;
    std::memcpy(&len, data + offset, sizeof(len));
    offset += sizeof(len);
    if (offset + len > size) {
      offset = size + 1;
      return;
    }
    offset += len;
  }

  static void skip_value(const char *data, size_t size, size_t &offset) {
    if (offset >= size) {
      offset = size + 1;
      return;
    }

    uint8_t type_index = static_cast<uint8_t>(data[offset++]);

    switch (type_index) {
      case 0:
        if (offset + sizeof(int64_t) > size) {
          offset = size + 1;
          return;
        }
        offset += sizeof(int64_t);
        return;
      case 1:
        if (offset + sizeof(double) > size) {
          offset = size + 1;
          return;
        }
        offset += sizeof(double);
        return;
      case 2:
        skip_string(data, size, offset);
        return;
      case 3:
        if (offset + sizeof(bool) > size) {
          offset = size + 1;
          return;
        }
        offset += sizeof(bool);
        return;
      default:
        offset = size + 1;
        return;
    }
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

  static auto deserialize_metadata(const char *data, size_t size, size_t &offset) -> MetadataMap {
    MetadataMap result;
    if (offset + sizeof(uint32_t) > size) {
      offset = size + 1;
      return result;
    }

    uint32_t count;
    std::memcpy(&count, data + offset, sizeof(count));
    offset += sizeof(count);

    for (uint32_t i = 0; i < count; ++i) {
      std::string key = deserialize_string(data, size, offset);
      if (offset > size) {
        return MetadataMap{};
      }
      MetadataValue value = deserialize_value(data, size, offset);
      if (offset > size) {
        return MetadataMap{};
      }
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

  static auto deserialize_value(const char *data, size_t size, size_t &offset) -> MetadataValue {
    if (offset >= size) {
      offset = size + 1;
      return int64_t{0};
    }

    uint8_t type_index = static_cast<uint8_t>(data[offset++]);

    switch (type_index) {
      case 0: {
        if (offset + sizeof(int64_t) > size) {
          offset = size + 1;
          return int64_t{0};
        }
        int64_t v;
        std::memcpy(&v, data + offset, sizeof(v));
        offset += sizeof(v);
        return v;
      }
      case 1: {
        if (offset + sizeof(double) > size) {
          offset = size + 1;
          return int64_t{0};
        }
        double v;
        std::memcpy(&v, data + offset, sizeof(v));
        offset += sizeof(v);
        return v;
      }
      case 2:
        return deserialize_string(data, size, offset);
      case 3: {
        if (offset + sizeof(bool) > size) {
          offset = size + 1;
          return int64_t{0};
        }
        bool v;
        std::memcpy(&v, data + offset, sizeof(v));
        offset += sizeof(v);
        return v;
      }
      default:
        offset = size + 1;
        return int64_t{0};
    }
  }
};
// NOLINTEND
}  // namespace alaya
