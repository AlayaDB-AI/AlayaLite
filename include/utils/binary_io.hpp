// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace alaya::binary_io {

/**
 * @brief Small helper for building length-prefixed binary payloads.
 *
 * Values are appended in host byte order. Strings and blobs are stored as an unsigned 64-bit byte
 * length followed by the raw bytes.
 */
class BinaryWriter {
 public:
  auto write_u8(uint8_t value) -> void { write_pod(value); }
  auto write_u32(uint32_t value) -> void { write_pod(value); }
  auto write_u64(uint64_t value) -> void { write_pod(value); }

  auto write_bytes(const char *data, size_t size) -> void {
    if (size == 0) {
      return;
    }
    buffer_.insert(buffer_.end(), data, data + size);
  }

  auto write_string(const std::string &value) -> void {
    write_u64(static_cast<uint64_t>(value.size()));
    write_bytes(value.data(), value.size());
  }

  auto write_blob(const std::vector<char> &value) -> void {
    write_u64(static_cast<uint64_t>(value.size()));
    write_bytes(value.data(), value.size());
  }

  auto write_blob(const std::string &value) -> void {
    write_u64(static_cast<uint64_t>(value.size()));
    write_bytes(value.data(), value.size());
  }

  template <typename T>
  auto write_vector_blob(const T *data, size_t element_count) -> void {
    auto bytes = static_cast<uint64_t>(element_count * sizeof(T));
    write_u64(bytes);
    write_bytes(reinterpret_cast<const char *>(data), static_cast<size_t>(bytes));
  }

  [[nodiscard]] auto finish() && -> std::vector<char> { return std::move(buffer_); }

 private:
  template <typename T>
  auto write_pod(const T &value) -> void {
    auto *ptr = reinterpret_cast<const char *>(&value);
    buffer_.insert(buffer_.end(), ptr, ptr + sizeof(T));
  }

  std::vector<char> buffer_;  ///< Accumulated serialized bytes returned by finish().
};

/**
 * @brief Bounds-checked reader for payloads produced by BinaryWriter.
 *
 * Each read advances the internal offset only after enough bytes are available. Failed reads return
 * std::nullopt so callers can reject truncated or malformed payloads.
 */
class BinaryReader {
 public:
  BinaryReader(const char *data, size_t size) : data_(data), size_(size) {}

  auto read_u8() -> std::optional<uint8_t> { return read_pod<uint8_t>(); }
  auto read_u32() -> std::optional<uint32_t> { return read_pod<uint32_t>(); }
  auto read_u64() -> std::optional<uint64_t> { return read_pod<uint64_t>(); }

  auto read_string() -> std::optional<std::string> {
    auto size = read_u64();
    if (!size.has_value()) {
      return std::nullopt;
    }
    if (remaining() < size.value()) {
      return std::nullopt;
    }
    std::string value(data_ + offset_, data_ + offset_ + static_cast<size_t>(size.value()));
    offset_ += static_cast<size_t>(size.value());
    return value;
  }

  auto read_blob() -> std::optional<std::vector<char>> {
    auto size = read_u64();
    if (!size.has_value()) {
      return std::nullopt;
    }
    if (remaining() < size.value()) {
      return std::nullopt;
    }
    std::vector<char> value(data_ + offset_, data_ + offset_ + static_cast<size_t>(size.value()));
    offset_ += static_cast<size_t>(size.value());
    return value;
  }

  auto read_fixed_blob(size_t size) -> std::optional<std::vector<char>> {
    if (remaining() < size) {
      return std::nullopt;
    }
    std::vector<char> value(data_ + offset_, data_ + offset_ + size);
    offset_ += size;
    return value;
  }

  [[nodiscard]] auto remaining() const -> size_t { return size_ - offset_; }

 private:
  template <typename T>
  auto read_pod() -> std::optional<T> {
    if (remaining() < sizeof(T)) {
      return std::nullopt;
    }
    T value{};
    std::memcpy(&value, data_ + offset_, sizeof(T));
    offset_ += sizeof(T);
    return value;
  }

  const char *data_{nullptr};  ///< Non-owning serialized payload pointer.
  size_t size_{0};             ///< Total available bytes in data_.
  size_t offset_{0};           ///< Current read offset in bytes.
};

}  // namespace alaya::binary_io
