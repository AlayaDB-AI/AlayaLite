// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils/log.hpp"
#include "utils/platform_fs.hpp"

namespace alaya::recovery {

namespace fs = std::filesystem;

constexpr uint32_t kWalFrameMagic = 0x414C5752U;    // ALWR
constexpr uint32_t kWalTrailerMagic = 0x5752414CU;  // WRAL
constexpr uint8_t kWalFormatVersion = 1;

enum class WalFrameType : uint8_t {
  PREPARE = 1,
  COMMIT = 2,
};

enum class MutationType : uint8_t {
  INSERT = 1,
  UPSERT = 2,
  REMOVE_BY_ITEM_ID = 3,
  REMOVE_BY_INTERNAL_ID = 4,
};

struct WalRecord {
  uint64_t op_id{0};
  MutationType mutation_type{MutationType::INSERT};
  std::vector<char> payload{};
};

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

  std::vector<char> buffer_{};
};

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

  const char *data_{nullptr};
  size_t size_{0};
  size_t offset_{0};
};

class WriteAheadLog {
 public:
  explicit WriteAheadLog(fs::path path) : path_(std::move(path)) {}

  ~WriteAheadLog() {
    std::lock_guard<std::mutex> lock(wal_mutex_);
    if (wal_stream_.is_open()) {
      wal_stream_.close();
    }
  }

  WriteAheadLog(const WriteAheadLog &) = delete;
  auto operator=(const WriteAheadLog &) -> WriteAheadLog & = delete;
  WriteAheadLog(WriteAheadLog &&) = delete;
  auto operator=(WriteAheadLog &&) -> WriteAheadLog & = delete;

  auto append_prepare(const WalRecord &record) const -> void {
    append_frame(WalFrameType::PREPARE, record);
  }

  auto append_commit(uint64_t op_id, MutationType mutation_type) const -> void {
    append_frame(WalFrameType::COMMIT, WalRecord{op_id, mutation_type, {}});
  }

  auto truncate() const -> void {
    std::lock_guard<std::mutex> lock(wal_mutex_);
    if (wal_stream_.is_open()) {
      wal_stream_.close();
    }
    std::error_code ec;
    fs::remove(path_, ec);
    if (ec) {
      LOG_WARN("WAL truncate failed: {}", ec.message());
    }
  }

  auto sync() const -> void {
    std::lock_guard<std::mutex> lock(wal_mutex_);
    if (wal_stream_.is_open()) {
      wal_stream_.flush();
    }
    platform::sync_file(path_);
  }

  [[nodiscard]] auto replayable_records(uint64_t applied_through,
                                        uint64_t *max_seen_op_id = nullptr) const
      -> std::vector<WalRecord> {
    // Flush any buffered writes so the reader sees all committed data.
    {
      std::lock_guard<std::mutex> lock(wal_mutex_);
      if (wal_stream_.is_open()) {
        wal_stream_.flush();
      }
    }

    std::vector<WalRecord> committed;
    std::unordered_map<uint64_t, WalRecord> pending;

    if (!fs::exists(path_)) {
      if (max_seen_op_id != nullptr) {
        *max_seen_op_id = applied_through;
      }
      return committed;
    }

    std::ifstream input(path_, std::ios::binary);
    if (!input.is_open()) {
      throw std::runtime_error("Failed to open WAL file at " + path_.string());
    }

    uint64_t max_op_id = applied_through;
    while (true) {
      auto header = read_frame_header(input);
      if (!header.has_value()) {
        break;
      }

      std::vector<char> payload(header->payload_size);
      if (header->payload_size > 0) {
        input.read(payload.data(), static_cast<std::streamsize>(payload.size()));
        if (!input) {
          LOG_WARN("WAL truncated while reading payload at {}", path_.string());
          break;
        }
      }

      uint32_t trailer = 0;
      input.read(reinterpret_cast<char *>(&trailer), sizeof(trailer));
      if (!input || trailer != kWalTrailerMagic) {
        LOG_WARN("WAL truncated or corrupted while reading trailer at {}", path_.string());
        break;
      }

      max_op_id = std::max(max_op_id, header->op_id);
      if (header->frame_type == WalFrameType::PREPARE) {
        pending[header->op_id] =
            WalRecord{header->op_id, header->mutation_type, std::move(payload)};
        continue;
      }

      auto pending_it = pending.find(header->op_id);
      if (pending_it == pending.end()) {
        LOG_WARN("WAL commit without prepare: op_id={}", header->op_id);
        continue;
      }
      if (pending_it->second.mutation_type != header->mutation_type) {
        LOG_WARN("WAL prepare/commit mutation mismatch: op_id={}", header->op_id);
        pending.erase(pending_it);
        continue;
      }
      if (header->op_id > applied_through) {
        committed.push_back(std::move(pending_it->second));
      }
      pending.erase(pending_it);
    }

    if (max_seen_op_id != nullptr) {
      *max_seen_op_id = max_op_id;
    }
    std::sort(committed.begin(), committed.end(), [](const WalRecord &lhs, const WalRecord &rhs) {
      return lhs.op_id < rhs.op_id;
    });
    return committed;
  }

 private:
  struct FrameHeader {
    WalFrameType frame_type{WalFrameType::PREPARE};
    MutationType mutation_type{MutationType::INSERT};
    uint64_t op_id{0};
    uint64_t payload_size{0};
  };

  auto ensure_stream_open() const -> void {
    if (!wal_stream_.is_open()) {
      fs::create_directories(path_.parent_path());
      wal_stream_.open(path_, std::ios::binary | std::ios::app);
      if (!wal_stream_.is_open()) {
        throw std::runtime_error("Failed to open WAL file for append at " + path_.string());
      }
    }
  }

  auto append_frame(WalFrameType frame_type, const WalRecord &record) const -> void {
    std::lock_guard<std::mutex> lock(wal_mutex_);
    ensure_stream_open();

    // Build the entire frame in a contiguous buffer to ensure a single
    // write call, avoiding partial frames from interleaved I/O.
    constexpr size_t kHeaderSize =
        sizeof(uint32_t) + sizeof(uint8_t) * 4 + sizeof(uint64_t) + sizeof(uint64_t);
    constexpr size_t kTrailerSize = sizeof(uint32_t);
    uint64_t payload_size = static_cast<uint64_t>(record.payload.size());
    std::vector<char> buf(kHeaderSize + static_cast<size_t>(payload_size) + kTrailerSize);
    char *ptr = buf.data();

    auto write_pod = [&ptr](const auto &val) {
      std::memcpy(ptr, &val, sizeof(val));
      ptr += sizeof(val);
    };

    write_pod(kWalFrameMagic);
    write_pod(kWalFormatVersion);
    write_pod(static_cast<uint8_t>(frame_type));
    write_pod(static_cast<uint8_t>(record.mutation_type));
    write_pod(static_cast<uint8_t>(0));  // reserved
    write_pod(record.op_id);
    write_pod(payload_size);
    if (payload_size > 0) {
      std::memcpy(ptr, record.payload.data(), static_cast<size_t>(payload_size));
      ptr += static_cast<size_t>(payload_size);
    }
    write_pod(kWalTrailerMagic);

    wal_stream_.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    wal_stream_.flush();

    // Only fsync on COMMIT frames. PREPARE frames are durable enough after
    // flush — a crash between PREPARE and COMMIT leaves an uncommitted record
    // that is correctly skipped during replay.
    if (frame_type == WalFrameType::COMMIT) {
      platform::sync_file(path_);
    }
  }

  [[nodiscard]] static auto read_frame_header(std::ifstream &input) -> std::optional<FrameHeader> {
    uint32_t magic = 0;
    input.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    if (input.eof()) {
      return std::nullopt;
    }
    if (!input) {
      LOG_WARN("WAL truncated while reading frame magic");
      return std::nullopt;
    }
    if (magic != kWalFrameMagic) {
      LOG_WARN("Unexpected WAL magic: {}", magic);
      return std::nullopt;
    }

    uint8_t version = 0;
    uint8_t frame_type = 0;
    uint8_t mutation_type = 0;
    uint8_t reserved = 0;
    uint64_t op_id = 0;
    uint64_t payload_size = 0;

    input.read(reinterpret_cast<char *>(&version), sizeof(version));
    input.read(reinterpret_cast<char *>(&frame_type), sizeof(frame_type));
    input.read(reinterpret_cast<char *>(&mutation_type), sizeof(mutation_type));
    input.read(reinterpret_cast<char *>(&reserved), sizeof(reserved));
    input.read(reinterpret_cast<char *>(&op_id), sizeof(op_id));
    input.read(reinterpret_cast<char *>(&payload_size), sizeof(payload_size));
    if (!input) {
      LOG_WARN("WAL truncated while reading frame header");
      return std::nullopt;
    }
    if (version != kWalFormatVersion) {
      LOG_WARN("Unsupported WAL version: {}", version);
      return std::nullopt;
    }
    return FrameHeader{
        static_cast<WalFrameType>(frame_type),
        static_cast<MutationType>(mutation_type),
        op_id,
        payload_size,
    };
  }

  fs::path path_;
  mutable std::mutex wal_mutex_;
  mutable std::ofstream wal_stream_;
};

}  // namespace alaya::recovery
