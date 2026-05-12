// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils/binary_io.hpp"
#include "utils/log.hpp"
#include "utils/platform_fs.hpp"

namespace alaya::recovery {

namespace fs = std::filesystem;

constexpr uint32_t kWalFrameMagic = 0x48454144U;     ///< WAL frame prefix magic, "HEAD".
constexpr uint32_t kWalTrailerMagic = 0x5441494CU;   ///< WAL frame trailer magic, "TAIL".
constexpr uint8_t kWalFormatVersion = 1;             ///< Current on-disk WAL frame version.
constexpr uint64_t kMaxWalPayloadSize = 1ULL << 30;  ///< Maximum accepted WAL payload bytes.

/**
 * @brief Identifies whether a WAL frame starts or commits a mutation.
 *
 * Recovery only replays records that have both a PREPARE frame with payload and a matching COMMIT
 * frame, which avoids applying half-written operations after a crash.
 */
enum class WalFrameType : uint8_t {
  kPrepare = 1,  ///< Contains the full mutation payload before commit.
  kCommit = 2,   ///< Marks the prepared operation as committed and re-playable.
};

/**
 * @brief Logical mutation category stored with a WAL record.
 *
 * The mutation type lets recovery route a replayed payload to the same insert, upsert or delete
 * path that originally produced the operation.
 */
enum class MutationType : uint8_t {
  kInsert = 1,              ///< Insert a new vector/scalar record.
  kUpsert = 2,              ///< Insert or replace a vector/scalar record.
  kRemoveByItemId = 3,      ///< Delete by external item id.
  kRemoveByInternalId = 4,  ///< Delete by internal storage id.
};

/**
 * @brief In-memory representation of one durable mutation from the WAL.
 *
 * PREPARE frames carry the serialized payload, while COMMIT frames validate that the prepared
 * operation became durable. Replayed records are returned with their original operation id and
 * mutation type.
 */
struct WalRecord {
  uint64_t op_id_{0};                                  ///< Monotonic operation id.
  MutationType mutation_type_{MutationType::kInsert};  ///< Mutation category for replay.
  std::vector<char> payload_;                          ///< Opaque serialized mutation payload.
};

using BinaryReader = binary_io::BinaryReader;
using BinaryWriter = binary_io::BinaryWriter;

/**
 * @brief Append-only write-ahead log used to recover operations after crashes.
 *
 * The log stores each operation as PREPARE plus COMMIT frames. Replay scans complete frames,
 * validates their prepare/commit pairing and returns only committed records newer than the active
 * snapshot.
 */
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
    append_frame(WalFrameType::kPrepare, record);
  }

  auto append_commit(uint64_t op_id, MutationType mutation_type) const -> void {
    append_frame(WalFrameType::kCommit,
                 WalRecord{.op_id_ = op_id, .mutation_type_ = mutation_type, .payload_ = {}});
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

  /**
   * @brief Reads the WAL and returns durable committed records that have not been applied yet.
   *
   * PREPARE frames are held until a matching COMMIT frame is seen. Truncated or corrupted tail
   * frames stop replay after logging a warning, preserving all complete committed records that were
   * written before the damaged section.
   *
   * @param applied_through Highest operation id already covered by the active snapshot.
   * @param max_seen_op_id Optional output for the largest operation id observed while scanning.
   * @return Committed WAL records sorted by operation id.
   */
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
    std::error_code file_size_ec;
    const auto wal_file_size = fs::file_size(path_, file_size_ec);
    if (file_size_ec) {
      LOG_WARN("Failed to stat WAL file at {}: {}", path_.string(), file_size_ec.message());
      if (max_seen_op_id != nullptr) {
        *max_seen_op_id = applied_through;
      }
      return committed;
    }

    uint64_t max_op_id = applied_through;
    while (true) {
      auto header = read_frame_header(input);
      if (!header.has_value()) {
        break;
      }
      if (!payload_size_is_plausible(input, wal_file_size, header->payload_size_)) {
        LOG_WARN("WAL payload size is invalid: op_id={} payload_size={} path={}",
                 header->op_id_,
                 header->payload_size_,
                 path_.string());
        break;
      }

      std::vector<char> payload(static_cast<size_t>(header->payload_size_));
      if (header->payload_size_ > 0) {
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

      max_op_id = std::max(max_op_id, header->op_id_);
      if (header->frame_type_ == WalFrameType::kPrepare) {
        pending[header->op_id_] = WalRecord{.op_id_ = header->op_id_,
                                            .mutation_type_ = header->mutation_type_,
                                            .payload_ = std::move(payload)};
        continue;
      }

      auto pending_it = pending.find(header->op_id_);
      if (pending_it == pending.end()) {
        LOG_WARN("WAL commit without prepare: op_id={}", header->op_id_);
        continue;
      }
      if (pending_it->second.mutation_type_ != header->mutation_type_) {
        LOG_WARN("WAL prepare/commit mutation mismatch: op_id={}", header->op_id_);
        pending.erase(pending_it);
        continue;
      }
      if (header->op_id_ > applied_through) {
        committed.push_back(std::move(pending_it->second));
      }
      pending.erase(pending_it);
    }

    if (max_seen_op_id != nullptr) {
      *max_seen_op_id = max_op_id;
    }
    std::ranges::sort(committed, [](const WalRecord &lhs, const WalRecord &rhs) {
      return lhs.op_id_ < rhs.op_id_;
    });
    return committed;
  }

 private:
  struct FrameHeader {
    WalFrameType frame_type_{WalFrameType::kPrepare};    ///< PREPARE payload or COMMIT marker.
    MutationType mutation_type_{MutationType::kInsert};  ///< Mutation category from the frame.
    uint64_t op_id_{0};                                  ///< Operation id for frame pairing.
    uint64_t payload_size_{0};                           ///< Payload byte count after header.
  };

  [[nodiscard]] static auto parse_frame_type(uint8_t frame_type) -> std::optional<WalFrameType> {
    switch (frame_type) {
      case static_cast<uint8_t>(WalFrameType::kPrepare):
        return WalFrameType::kPrepare;
      case static_cast<uint8_t>(WalFrameType::kCommit):
        return WalFrameType::kCommit;
      default:
        return std::nullopt;
    }
  }

  [[nodiscard]] static auto parse_mutation_type(uint8_t mutation_type)
      -> std::optional<MutationType> {
    switch (mutation_type) {
      case static_cast<uint8_t>(MutationType::kInsert):
        return MutationType::kInsert;
      case static_cast<uint8_t>(MutationType::kUpsert):
        return MutationType::kUpsert;
      case static_cast<uint8_t>(MutationType::kRemoveByItemId):
        return MutationType::kRemoveByItemId;
      case static_cast<uint8_t>(MutationType::kRemoveByInternalId):
        return MutationType::kRemoveByInternalId;
      default:
        return std::nullopt;
    }
  }

  [[nodiscard]] static auto payload_size_is_plausible(std::ifstream &input,
                                                      uintmax_t wal_file_size,
                                                      uint64_t payload_size) -> bool {
    if (payload_size > kMaxWalPayloadSize) {
      return false;
    }
    if (payload_size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
      return false;
    }
    if (payload_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      return false;
    }

    const auto payload_start = input.tellg();
    if (payload_start == std::ifstream::pos_type(-1)) {
      return false;
    }
    const auto payload_offset = static_cast<uintmax_t>(payload_start);
    if (payload_offset > wal_file_size) {
      return false;
    }

    constexpr uintmax_t kTrailerSize = sizeof(uint32_t);
    const auto remaining_bytes = wal_file_size - payload_offset;
    return remaining_bytes >= kTrailerSize &&
           static_cast<uintmax_t>(payload_size) <= remaining_bytes - kTrailerSize;
  }

  auto ensure_stream_open() const -> void {
    if (!wal_stream_.is_open()) {
      fs::create_directories(path_.parent_path());
      wal_stream_.open(path_, std::ios::binary | std::ios::app);
      if (!wal_stream_.is_open()) {
        throw std::runtime_error("Failed to open WAL file for append at " + path_.string());
      }
    }
  }

  /**
   * @brief Serializes and appends one WAL frame for a PREPARE or COMMIT operation.
   *
   * The full frame is assembled into a contiguous buffer before writing so the header, payload and
   * trailer are emitted together. COMMIT frames are fsynced to make the commit boundary durable for
   * crash recovery.
   */
  auto append_frame(WalFrameType frame_type, const WalRecord &record) const -> void {
    std::lock_guard<std::mutex> lock(wal_mutex_);
    ensure_stream_open();

    // Build the entire frame in a contiguous buffer to ensure a single
    // write call, avoiding partial frames from interleaved I/O.
    constexpr size_t kHeaderSize =
        sizeof(uint32_t) + sizeof(uint8_t) * 4 + sizeof(uint64_t) + sizeof(uint64_t);
    constexpr size_t kTrailerSize = sizeof(uint32_t);
    auto payload_size = static_cast<uint64_t>(record.payload_.size());
    if (payload_size > kMaxWalPayloadSize) {
      throw std::runtime_error("WAL payload exceeds maximum frame size");
    }

    std::vector<char> buf(kHeaderSize + static_cast<size_t>(payload_size) + kTrailerSize);
    char *ptr = buf.data();

    auto write_pod = [&ptr](const auto &val) {
      std::memcpy(ptr, &val, sizeof(val));
      ptr += sizeof(val);
    };

    write_pod(kWalFrameMagic);

    write_pod(kWalFormatVersion);
    write_pod(static_cast<uint8_t>(frame_type));
    write_pod(static_cast<uint8_t>(record.mutation_type_));
    write_pod(static_cast<uint8_t>(0));  // reserved
    write_pod(record.op_id_);
    write_pod(payload_size);
    if (payload_size > 0) {
      std::memcpy(ptr, record.payload_.data(), static_cast<size_t>(payload_size));
      ptr += static_cast<size_t>(payload_size);
    }

    write_pod(kWalTrailerMagic);

    wal_stream_.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    wal_stream_.flush();

    // Only fsync on COMMIT frames. PREPARE frames are durable enough after
    // flush — a crash between PREPARE and COMMIT leaves an uncommitted record
    // that is correctly skipped during replay.
    if (frame_type == WalFrameType::kCommit) {
      platform::sync_file(path_);
    }
  }

  /**
   * @brief Reads and validates the fixed-size WAL frame header from the input stream.
   *
   * The reader returns std::nullopt on EOF, truncated headers, unexpected magic values or
   * unsupported format versions so the replay loop can stop at the last complete valid frame.
   */
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
    const auto parsed_frame_type = parse_frame_type(frame_type);
    if (!parsed_frame_type.has_value()) {
      LOG_WARN("Invalid WAL frame type: {}", static_cast<unsigned>(frame_type));
      return std::nullopt;
    }
    const auto parsed_mutation_type = parse_mutation_type(mutation_type);
    if (!parsed_mutation_type.has_value()) {
      LOG_WARN("Invalid WAL mutation type: {}", static_cast<unsigned>(mutation_type));
      return std::nullopt;
    }
    return FrameHeader{
        .frame_type_ = parsed_frame_type.value(),
        .mutation_type_ = parsed_mutation_type.value(),
        .op_id_ = op_id,
        .payload_size_ = payload_size,
    };
  }

  fs::path path_;                     ///< Filesystem path of the WAL file.
  mutable std::mutex wal_mutex_;      ///< Guards appends, flushes, truncation and stream state.
  mutable std::ofstream wal_stream_;  ///< Lazily opened append stream for WAL frames.
};

}  // namespace alaya::recovery
