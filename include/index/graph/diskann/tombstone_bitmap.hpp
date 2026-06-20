// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file tombstone_bitmap.hpp
 * @brief Deleted-slot bitmap for in-place DiskANN updates.
 *
 * A word-packed bitset with one bit per slot id: `set(id)` marks a slot deleted
 * (tombstoned), `clear(id)` marks it live, `is_deleted(id)` queries the status.
 * The bitmap auto-grows when an id beyond the current capacity is touched, and a
 * running deleted-bit count makes the safety-net threshold check O(1).
 *
 * Out-of-range ids read as *live* (`is_deleted` returns false), so a freshly
 * appended slot whose id exceeds the saved capacity is implicitly live without
 * forcing a grow. The on-disk format is an explicit little-endian blob with a
 * magic header, decoupled from the host bitset representation.
 */

#pragma once

#include <bit>
#include <cstdint>
#include <fstream>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace alaya::diskann {

static_assert(std::endian::native == std::endian::little,
              "TombstoneBitmap on-disk format assumes little-endian host");

class TombstoneBitmap {
 public:
  static constexpr uint64_t kMagic = 0x414C5954424D5031ULL;  // "ALYTBMP1"

  TombstoneBitmap() = default;

  /// Pre-size the bitmap to hold at least @p initial_bits (all live).
  explicit TombstoneBitmap(uint64_t initial_bits) { ensure_capacity(initial_bits); }

  /// Mark slot @p id as deleted (grows the bitmap if @p id is out of range).
  void set(uint64_t id) {
    ensure_capacity(id + 1);
    uint64_t &w = words_[id >> 6];
    const uint64_t mask = uint64_t{1} << (id & 63);
    if ((w & mask) == 0) {
      w |= mask;
      ++count_;
    }
  }

  /// Mark slot @p id as live. Clearing an out-of-range id is a no-op (already
  /// live), so no allocation is forced.
  void clear(uint64_t id) {
    if (id >= num_bits_) {
      return;
    }
    uint64_t &w = words_[id >> 6];
    const uint64_t mask = uint64_t{1} << (id & 63);
    if ((w & mask) != 0) {
      w &= ~mask;
      --count_;
    }
  }

  /// True iff slot @p id is tombstoned. Out-of-range ids are live (false).
  [[nodiscard]] bool is_deleted(uint64_t id) const {
    if (id >= num_bits_) {
      return false;
    }
    return ((words_[id >> 6] >> (id & 63)) & uint64_t{1}) != 0;
  }

  /// Number of tombstoned (set) bits.
  [[nodiscard]] uint64_t count() const { return count_; }

  /// Bit capacity (word-aligned, >= the highest id ever touched + 1).
  [[nodiscard]] uint64_t capacity() const { return num_bits_; }

  /// Write [magic | num_words | words] into @p out. Reusable so an enclosing
  /// container (e.g. SlotAllocator) can embed the bitmap in its own file.
  void serialize(std::ostream &out) const {
    const uint64_t magic = kMagic;
    const uint64_t n_words = words_.size();
    out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char *>(&n_words), sizeof(n_words));
    if (n_words > 0) {
      out.write(reinterpret_cast<const char *>(words_.data()),
                static_cast<std::streamsize>(n_words * sizeof(uint64_t)));
    }
  }

  /// Read state previously written by serialize(). The deleted count is
  /// recomputed via popcount, so it is never trusted from disk.
  void deserialize(std::istream &in) {
    uint64_t magic = 0;
    uint64_t n_words = 0;
    in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char *>(&n_words), sizeof(n_words));
    if (!in || magic != kMagic) {
      throw std::runtime_error("TombstoneBitmap::deserialize: bad magic/truncated");
    }
    words_.assign(n_words, 0);
    if (n_words > 0) {
      in.read(reinterpret_cast<char *>(words_.data()),
              static_cast<std::streamsize>(n_words * sizeof(uint64_t)));
      if (!in) {
        throw std::runtime_error("TombstoneBitmap::deserialize: words truncated");
      }
    }
    num_bits_ = n_words * 64;
    count_ = 0;
    for (const uint64_t w : words_) {
      count_ += static_cast<uint64_t>(std::popcount(w));
    }
  }

  /// Persist to a standalone @p path (truncating).
  void save(const std::string &path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("TombstoneBitmap::save: cannot open " + path);
    }
    serialize(out);
    if (!out) {
      throw std::runtime_error("TombstoneBitmap::save: write failed " + path);
    }
  }

  /// Restore from a standalone @p path written by save().
  void load(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("TombstoneBitmap::load: cannot open " + path);
    }
    deserialize(in);
  }

 private:
  /// Grow so that bit index (bits-1) is addressable. New words are zero (live).
  void ensure_capacity(uint64_t bits) {
    if (bits <= num_bits_) {
      return;
    }
    const uint64_t need_words = (bits + 63) / 64;
    if (need_words > words_.size()) {
      words_.resize(need_words, 0);
    }
    num_bits_ = words_.size() * 64;
  }

  std::vector<uint64_t> words_;
  uint64_t num_bits_ = 0;
  uint64_t count_ = 0;
};

}  // namespace alaya::diskann
