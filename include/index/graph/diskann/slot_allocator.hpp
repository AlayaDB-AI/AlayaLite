// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file slot_allocator.hpp
 * @brief Free-list slot allocator with an embedded tombstone bitmap.
 *
 * Slot lifecycle for in-place DiskANN updates (design D5):
 *   - `alloc()` reuses the most-recently-freed slot (LIFO) if the free list is
 *     non-empty, otherwise appends a fresh id (`next_fresh_id_++`). The slot
 *     stays tombstoned ("dark") until `publish()` — concurrent searches must
 *     not see a slot whose node record / PQ code is not written yet (a reused
 *     slot still has stale in-edges pointing at the old node's bytes).
 *   - `publish(id)` clears the tombstone once the slot's data is fully written
 *     (mirrors Yi, which adds a node to its live set only after the disk
 *     append).
 *   - `free(id)` pushes the slot onto the free list and tombstones it.
 *   - delete-time graph repair can split that last step: `mark_removed(id)`
 *     makes searches treat the slot as dead while it remains unavailable for
 *     reuse, then `release(id)` makes it allocatable after all in-neighbors
 *     have been patched.
 *
 * The allocator owns the `TombstoneBitmap` so that alloc/free keep liveness and
 * reuse in lockstep, and so a single `save()`/`load()` round-trips the complete
 * allocation state (free list + next id + tombstones) to one file.
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "index/graph/diskann/tombstone_bitmap.hpp"

namespace alaya::diskann {

class SlotAllocator {
 public:
  static constexpr uint64_t kMagic = 0x414C59534C4F5431ULL;  // "ALYSLOT1"

  SlotAllocator() = default;

  /// Initialize for an index that already holds @p num_existing nodes
  /// (ids [0, num_existing) are live; the next append id is num_existing).
  explicit SlotAllocator(uint32_t num_existing)
      : next_fresh_id_(num_existing), tombstone_(num_existing) {}

  /// Reset to manage an index of @p num_existing live nodes with an empty free
  /// list. Used by load() of an updatable index.
  void reset(uint32_t num_existing) {
    free_list_.clear();
    next_fresh_id_ = num_existing;
    tombstone_.reset(num_existing);
  }

  /// Allocate a slot: reuse the most recently freed id (LIFO), else append a
  /// fresh id. The returned slot stays tombstoned until publish().
  uint32_t alloc() {
    uint32_t id;
    if (!free_list_.empty()) {
      id = free_list_.back();
      free_list_.pop_back();
    } else {
      id = next_fresh_id_++;
    }
    tombstone_.set(id);
    return id;
  }

  /// Make an allocated slot visible to searches (its data is on disk/in cache).
  void publish(uint32_t id) { tombstone_.clear(id); }

  /// Free @p id: push it onto the free list and tombstone it.
  void free(uint32_t id) {
    free_list_.push_back(id);
    tombstone_.set(id);
  }

  /// Tombstone @p id without putting it on the free list. Delete-time repair
  /// must patch in-neighbors during this window, before the slot can be reused.
  void mark_removed(uint32_t id) { tombstone_.set(id); }

  /// Make a mark_removed() slot reusable after its in-neighbor repair window.
  /// Re-asserts the tombstone (a no-op in the intended mark_removed() ->
  /// release() sequence) so misuse on a live slot can never leave it
  /// simultaneously searchable and allocatable.
  void release(uint32_t id) {
    tombstone_.set(id);
    free_list_.push_back(id);
  }

  [[nodiscard]] bool is_deleted(uint32_t id) const { return tombstone_.is_deleted(id); }
  [[nodiscard]] uint32_t next_fresh_id() const { return next_fresh_id_; }
  [[nodiscard]] uint64_t free_count() const { return free_list_.size(); }
  [[nodiscard]] uint64_t tombstone_count() const { return tombstone_.count(); }

  TombstoneBitmap &tombstone() { return tombstone_; }
  [[nodiscard]] const TombstoneBitmap &tombstone() const { return tombstone_; }
  [[nodiscard]] const std::vector<uint32_t> &free_list() const { return free_list_; }

  /// Persist [magic | next_fresh_id | free_list | tombstone] to @p path.
  void save(const std::string &path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("SlotAllocator::save: cannot open " + path);
    }
    const uint64_t magic = kMagic;
    const uint64_t next = next_fresh_id_;
    const uint64_t n_free = free_list_.size();
    out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char *>(&next), sizeof(next));
    out.write(reinterpret_cast<const char *>(&n_free), sizeof(n_free));
    if (n_free > 0) {
      out.write(reinterpret_cast<const char *>(free_list_.data()),
                static_cast<std::streamsize>(n_free * sizeof(uint32_t)));
    }
    tombstone_.serialize(out);
    if (!out) {
      throw std::runtime_error("SlotAllocator::save: write failed " + path);
    }
  }

  /// Restore complete allocator state from @p path.
  void load(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("SlotAllocator::load: cannot open " + path);
    }
    uint64_t magic = 0;
    uint64_t next = 0;
    uint64_t n_free = 0;
    in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char *>(&next), sizeof(next));
    in.read(reinterpret_cast<char *>(&n_free), sizeof(n_free));
    if (!in || magic != kMagic) {
      throw std::runtime_error("SlotAllocator::load: bad magic/truncated " + path);
    }
    free_list_.assign(n_free, 0);
    if (n_free > 0) {
      in.read(reinterpret_cast<char *>(free_list_.data()),
              static_cast<std::streamsize>(n_free * sizeof(uint32_t)));
      if (!in) {
        throw std::runtime_error("SlotAllocator::load: free list truncated " + path);
      }
    }
    if (next > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("SlotAllocator::load: next_fresh_id overflows u32 " + path);
    }
    next_fresh_id_ = static_cast<uint32_t>(next);
    tombstone_.deserialize(in);
  }

 private:
  std::vector<uint32_t> free_list_;
  uint32_t next_fresh_id_ = 0;
  TombstoneBitmap tombstone_;
};

}  // namespace alaya::diskann
