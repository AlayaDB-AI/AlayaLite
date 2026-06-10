// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/graph/diskann/slot_allocator.hpp"
#include "index/graph/diskann/tombstone_bitmap.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace {

using alaya::diskann::SlotAllocator;
using alaya::diskann::TombstoneBitmap;

// Unique temp path per call so save/load tests never collide across the suite.
std::string temp_path(const char *stem) {
  static std::atomic<uint64_t> counter{0};
  auto p = std::filesystem::temp_directory_path() / ("diskann_ts_" + std::string(stem) + "_" +
                                                     std::to_string(counter.fetch_add(1)) + ".bin");
  return p.string();
}

struct ScopedFile {
  std::string path;
  explicit ScopedFile(const char *stem) : path(temp_path(stem)) {}
  ~ScopedFile() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

// --------------------------- TombstoneBitmap (task 1.3) ---------------------

TEST(TombstoneBitmapTest, SetClearQuery) {
  TombstoneBitmap bm;
  EXPECT_FALSE(bm.is_deleted(5));
  bm.set(5);
  EXPECT_TRUE(bm.is_deleted(5));
  bm.clear(5);
  EXPECT_FALSE(bm.is_deleted(5));
}

TEST(TombstoneBitmapTest, AutoGrowBeyondInitialCapacity) {
  TombstoneBitmap bm(64);
  EXPECT_GE(bm.capacity(), 64u);
  EXPECT_FALSE(bm.is_deleted(100));  // out of range reads live
  bm.set(100);
  EXPECT_GE(bm.capacity(), 101u);
  EXPECT_TRUE(bm.is_deleted(100));
}

TEST(TombstoneBitmapTest, CountAfterMixedOperations) {
  TombstoneBitmap bm;
  bm.set(1);
  bm.set(5);
  bm.set(10);
  bm.clear(5);
  EXPECT_EQ(bm.count(), 2u);
  // Idempotent set/clear must not skew the running count.
  bm.set(1);
  bm.clear(999);  // never set, out of range
  EXPECT_EQ(bm.count(), 2u);
}

TEST(TombstoneBitmapTest, SaveLoadRoundTrip) {
  ScopedFile f("bm");
  {
    TombstoneBitmap bm(128);
    bm.set(3);
    bm.set(7);
    bm.set(200);  // forces a grow past the saved 128
    bm.save(f.path);
  }
  TombstoneBitmap reloaded;
  reloaded.load(f.path);
  EXPECT_TRUE(reloaded.is_deleted(3));
  EXPECT_TRUE(reloaded.is_deleted(7));
  EXPECT_TRUE(reloaded.is_deleted(200));
  EXPECT_FALSE(reloaded.is_deleted(4));
  EXPECT_EQ(reloaded.count(), 3u);
}

TEST(TombstoneBitmapTest, LoadRejectsGarbage) {
  ScopedFile f("bad");
  {
    std::ofstream out(f.path, std::ios::binary | std::ios::trunc);
    const char junk[4] = {1, 2, 3, 4};
    out.write(junk, sizeof(junk));
  }
  TombstoneBitmap bm;
  EXPECT_THROW(bm.load(f.path), std::runtime_error);
}

// --------------------------- SlotAllocator (task 1.4) -----------------------

TEST(SlotAllocatorTest, AllocFromFreeListLIFO) {
  SlotAllocator a;
  a.free(7);
  a.free(3);
  a.free(12);  // free list (top -> bottom): 12, 3, 7
  EXPECT_EQ(a.alloc(), 12u);
  EXPECT_EQ(a.alloc(), 3u);
  EXPECT_EQ(a.alloc(), 7u);
  EXPECT_EQ(a.free_count(), 0u);
}

TEST(SlotAllocatorTest, AllocAppendMode) {
  SlotAllocator a(1000);  // next_fresh_id_ = 1000, empty free list
  EXPECT_EQ(a.alloc(), 1000u);
  EXPECT_EQ(a.next_fresh_id(), 1001u);
  EXPECT_EQ(a.alloc(), 1001u);
}

TEST(SlotAllocatorTest, FreeThenReallocReturnsSameSlotAndTogglesTombstone) {
  SlotAllocator a(10);
  a.free(5);
  EXPECT_TRUE(a.is_deleted(5));
  EXPECT_EQ(a.free_count(), 1u);
  EXPECT_EQ(a.alloc(), 5u);       // reuse the freed slot
  EXPECT_FALSE(a.is_deleted(5));  // alloc marks it live again
}

TEST(SlotAllocatorTest, SaveLoadRoundTripWithBitmap) {
  ScopedFile f("slot");
  {
    SlotAllocator a(100);  // next_fresh_id_ = 100
    a.free(3);
    a.free(7);  // free list (top -> bottom): 7, 3; tombstones {3,7}
    a.save(f.path);
  }
  SlotAllocator b;
  b.load(f.path);
  EXPECT_EQ(b.next_fresh_id(), 100u);
  EXPECT_EQ(b.free_count(), 2u);
  EXPECT_TRUE(b.is_deleted(3));
  EXPECT_TRUE(b.is_deleted(7));
  EXPECT_EQ(b.tombstone_count(), 2u);
  // LIFO order survives the round trip.
  EXPECT_EQ(b.alloc(), 7u);
  EXPECT_EQ(b.alloc(), 3u);
}

}  // namespace
