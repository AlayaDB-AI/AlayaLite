// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "recovery/recovery_manager.hpp"
#include "recovery/snapshot_manifest.hpp"
#include "recovery/write_ahead_log.hpp"

namespace alaya::recovery {
// NOLINTBEGIN
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static auto make_payload(const std::string &content) -> std::vector<char> {
  return {content.begin(), content.end()};
}

static auto payload_string(const WalRecord &record) -> std::string {
  return {record.payload.begin(), record.payload.end()};
}

// ---------------------------------------------------------------------------
// WalTest
// ---------------------------------------------------------------------------

class WalTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string name = std::string(info->test_suite_name()) + "_" + info->name();
    std::replace(name.begin(), name.end(), '/', '_');
    std::replace(name.begin(), name.end(), ' ', '_');

    temp_dir_ = fs::temp_directory_path() / ("wal_test_" + name);
    fs::remove_all(temp_dir_);
    fs::create_directories(temp_dir_);
    wal_path_ = temp_dir_ / "wal.bin";
  }

  void TearDown() override {
    std::error_code ec;
    fs::remove_all(temp_dir_, ec);
  }

  fs::path temp_dir_;
  fs::path wal_path_;
};

TEST_F(WalTest, PrepareAndCommitRoundtrip) {
  WriteAheadLog wal(wal_path_);
  WalRecord record{1, MutationType::INSERT, make_payload("hello")};
  wal.append_prepare(record);
  wal.append_commit(1, MutationType::INSERT);

  auto records = wal.replayable_records(0);
  ASSERT_EQ(records.size(), 1U);
  EXPECT_EQ(records[0].op_id, 1U);
  EXPECT_EQ(records[0].mutation_type, MutationType::INSERT);
  EXPECT_EQ(payload_string(records[0]), "hello");
}

TEST_F(WalTest, PrepareWithoutCommitNotReplayable) {
  WriteAheadLog wal(wal_path_);
  wal.append_prepare({1, MutationType::INSERT, make_payload("orphan")});

  auto records = wal.replayable_records(0);
  EXPECT_TRUE(records.empty());
}

TEST_F(WalTest, MixedCommittedAndUncommitted) {
  WriteAheadLog wal(wal_path_);

  // Commit ops 1 and 3, leave op 2 uncommitted
  wal.append_prepare({1, MutationType::INSERT, make_payload("op1")});
  wal.append_commit(1, MutationType::INSERT);

  wal.append_prepare({2, MutationType::UPSERT, make_payload("op2")});
  // no commit for op 2

  wal.append_prepare({3, MutationType::REMOVE_BY_ITEM_ID, make_payload("op3")});
  wal.append_commit(3, MutationType::REMOVE_BY_ITEM_ID);

  auto records = wal.replayable_records(0);
  ASSERT_EQ(records.size(), 2U);
  EXPECT_EQ(records[0].op_id, 1U);
  EXPECT_EQ(records[1].op_id, 3U);
}

TEST_F(WalTest, AppliedThroughFiltersOlderRecords) {
  WriteAheadLog wal(wal_path_);

  for (uint64_t i = 1; i <= 3; ++i) {
    wal.append_prepare({i, MutationType::INSERT, make_payload("op" + std::to_string(i))});
    wal.append_commit(i, MutationType::INSERT);
  }

  auto records = wal.replayable_records(2);
  ASSERT_EQ(records.size(), 1U);
  EXPECT_EQ(records[0].op_id, 3U);
}

TEST_F(WalTest, TruncatedWalHandledGracefully) {
  WriteAheadLog wal(wal_path_);
  wal.append_prepare({1, MutationType::INSERT, make_payload("data")});
  wal.append_commit(1, MutationType::INSERT);

  // Corrupt the file by removing the last 4 bytes (trailer)
  {
    auto file_size = fs::file_size(wal_path_);
    ASSERT_GT(file_size, 4U);
    fs::resize_file(wal_path_, file_size - 4);
  }

  // Create a new WAL instance to read the corrupted file
  WriteAheadLog wal2(wal_path_);
  auto records = wal2.replayable_records(0);
  // The truncated commit frame should be skipped; op 1 has a valid prepare
  // but its commit is corrupted, so it should not be replayable.
  EXPECT_TRUE(records.empty());
}

TEST_F(WalTest, ConcurrentWritesDoNotCorrupt) {
  constexpr int kThreads = 8;
  constexpr int kOpsPerThread = 50;

  {
    WriteAheadLog wal(wal_path_);

    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&wal, t]() {
        for (int i = 0; i < kOpsPerThread; ++i) {
          uint64_t op_id = static_cast<uint64_t>(t * 1000 + i + 1);
          auto payload = make_payload("t" + std::to_string(t) + "_" + std::to_string(i));
          wal.append_prepare({op_id, MutationType::INSERT, payload});
          wal.append_commit(op_id, MutationType::INSERT);
        }
      });
    }

    for (auto &th : threads) {
      th.join();
    }
  }  // WAL destructor closes the write stream

  // Read with a fresh instance (simulates process restart)
  WriteAheadLog reader(wal_path_);
  auto records = reader.replayable_records(0);
  // All kThreads * kOpsPerThread operations should be committed
  EXPECT_EQ(records.size(), static_cast<size_t>(kThreads * kOpsPerThread));

  // Verify all unique op_ids are present
  std::set<uint64_t> op_ids;
  for (const auto &rec : records) {
    op_ids.insert(rec.op_id);
  }
  EXPECT_EQ(op_ids.size(), static_cast<size_t>(kThreads * kOpsPerThread));
}

TEST_F(WalTest, TruncateRemovesWalFile) {
  WriteAheadLog wal(wal_path_);
  wal.append_prepare({1, MutationType::INSERT, make_payload("data")});
  wal.append_commit(1, MutationType::INSERT);
  ASSERT_TRUE(fs::exists(wal_path_));

  wal.truncate();
  EXPECT_FALSE(fs::exists(wal_path_));

  auto records = wal.replayable_records(0);
  EXPECT_TRUE(records.empty());

  // Can write new records after truncation
  wal.append_prepare({2, MutationType::UPSERT, make_payload("new")});
  wal.append_commit(2, MutationType::UPSERT);

  records = wal.replayable_records(0);
  ASSERT_EQ(records.size(), 1U);
  EXPECT_EQ(records[0].op_id, 2U);
}

TEST_F(WalTest, MaxSeenOpIdTracksHighest) {
  WriteAheadLog wal(wal_path_);

  // Committed op 3
  wal.append_prepare({3, MutationType::INSERT, make_payload("op3")});
  wal.append_commit(3, MutationType::INSERT);

  // Uncommitted op 5 (prepare only)
  wal.append_prepare({5, MutationType::UPSERT, make_payload("op5")});

  uint64_t max_seen = 0;
  auto records = wal.replayable_records(0, &max_seen);
  ASSERT_EQ(records.size(), 1U);
  EXPECT_EQ(records[0].op_id, 3U);
  EXPECT_EQ(max_seen, 5U);
}

// ---------------------------------------------------------------------------
// SnapshotManifestTest
// ---------------------------------------------------------------------------

TEST(SnapshotManifestTest, SerializeDeserializeRoundtrip) {
  SnapshotManifest original;
  original.format_version = 1;
  original.snapshot_id = "snapshot-123456";
  original.reason = "test_reason";
  original.applied_through_op_id = 42;
  original.created_unix_ms = 1700000000000ULL;
  original.graph_file = "graph.bin";
  original.data_file = "data.bin";
  original.quant_file = "quant.bin";
  original.rocksdb_dir = "rocksdb";

  auto serialized = original.serialize();
  auto deserialized = SnapshotManifest::deserialize(serialized);

  ASSERT_TRUE(deserialized.has_value());
  EXPECT_EQ(deserialized->format_version, original.format_version);
  EXPECT_EQ(deserialized->snapshot_id, original.snapshot_id);
  EXPECT_EQ(deserialized->reason, original.reason);
  EXPECT_EQ(deserialized->applied_through_op_id, original.applied_through_op_id);
  EXPECT_EQ(deserialized->created_unix_ms, original.created_unix_ms);
  EXPECT_EQ(deserialized->graph_file, original.graph_file);
  EXPECT_EQ(deserialized->data_file, original.data_file);
  EXPECT_EQ(deserialized->quant_file, original.quant_file);
  EXPECT_EQ(deserialized->rocksdb_dir, original.rocksdb_dir);
}

TEST(SnapshotManifestTest, DeserializeRejectsEmptySnapshotId) {
  std::string raw = "format_version=1\nsnapshot_id=\nreason=test\n";
  auto result = SnapshotManifest::deserialize(raw);
  EXPECT_FALSE(result.has_value());
}

// ---------------------------------------------------------------------------
// RecoveryManagerTest
// ---------------------------------------------------------------------------

class RecoveryManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string name = std::string(info->test_suite_name()) + "_" + info->name();
    std::replace(name.begin(), name.end(), '/', '_');
    std::replace(name.begin(), name.end(), ' ', '_');

    temp_dir_ = fs::temp_directory_path() / ("recovery_mgr_test_" + name);
    fs::remove_all(temp_dir_);
    fs::create_directories(temp_dir_);

    root_dir_ = temp_dir_ / "recovery";
    active_rocksdb_ = temp_dir_ / "active_rocksdb";
  }

  void TearDown() override {
    std::error_code ec;
    fs::remove_all(temp_dir_, ec);
  }

  auto make_manager() -> RecoveryManager { return RecoveryManager(root_dir_, active_rocksdb_); }

  auto make_manifest(const std::string &snapshot_id, uint64_t applied_through) -> SnapshotManifest {
    SnapshotManifest manifest;
    manifest.snapshot_id = snapshot_id;
    manifest.reason = "test";
    manifest.applied_through_op_id = applied_through;
    manifest.created_unix_ms = SnapshotManifest::current_unix_ms();
    return manifest;
  }

  fs::path temp_dir_;
  fs::path root_dir_;
  fs::path active_rocksdb_;
};

TEST_F(RecoveryManagerTest, PublishAndReadBackSnapshot) {
  auto mgr = make_manager();
  mgr.ensure_layout();

  auto snapshot_dir = mgr.create_snapshot_dir();
  auto manifest = make_manifest(snapshot_dir.filename().string(), 10);
  mgr.publish_snapshot(manifest, snapshot_dir);

  auto current = mgr.current_snapshot();
  ASSERT_TRUE(current.has_value());
  EXPECT_EQ(current->snapshot_id, manifest.snapshot_id);
  EXPECT_EQ(current->applied_through_op_id, 10U);

  auto current_dir = mgr.current_snapshot_dir();
  ASSERT_TRUE(current_dir.has_value());
  EXPECT_EQ(*current_dir, snapshot_dir);
}

TEST_F(RecoveryManagerTest, OldSnapshotsRemovedAfterPublish) {
  auto mgr = make_manager();
  mgr.ensure_layout();

  // Create snapshot dirs with deterministic unique names to avoid
  // same-millisecond collisions from create_snapshot_dir().
  auto snapshots_base = root_dir_ / "snapshots";
  auto dir_a = snapshots_base / "snapshot-aaa";
  auto dir_b = snapshots_base / "snapshot-bbb";
  fs::create_directories(dir_a);
  fs::create_directories(dir_b);

  // Publish snapshot A
  auto manifest_a = make_manifest("snapshot-aaa", 5);
  mgr.publish_snapshot(manifest_a, dir_a);
  ASSERT_TRUE(fs::exists(dir_a));

  // Publish snapshot B
  auto manifest_b = make_manifest("snapshot-bbb", 10);
  mgr.publish_snapshot(manifest_b, dir_b);

  // Snapshot A should be removed, B should exist
  EXPECT_FALSE(fs::exists(dir_a));
  EXPECT_TRUE(fs::exists(dir_b));

  auto current = mgr.current_snapshot();
  ASSERT_TRUE(current.has_value());
  EXPECT_EQ(current->snapshot_id, manifest_b.snapshot_id);
}

TEST_F(RecoveryManagerTest, WalTruncatedAfterPublish) {
  auto mgr = make_manager();
  mgr.ensure_layout();

  // Write some WAL records
  WalRecord record{1, MutationType::INSERT, make_payload("test")};
  mgr.append_prepare(record);
  mgr.append_commit(1, MutationType::INSERT);

  auto before = mgr.replayable_records(0);
  ASSERT_EQ(before.size(), 1U);

  // Publish snapshot
  auto snapshot_dir = mgr.create_snapshot_dir();
  auto manifest = make_manifest(snapshot_dir.filename().string(), 1);
  mgr.publish_snapshot(manifest, snapshot_dir);

  // WAL should be truncated
  auto after = mgr.replayable_records(0);
  EXPECT_TRUE(after.empty());
}

TEST_F(RecoveryManagerTest, NextOperationIdReflectsState) {
  auto mgr = make_manager();
  mgr.ensure_layout();

  // Initially, no WAL and no snapshot -> next_operation_id = 1
  EXPECT_EQ(mgr.next_operation_id(), 1U);

  // Write a WAL record
  mgr.append_prepare({5, MutationType::INSERT, make_payload("data")});
  mgr.append_commit(5, MutationType::INSERT);

  // next_operation_id should be 6
  EXPECT_EQ(mgr.next_operation_id(), 6U);
}

// NOLINTEND
}  // namespace alaya::recovery
