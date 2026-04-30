/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "index/disk/disk_collection.hpp"
#include "index/disk/disk_flat_builder.hpp"
#include "index/disk/disk_flat_searcher.hpp"
#include "index/disk/segment_factory.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "utils/metric_type.hpp"

namespace alaya::disk {

namespace {

class DiskCollectionFactoryDispatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto pid_str = std::to_string(static_cast<long long>(::getpid()));
    auto base = std::filesystem::temp_directory_path() /
                ("alaya_coll_disp_" + pid_str + "_" +
                 ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::remove_all(base);
    std::filesystem::create_directories(base);
    tmp_root_ = base;
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_root_, ec);
  }

  static auto make_random_vectors(uint64_t n, uint32_t dim, uint32_t seed = 42)
      -> std::vector<float> {
    std::vector<float> out(n * dim);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto &v : out) {
      v = dist(rng);
    }
    return out;
  }

  static auto sequential_labels(uint64_t n, uint64_t base = 0) -> std::vector<uint64_t> {
    std::vector<uint64_t> out(n);
    std::iota(out.begin(), out.end(), base);
    return out;
  }

  // Truncates / rewrites a value in collection_manifest.txt under the given
  // collection root. Used to fabricate disk_vamana / disk_laser manifests.
  static void rewrite_collection_manifest_index_type(const std::filesystem::path &coll,
                                                     const std::string &engine_str) {
    std::ofstream ofs(coll / "collection_manifest.txt");
    ofs << "version=1\n"
        << "dim=4\n"
        << "metric=L2\n"
        << "index_type=" << engine_str << "\n"
        << "next_segment_id=1\n";
  }

  std::filesystem::path tmp_root_;
};

// --------------------------------------------------------------------------
// Tasks 6.2 — 6.4: positive Flat dispatch tests.
// --------------------------------------------------------------------------

TEST_F(DiskCollectionFactoryDispatchTest, disk_collection_constructor_uses_factory_for_flat) {
  const auto coll = tmp_root_ / "coll";
  EXPECT_NO_THROW(DiskCollection(coll, 16, MetricType::L2, DiskIndexType::Flat));
  EXPECT_TRUE(std::filesystem::exists(coll / "collection_manifest.txt"));
  EXPECT_TRUE(std::filesystem::is_directory(coll / "segments"));
}

TEST_F(DiskCollectionFactoryDispatchTest, disk_collection_open_uses_factory_for_flat) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 64;
  const auto coll = tmp_root_ / "coll";
  auto vectors = make_random_vectors(kN, kDim, 1);
  auto labels = sequential_labels(kN, 100);
  std::vector<DiskSearchHit> baseline;
  {
    DiskCollection col(coll, kDim, MetricType::L2, DiskIndexType::Flat);
    col.add_batch(vectors.data(), labels.data(), kN);
    col.flush();
    auto query = make_random_vectors(1, kDim, 99);
    DiskSearchOptions opts;
    opts.top_k = 5;
    baseline = col.search(query.data(), opts);
  }
  // Reopen via factory and confirm hits are equivalent for the same query.
  auto col2 = DiskCollection::open(coll);
  EXPECT_EQ(col2.size(), kN);
  auto query = make_random_vectors(1, kDim, 99);
  DiskSearchOptions opts;
  opts.top_k = 5;
  auto hits = col2.search(query.data(), opts);
  ASSERT_EQ(hits.size(), baseline.size());
  for (size_t i = 0; i < hits.size(); ++i) {
    EXPECT_EQ(hits[i].label, baseline[i].label);
    EXPECT_EQ(hits[i].distance, baseline[i].distance);
  }
}

TEST_F(DiskCollectionFactoryDispatchTest, disk_collection_flush_uses_factory_for_flat) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 32;
  const auto coll = tmp_root_ / "coll";
  auto vectors = make_random_vectors(kN, kDim, 1);
  auto labels = sequential_labels(kN, 100);
  {
    DiskCollection col(coll, kDim, MetricType::L2, DiskIndexType::Flat);
    col.add_batch(vectors.data(), labels.data(), kN);
    col.flush();
  }
  // Reopen in a fresh DiskCollection and confirm search returns the expected
  // labels — proves the factory wrote a valid segment that the factory can
  // reopen.
  auto col2 = DiskCollection::open(coll);
  EXPECT_EQ(col2.size(), kN);
  auto query = make_random_vectors(1, kDim, 99);
  DiskSearchOptions opts;
  opts.top_k = 3;
  auto hits = col2.search(query.data(), opts);
  ASSERT_EQ(hits.size(), 3u);
  for (size_t i = 1; i < hits.size(); ++i) {
    EXPECT_LE(hits[i - 1].distance, hits[i].distance);
  }
}

// --------------------------------------------------------------------------
// Task 6.5: cross-segment uniqueness must NOT depend on dynamic_cast against
// DiskFlatSegmentSearcher. We cannot literally inject a non-Flat searcher
// (v1 only registers Flat in the factory), but we can prove the inventory
// path goes through `manifest.ids_file` by hand-building a segment directory
// (no DiskCollection mutator involved) and then triggering the cross-segment
// uniqueness code path against it.
// --------------------------------------------------------------------------

TEST_F(DiskCollectionFactoryDispatchTest,
       duplicate_label_check_does_not_depend_on_DiskFlatSegmentSearcher_dynamic_cast) {
  constexpr uint32_t kDim = 4;
  const auto coll = tmp_root_ / "coll";
  std::filesystem::create_directories(coll / "segments");

  // Hand-build segment seg_00000001 with label=42 — bypassing
  // DiskCollection's mutator path entirely.
  const auto seg1_dir = coll / "segments" / "seg_00000001";
  {
    DiskFlatBuilder b(kDim, MetricType::L2);
    std::vector<float> v(kDim, 1.0F);
    std::vector<uint64_t> l{42};
    b.add_batch(v.data(), l.data(), 1);
    b.finish(seg1_dir);
  }

  // Hand-write the collection manifest listing seg_00000001.
  {
    std::ofstream ofs(coll / "collection_manifest.txt");
    ofs << "version=1\n"
        << "dim=" << kDim << "\n"
        << "metric=L2\n"
        << "index_type=disk_flat\n"
        << "next_segment_id=2\n"
        << "segment=seg_00000001\n";
  }

  // Open through DiskCollection — the inventory step on flush() reads
  // seg_00000001's manifest.ids_file, NOT a downcast handle.
  auto col = DiskCollection::open(coll);
  EXPECT_EQ(col.size(), 1u);

  // Attempt to flush a pending payload that collides with the existing label.
  std::vector<float> v(kDim, 0.0F);
  std::vector<uint64_t> l{42};
  col.add_batch(v.data(), l.data(), 1);
  EXPECT_THROW(col.flush(), std::invalid_argument);
  // No new segment file appears.
  EXPECT_FALSE(std::filesystem::exists(coll / "segments" / "seg_00000002"));
}

// --------------------------------------------------------------------------
// Task 6.6: same shape but two segments, validates per-segment iteration.
// --------------------------------------------------------------------------

TEST_F(DiskCollectionFactoryDispatchTest,
       ids_file_label_inventory_detects_duplicate_across_segments) {
  constexpr uint32_t kDim = 4;
  const auto coll = tmp_root_ / "coll";
  std::filesystem::create_directories(coll / "segments");

  const auto seg1_dir = coll / "segments" / "seg_00000001";
  const auto seg2_dir = coll / "segments" / "seg_00000002";
  // Build seg_00000001: labels {10, 20, 30}.
  {
    DiskFlatBuilder b(kDim, MetricType::L2);
    auto v = make_random_vectors(3, kDim, 1);
    std::vector<uint64_t> l{10, 20, 30};
    b.add_batch(v.data(), l.data(), 3);
    b.finish(seg1_dir);
  }
  // Build seg_00000002: labels {40, 50, 60}.
  {
    DiskFlatBuilder b(kDim, MetricType::L2);
    auto v = make_random_vectors(3, kDim, 2);
    std::vector<uint64_t> l{40, 50, 60};
    b.add_batch(v.data(), l.data(), 3);
    b.finish(seg2_dir);
  }
  {
    std::ofstream ofs(coll / "collection_manifest.txt");
    ofs << "version=1\n"
        << "dim=" << kDim << "\n"
        << "metric=L2\n"
        << "index_type=disk_flat\n"
        << "next_segment_id=3\n"
        << "segment=seg_00000001\n"
        << "segment=seg_00000002\n";
  }

  auto col = DiskCollection::open(coll);
  EXPECT_EQ(col.size(), 6u);

  // Pending label 50 collides with seg_00000002 (the second segment), so the
  // inventory must walk every listed segment, not just the first.
  std::vector<float> v(kDim, 0.0F);
  std::vector<uint64_t> l{50};
  col.add_batch(v.data(), l.data(), 1);
  try {
    col.flush();
    FAIL() << "expected throw on duplicate cross-segment label";
  } catch (const std::invalid_argument &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("50"), std::string::npos) << msg;
  }
}

// --------------------------------------------------------------------------
// Tasks 6.7 — 6.10: Vamana now passes the C++ factory gate; Laser remains
// unsupported through the same dual-substring error contract.
// --------------------------------------------------------------------------

TEST_F(DiskCollectionFactoryDispatchTest, disk_collection_constructor_accepts_disk_vamana) {
  const auto coll = tmp_root_ / "coll_v";
  EXPECT_NO_THROW(DiskCollection(coll, 8, MetricType::L2, DiskIndexType::Vamana));
  auto manifest = CollectionManifest::load(coll / "collection_manifest.txt");
  EXPECT_EQ(manifest.index_type, DiskIndexType::Vamana);
}

TEST_F(DiskCollectionFactoryDispatchTest, disk_collection_constructor_rejects_disk_laser) {
  const auto coll = tmp_root_ / "coll_l";
  try {
    (void)DiskCollection(coll, 8, MetricType::L2, DiskIndexType::Laser);
    FAIL() << "expected throw on Laser";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("disk_laser"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not implemented in v1"), std::string::npos) << msg;
  }
  EXPECT_FALSE(std::filesystem::exists(coll));
}

TEST_F(DiskCollectionFactoryDispatchTest, disk_collection_open_accepts_empty_disk_vamana_manifest) {
  const auto coll = tmp_root_ / "coll_v";
  std::filesystem::create_directories(coll / "segments");
  rewrite_collection_manifest_index_type(coll, "disk_vamana");
  auto col = DiskCollection::open(coll);
  EXPECT_EQ(col.size(), 0u);
}

TEST_F(DiskCollectionFactoryDispatchTest, disk_collection_open_rejects_disk_laser_manifest) {
  const auto coll = tmp_root_ / "coll_l";
  std::filesystem::create_directories(coll / "segments");
  rewrite_collection_manifest_index_type(coll, "disk_laser");
  try {
    (void)DiskCollection::open(coll);
    FAIL() << "expected throw on disk_laser manifest";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("disk_laser"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not implemented in v1"), std::string::npos) << msg;
  }
}

// --------------------------------------------------------------------------
// Tasks 7.1 — 7.4: negative tests for ids file inventory.
// --------------------------------------------------------------------------

namespace {

// Helper: build a 1-segment collection containing label 1 so subsequent
// flushes will trigger the cross-segment inventory path.
void seed_one_segment_collection(const std::filesystem::path &coll, uint32_t dim) {
  DiskCollection col(coll, dim, MetricType::L2, DiskIndexType::Flat);
  std::vector<float> v(dim, 1.0F);
  std::vector<uint64_t> l{1};
  col.add_batch(v.data(), l.data(), 1);
  col.flush();
}

}  // namespace

// These tests must corrupt the ids file AFTER DiskCollection::open succeeds.
// The DiskFlatSegmentSearcher constructor (invoked by open) mmap's the ids
// file and validates its size at open time, so corrupting the file before
// open would short-circuit the test at open() with the searcher's pre-existing
// validation. The point of these tests is to exercise the FLUSH-time
// engine-agnostic inventory path that fresh-mmap's the file and re-validates.

TEST_F(DiskCollectionFactoryDispatchTest, flush_throws_on_missing_ids_file) {
  constexpr uint32_t kDim = 4;
  const auto coll = tmp_root_ / "coll";
  seed_one_segment_collection(coll, kDim);
  auto col = DiskCollection::open(coll);

  // Remove the ids file AFTER open. The held searcher mmap stays valid (it
  // referenced the file by inode); the flush-time inventory does a fresh
  // open that hits ENOENT.
  const auto ids_path = coll / "segments" / "seg_00000001" / "ids.u64.bin";
  std::filesystem::remove(ids_path);

  std::vector<float> v(kDim, 2.0F);
  std::vector<uint64_t> l{99};
  col.add_batch(v.data(), l.data(), 1);
  try {
    col.flush();
    FAIL() << "expected throw on missing ids_file";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("ids.u64.bin"), std::string::npos) << msg;
    EXPECT_NE(msg.find("seg_00000001"), std::string::npos) << msg;
  }
  // No second segment published.
  EXPECT_FALSE(std::filesystem::exists(coll / "segments" / "seg_00000002"));
}

TEST_F(DiskCollectionFactoryDispatchTest, flush_throws_on_truncated_ids_file) {
  constexpr uint32_t kDim = 4;
  const auto coll = tmp_root_ / "coll";
  // Seed with two ids so the inventory size check has something to compare
  // against (count=2 → expected 16 bytes).
  {
    DiskCollection col(coll, kDim, MetricType::L2, DiskIndexType::Flat);
    std::vector<float> v(kDim * 2, 1.0F);
    std::vector<uint64_t> l{1, 2};
    col.add_batch(v.data(), l.data(), 2);
    col.flush();
  }
  auto col = DiskCollection::open(coll);

  // Truncate AFTER open succeeds. New flush-time mmap will see 9 bytes vs
  // expected 16 and throw.
  const auto ids_path = coll / "segments" / "seg_00000001" / "ids.u64.bin";
  std::filesystem::resize_file(ids_path, 9);

  std::vector<float> v(kDim, 2.0F);
  std::vector<uint64_t> l{99};
  col.add_batch(v.data(), l.data(), 1);
  try {
    col.flush();
    FAIL() << "expected throw on truncated ids_file";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("size mismatch"), std::string::npos) << msg;
    EXPECT_NE(msg.find("seg_00000001"), std::string::npos) << msg;
  }
}

TEST_F(DiskCollectionFactoryDispatchTest, flush_throws_on_oversized_ids_file) {
  constexpr uint32_t kDim = 4;
  const auto coll = tmp_root_ / "coll";
  seed_one_segment_collection(coll, kDim);
  auto col = DiskCollection::open(coll);

  // count=1 → expected 8 bytes; pad to 16 bytes after open succeeds.
  const auto ids_path = coll / "segments" / "seg_00000001" / "ids.u64.bin";
  std::filesystem::resize_file(ids_path, 16);

  std::vector<float> v(kDim, 2.0F);
  std::vector<uint64_t> l{99};
  col.add_batch(v.data(), l.data(), 1);
  try {
    col.flush();
    FAIL() << "expected throw on oversized ids_file";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("size mismatch"), std::string::npos) << msg;
    EXPECT_NE(msg.find("seg_00000001"), std::string::npos) << msg;
  }
}

TEST_F(DiskCollectionFactoryDispatchTest, flush_throws_on_symlinked_ids_file) {
  constexpr uint32_t kDim = 4;
  const auto coll = tmp_root_ / "coll";
  seed_one_segment_collection(coll, kDim);
  auto col = DiskCollection::open(coll);

  // Build a sibling decoy file with the same byte size as ids.u64.bin.
  const auto ids_path = coll / "segments" / "seg_00000001" / "ids.u64.bin";
  const auto decoy = tmp_root_ / "decoy_ids.bin";
  {
    const uint64_t fake_id = 7;
    std::ofstream ofs(decoy, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(&fake_id), sizeof(fake_id));
  }
  // Replace the file with a symlink AFTER open succeeded. flush-time mmap
  // hits ELOOP via O_NOFOLLOW.
  std::filesystem::remove(ids_path);
  std::error_code ec;
  std::filesystem::create_symlink(decoy, ids_path, ec);
  if (ec) {
    GTEST_SKIP() << "symlink creation failed: " << ec.message();
  }

  std::vector<float> v(kDim, 2.0F);
  std::vector<uint64_t> l{99};
  col.add_batch(v.data(), l.data(), 1);
  EXPECT_THROW(col.flush(), std::runtime_error);
}

// --------------------------------------------------------------------------
// Tasks 8.1 — 8.2: hot-path invariants. The Flat search must not open new
// files or remap memory after construction.
// --------------------------------------------------------------------------

namespace {

auto count_open_fds() -> size_t {
  size_t n = 0;
  std::error_code ec;
  for ([[maybe_unused]] const auto &entry :
       std::filesystem::directory_iterator("/proc/self/fd", ec)) {
    ++n;
  }
  return n;
}

}  // namespace

TEST_F(DiskCollectionFactoryDispatchTest, flat_search_inner_loop_does_not_open_files) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 64;
  const auto seg_dir = tmp_root_ / "seg_00000001";
  {
    DiskFlatBuilder b(kDim, MetricType::L2);
    auto v = make_random_vectors(kN, kDim, 1);
    auto l = sequential_labels(kN, 0);
    b.add_batch(v.data(), l.data(), kN);
    b.finish(seg_dir);
  }
  DiskFlatSegmentSearcher searcher(seg_dir);

  // Warm up: do one search so any lazy initialization completes.
  auto query = make_random_vectors(1, kDim, 99);
  DiskSearchOptions opts;
  opts.top_k = 4;
  (void)searcher.search(query.data(), opts);

  const size_t before = count_open_fds();
  for (int i = 0; i < 1000; ++i) {
    (void)searcher.search(query.data(), opts);
  }
  const size_t after = count_open_fds();
  EXPECT_EQ(before, after)
      << "search() must not open any new file descriptors after construction "
         "(before=" << before << " after=" << after << ")";
}

TEST_F(DiskCollectionFactoryDispatchTest, flat_search_inner_loop_does_not_remap) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 64;
  const auto seg_dir = tmp_root_ / "seg_00000001";
  {
    DiskFlatBuilder b(kDim, MetricType::L2);
    auto v = make_random_vectors(kN, kDim, 1);
    auto l = sequential_labels(kN, 0);
    b.add_batch(v.data(), l.data(), kN);
    b.finish(seg_dir);
  }
  DiskFlatSegmentSearcher searcher(seg_dir);
  // The labels() pointer is the start of the ids_mmap region. If the searcher
  // remapped during search(), this address would change.
  const uint64_t *labels_addr_before = searcher.labels();
  auto query = make_random_vectors(1, kDim, 99);
  DiskSearchOptions opts;
  opts.top_k = 4;
  for (int i = 0; i < 1000; ++i) {
    (void)searcher.search(query.data(), opts);
  }
  const uint64_t *labels_addr_after = searcher.labels();
  EXPECT_EQ(labels_addr_before, labels_addr_after)
      << "search() must not remap the ids mmap region";
}

}  // namespace
}  // namespace alaya::disk
