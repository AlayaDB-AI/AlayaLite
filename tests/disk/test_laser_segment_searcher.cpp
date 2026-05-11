// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "index/disk/laser_segment_importer.hpp"
#include "index/disk/laser_segment_searcher.hpp"
#include "index/disk/segment_factory.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "utils/metric_type.hpp"

#ifndef ALAYA_LASER_FIXTURE_DIR
  #define ALAYA_LASER_FIXTURE_DIR ""
#endif

#ifndef ALAYA_LASER_FIXTURE_PREFIX
  #define ALAYA_LASER_FIXTURE_PREFIX "dsqg_seg_00000001"
#endif

namespace alaya::disk {
namespace {

static void expect_runtime_message_contains(const std::function<void()> &fn,
                                            const std::vector<std::string> &needles) {
  try {
    fn();
    FAIL() << "expected std::runtime_error";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    for (const auto &needle : needles) {
      EXPECT_NE(msg.find(needle), std::string::npos) << msg;
    }
  }
}

#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0

constexpr uint32_t kFixtureDim = 128;
constexpr uint64_t kFixtureCount = 2048;
constexpr uint32_t kFixtureR = 64;
constexpr uint32_t kTopK = 10;

auto fixture_dir() -> std::filesystem::path {
  return std::filesystem::path(ALAYA_LASER_FIXTURE_DIR);
}

auto fixture_prefix() -> std::string { return std::string(ALAYA_LASER_FIXTURE_PREFIX); }

auto fixture_index_name() -> std::string {
  return fixture_prefix() + "_R" + std::to_string(kFixtureR) + "_MD" + std::to_string(kFixtureDim) +
         ".index";
}

auto fixture_required_artifacts() -> std::vector<std::string> {
  const auto index = fixture_index_name();
  return {
      index,
      index + "_rotator",
      index + "_cache_ids",
      index + "_cache_nodes",
  };
}

auto fixture_has_required_files(const std::filesystem::path &dir) -> bool {
  if (dir.empty()) {
    return false;
  }

  auto required = fixture_required_artifacts();
  required.push_back(fixture_prefix() + "_input.fbin");
  return std::all_of(required.begin(), required.end(), [&](const auto &name) {
    std::error_code ec;
    const auto path = dir / name;
    if (!std::filesystem::is_regular_file(path, ec) || ec) {
      return false;
    }
    const auto size = std::filesystem::file_size(path, ec);
    return !ec && size > 0;
  });
}

auto fixture_skip_reason() -> std::string {
  if (!engine_supported_v1(DiskIndexType::Laser)) {
    return "disk_laser is not registered in this build";
  }
  const auto dir = fixture_dir();
  if (!fixture_has_required_files(dir)) {
    return "LASER fixture is missing or incomplete under " + dir.string();
  }
  return {};
}

auto labels(uint64_t base = 0, uint64_t step = 1) -> std::vector<uint64_t> {
  std::vector<uint64_t> out(kFixtureCount);
  for (uint64_t i = 0; i < kFixtureCount; ++i) {
    out[i] = base + step * i;
  }
  return out;
}

auto label_set(const std::vector<uint64_t> &ids) -> std::unordered_set<uint64_t> {
  return {ids.begin(), ids.end()};
}

auto is_nan_bits(float value) -> bool {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return (bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0;
}

auto read_fixture_vectors() -> std::vector<float> {
  const auto path = fixture_dir() / (fixture_prefix() + "_input.fbin");
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open fixture vectors: " + path.string());
  }

  int32_t count = 0;
  int32_t dim = 0;
  input.read(reinterpret_cast<char *>(&count), sizeof(count));
  input.read(reinterpret_cast<char *>(&dim), sizeof(dim));
  if (count != static_cast<int32_t>(kFixtureCount) || dim != static_cast<int32_t>(kFixtureDim)) {
    throw std::runtime_error("unexpected fixture vector header in " + path.string());
  }

  std::vector<float> out(static_cast<size_t>(count) * static_cast<size_t>(dim));
  input.read(reinterpret_cast<char *>(out.data()),
             static_cast<std::streamsize>(out.size() * sizeof(float)));
  if (!input) {
    throw std::runtime_error("short fixture vector read: " + path.string());
  }
  return out;
}

class LaserSegmentSearcherTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    tmp_root_ = std::filesystem::temp_directory_path() /
                ("alaya_laser_searcher_" + std::to_string(::getpid()) + "_" + test_name);
    std::filesystem::remove_all(tmp_root_);
    seg_dir_ = tmp_root_ / "seg_00000001";
    src_dir_ = tmp_root_ / "src";
    std::filesystem::create_directories(tmp_root_);
    std::filesystem::create_directories(src_dir_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_root_, ec);
  }

  static auto base_manifest(MetricType metric = MetricType::L2) -> SegmentManifest {
    SegmentManifest manifest;
    manifest.segment_id = "seg_00000001";
    manifest.index_type = DiskIndexType::Laser;
    manifest.metric = metric;
    manifest.dim = kFixtureDim;
    manifest.count = 42;
    manifest.ids_file = "ids.u64.bin";
    manifest.vectors_file = "";
    manifest.x_extras["x_laser_filename_prefix"] = "dsqg_seg_00000001";
    manifest.x_extras["x_R"] = std::to_string(kFixtureR);
    manifest.x_extras["x_main_dim"] = std::to_string(kFixtureDim);
    manifest.x_extras["x_laser_index_file"] = "dsqg_seg_00000001_R64_MD128.index";
    manifest.x_extras["x_laser_search_dram_budget_gb"] = "0.5";
    return manifest;
  }

  void write_manifest(const SegmentManifest &manifest) const {
    std::filesystem::create_directories(seg_dir_);
    manifest.save(seg_dir_ / "manifest.txt");
  }

  void write_ids(uint64_t count) const {
    std::filesystem::create_directories(seg_dir_);
    std::ofstream out(seg_dir_ / "ids.u64.bin", std::ios::binary);
    for (uint64_t i = 0; i < count; ++i) {
      const uint64_t label = 1000 + i;
      out.write(reinterpret_cast<const char *>(&label), sizeof(label));
    }
  }

  void write_index_with_count(uint64_t count) const {
    std::filesystem::create_directories(seg_dir_);
    const auto index_path = seg_dir_ / "dsqg_seg_00000001_R64_MD128.index";
    std::ofstream out(index_path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(&count), sizeof(count));
  }

  auto import_fixture_segment(const std::vector<uint64_t> &ids,
                              const std::filesystem::path &src_dir = fixture_dir()) const
      -> SegmentManifest {
    LaserSegmentImporter importer(kFixtureDim, MetricType::L2, {});
    return importer.import_from(src_dir, ids.data(), ids.size(), seg_dir_);
  }

  auto fixture_query(uint64_t row = 0) const -> std::vector<float> {
    const auto vectors = read_fixture_vectors();
    if (row >= kFixtureCount) {
      throw std::runtime_error("fixture query row out of range");
    }
    std::vector<float> query(kFixtureDim);
    std::copy_n(vectors.data() + static_cast<size_t>(row) * kFixtureDim, kFixtureDim, query.data());
    return query;
  }

  void copy_required_fixture_artifacts(const std::filesystem::path &dst_dir) const {
    std::filesystem::create_directories(dst_dir);
    for (const auto &name : fixture_required_artifacts()) {
      std::error_code ec;
      std::filesystem::copy_file(fixture_dir() / name, dst_dir / name, ec);
      if (ec) {
        throw std::runtime_error("failed to copy fixture artifact " + name + ": " + ec.message());
      }
    }
  }

  std::filesystem::path tmp_root_;
  std::filesystem::path seg_dir_;
  std::filesystem::path src_dir_;
};

TEST_F(LaserSegmentSearcherTest, loads_native_index) {
  if (const auto reason = fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }

  auto ids = labels();
  const auto manifest = import_fixture_segment(ids);

  LaserSegmentSearcher searcher(seg_dir_);

  // searcher.size() reflects QuantizedGraph::num_vertices() because the
  // constructor cross-validates index metadata count against manifest count
  // (laser_segment_searcher.hpp:301-307); this assertion plus the constructor
  // returning is the D9 "loads native index" check.
  EXPECT_EQ(manifest.count, kFixtureCount);
  EXPECT_EQ(searcher.type(), DiskIndexType::Laser);
  EXPECT_EQ(searcher.dim(), kFixtureDim);
  EXPECT_EQ(searcher.size(), kFixtureCount);
}

TEST_F(LaserSegmentSearcherTest, smoke_search) {
  if (const auto reason = fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }

  auto ids = labels();
  const auto expected_labels = label_set(ids);
  import_fixture_segment(ids);
  LaserSegmentSearcher searcher(seg_dir_);

  DiskSearchOptions opts;
  opts.top_k = kTopK;
  opts.ef = 64;
  opts.beam_width = 4;
  const auto query = fixture_query(11);
  const auto hits = searcher.search(query.data(), opts);

  ASSERT_FALSE(hits.empty());
  EXPECT_LE(hits.size(), opts.top_k);
  for (const auto &hit : hits) {
    EXPECT_TRUE(expected_labels.contains(hit.label)) << hit.label;
    EXPECT_TRUE(is_nan_bits(hit.distance)) << hit.distance;
  }
}

TEST_F(LaserSegmentSearcherTest, top_k_larger_than_count_caps) {
  if (const auto reason = fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }

  auto ids = labels();
  const auto expected_labels = label_set(ids);
  import_fixture_segment(ids);
  LaserSegmentSearcher searcher(seg_dir_);

  DiskSearchOptions opts;
  opts.top_k = static_cast<uint32_t>(kFixtureCount + 1);
  opts.ef = 64;
  opts.beam_width = 4;
  const auto query = fixture_query(3);
  const auto hits = searcher.search(query.data(), opts);

  EXPECT_EQ(hits.size(), kFixtureCount);
  for (const auto &hit : hits) {
    EXPECT_TRUE(expected_labels.contains(hit.label)) << hit.label;
    EXPECT_TRUE(is_nan_bits(hit.distance)) << hit.distance;
  }
}

TEST_F(LaserSegmentSearcherTest, external_label_mapping) {
  if (const auto reason = fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }

  auto ids = labels(1000, 1000);
  const auto expected_labels = label_set(ids);
  import_fixture_segment(ids);
  LaserSegmentSearcher searcher(seg_dir_);

  DiskSearchOptions opts;
  opts.top_k = kTopK;
  opts.ef = 64;
  const auto query = fixture_query(37);
  const auto hits = searcher.search(query.data(), opts);

  ASSERT_FALSE(hits.empty());
  for (const auto &hit : hits) {
    EXPECT_TRUE(expected_labels.contains(hit.label)) << hit.label;
    EXPECT_GE(hit.label, 1000U);
    EXPECT_EQ(hit.label % 1000U, 0U);
  }
}

TEST_F(LaserSegmentSearcherTest, distance_is_nan_in_v1) {
  if (const auto reason = fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }

  auto ids = labels();
  import_fixture_segment(ids);
  const auto manifest = SegmentManifest::load(seg_dir_ / "manifest.txt");
  ASSERT_EQ(manifest.x_extras.at("x_laser_distance_field_supported"), "false");

  LaserSegmentSearcher searcher(seg_dir_);
  DiskSearchOptions opts;
  opts.top_k = kTopK;
  opts.ef = 64;
  const auto query = fixture_query(73);
  const auto hits = searcher.search(query.data(), opts);

  ASSERT_FALSE(hits.empty());
  for (const auto &hit : hits) {
    EXPECT_TRUE(is_nan_bits(hit.distance)) << hit.distance;
  }
}

TEST_F(LaserSegmentSearcherTest, set_params_skipped_on_repeat) {
  if (const auto reason = fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }

  auto ids = labels();
  import_fixture_segment(ids);
  LaserSegmentSearcher searcher(seg_dir_);
  const auto query = fixture_query(19);

  DiskSearchOptions opts;
  opts.top_k = kTopK;
  opts.ef = 64;
  opts.beam_width = 4;

  EXPECT_EQ(searcher.set_params_call_count(), 0U);
  (void)searcher.search(query.data(), opts);
  EXPECT_EQ(searcher.set_params_call_count(), 1U);
  (void)searcher.search(query.data(), opts);
  EXPECT_EQ(searcher.set_params_call_count(), 1U);
}

TEST_F(LaserSegmentSearcherTest, set_params_called_on_change) {
  if (const auto reason = fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }

  auto ids = labels();
  import_fixture_segment(ids);
  LaserSegmentSearcher searcher(seg_dir_);
  const auto query = fixture_query(23);

  DiskSearchOptions first;
  first.top_k = kTopK;
  first.ef = 32;
  first.beam_width = 4;

  DiskSearchOptions second = first;
  second.ef = 64;
  second.beam_width = 8;

  (void)searcher.search(query.data(), first);
  EXPECT_EQ(searcher.set_params_call_count(), 1U);
  (void)searcher.search(query.data(), second);
  EXPECT_EQ(searcher.set_params_call_count(), 2U);
}

TEST_F(LaserSegmentSearcherTest, search_does_not_reopen_files) {
  const auto repo_root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
  const auto header_path = repo_root / "include" / "index" / "disk" / "laser_segment_searcher.hpp";
  std::ifstream input(header_path);
  ASSERT_TRUE(input.is_open()) << header_path;
  const std::string source((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());

  const auto start = source.find("auto search(const float *query");
  ASSERT_NE(start, std::string::npos);
  const auto end = source.find("  auto size() const", start);
  ASSERT_NE(end, std::string::npos);
  const auto body = source.substr(start, end - start);

  for (const auto *forbidden :
       {"SegmentManifest::load", "MMapFile", "::open", "mmap", "load_disk_index", "kMetricMap"}) {
    EXPECT_EQ(body.find(forbidden), std::string::npos) << forbidden << " found in search()";
  }
}

TEST_F(LaserSegmentSearcherTest, ids_file_size_mismatch_throws) {
  if (const auto reason = fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }

  auto ids = labels();
  import_fixture_segment(ids);
  std::filesystem::resize_file(seg_dir_ / "ids.u64.bin", (kFixtureCount - 1) * sizeof(uint64_t));

  expect_runtime_message_contains(
      [&] {
        LaserSegmentSearcher searcher(seg_dir_);
      },
      {"ids file size mismatch",
       std::to_string(kFixtureCount * sizeof(uint64_t)),
       seg_dir_.string()});
}

TEST_F(LaserSegmentSearcherTest, works_without_optional_medoids_or_pca) {
  if (const auto reason = fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }

  copy_required_fixture_artifacts(src_dir_);
  auto ids = labels();
  import_fixture_segment(ids, src_dir_);
  const auto manifest = SegmentManifest::load(seg_dir_ / "manifest.txt");
  EXPECT_FALSE(manifest.x_extras.contains("x_laser_medoids_file"));
  EXPECT_FALSE(manifest.x_extras.contains("x_laser_medoids_indices_file"));
  EXPECT_FALSE(manifest.x_extras.contains("x_laser_pca_file"));

  LaserSegmentSearcher searcher(seg_dir_);
  DiskSearchOptions opts;
  opts.top_k = kTopK;
  opts.ef = 64;
  const auto query = fixture_query(101);
  const auto hits = searcher.search(query.data(), opts);

  EXPECT_FALSE(hits.empty());
}

TEST_F(LaserSegmentSearcherTest, cos_metric_rejected) {
  write_manifest(base_manifest(MetricType::COS));

  expect_runtime_message_contains(
      [&] {
        LaserSegmentSearcher searcher(seg_dir_);
      },
      {"cos", "not implemented in v1"});
}

TEST_F(LaserSegmentSearcherTest, x_laser_filename_prefix_missing_throws) {
  if (!engine_supported_v1(DiskIndexType::Laser)) {
    GTEST_SKIP() << "Laser factory gate is not enabled until Phase 5";
  }

  auto manifest = base_manifest();
  manifest.x_extras.erase("x_laser_filename_prefix");
  write_manifest(manifest);

  expect_runtime_message_contains(
      [&] {
        LaserSegmentSearcher searcher(seg_dir_);
      },
      {"x_laser_filename_prefix missing", seg_dir_.string()});
}

TEST_F(LaserSegmentSearcherTest, manifest_count_disagrees_with_index_metadata_throws) {
  if (!engine_supported_v1(DiskIndexType::Laser)) {
    GTEST_SKIP() << "Laser factory gate is not enabled until Phase 5";
  }

  write_manifest(base_manifest());
  write_ids(42);
  write_index_with_count(43);

  expect_runtime_message_contains(
      [&] {
        LaserSegmentSearcher searcher(seg_dir_);
      },
      {"42", "43", "dsqg_seg_00000001_R64_MD128.index", seg_dir_.string()});
}

#else

TEST(LaserSegmentSearcherUnsupportedBuildTest, constructor_throws_dual_substring) {
  expect_runtime_message_contains(
      [&] {
        LaserSegmentSearcher searcher(std::filesystem::path("/tmp/seg_00000001"));
      },
      {"disk_laser", "not implemented in v1"});
}

#endif

}  // namespace
}  // namespace alaya::disk
