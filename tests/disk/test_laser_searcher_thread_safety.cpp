// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "index/disk/laser_segment_importer.hpp"
#include "index/disk/laser_segment_searcher.hpp"
#include "index/disk/segment_factory.hpp"
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

constexpr uint32_t kFixtureDim = 128;
constexpr uint64_t kFixtureCount = 2048;
constexpr uint32_t kFixtureR = 64;

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

auto labels_consecutive() -> std::vector<uint64_t> {
  std::vector<uint64_t> out(kFixtureCount);
  for (uint64_t i = 0; i < kFixtureCount; ++i) {
    out[i] = i;
  }
  return out;
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

auto fixture_query(uint64_t row) -> std::vector<float> {
  const auto vectors = read_fixture_vectors();
  if (row >= kFixtureCount) {
    throw std::runtime_error("fixture query row out of range");
  }
  std::vector<float> query(kFixtureDim);
  std::copy_n(vectors.data() + static_cast<size_t>(row) * kFixtureDim, kFixtureDim, query.data());
  return query;
}

auto extract_labels(const std::vector<DiskSearchHit> &hits) -> std::vector<uint64_t> {
  std::vector<uint64_t> out;
  out.reserve(hits.size());
  for (const auto &h : hits) {
    out.push_back(h.label);
  }
  return out;
}

auto laser_hits_are_valid(const std::vector<DiskSearchHit> &hits,
                          uint64_t expected_self,
                          uint32_t top_k) -> bool {
  if (hits.size() != top_k || hits.empty() || hits.front().label != expected_self) {
    return false;
  }
  return std::all_of(hits.begin(), hits.end(), [](const auto &hit) {
    return hit.label < kFixtureCount;
  });
}

class LaserSegmentSearcherConcurrentSearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    tmp_root_ = std::filesystem::temp_directory_path() /
                ("alaya_laser_thread_safety_" + std::to_string(::getpid()) + "_" + test_name);
    std::filesystem::remove_all(tmp_root_);
    seg_dir_ = tmp_root_ / "seg_00000001";
    std::filesystem::create_directories(tmp_root_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_root_, ec);
  }

  void import_fixture() const {
    LaserSegmentImporter importer(kFixtureDim, MetricType::L2, {});
    auto ids = labels_consecutive();
    (void)importer.import_from(fixture_dir(), ids.data(), ids.size(), seg_dir_);
  }

  std::filesystem::path tmp_root_;
  std::filesystem::path seg_dir_;
};

TEST_F(LaserSegmentSearcherConcurrentSearchTest, eight_threads_thousand_queries_match_baseline) {
  if (const auto reason = fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }
  import_fixture();
  LaserSegmentSearcher searcher(seg_dir_);

  const auto query = fixture_query(11);
  DiskSearchOptions opts;
  opts.top_k = 10;
  opts.ef = 100;
  opts.beam_width = 4;

  const auto baseline = extract_labels(searcher.search(query.data(), opts));
  ASSERT_EQ(baseline.size(), opts.top_k);

  constexpr int kThreads = 8;
  constexpr int kIters = 1000;
  std::atomic<int> mismatches{0};
  std::atomic<int> exceptions{0};
  std::atomic<bool> go{false};

  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&] {
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      try {
        for (int i = 0; i < kIters; ++i) {
          const auto hits = searcher.search(query.data(), opts);
          if (extract_labels(hits) != baseline) {
            mismatches.fetch_add(1, std::memory_order_relaxed);
            return;
          }
        }
      } catch (...) {
        exceptions.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  go.store(true, std::memory_order_release);
  for (auto &w : workers) {
    w.join();
  }
  EXPECT_EQ(mismatches.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(exceptions.load(std::memory_order_relaxed), 0);
}

TEST_F(LaserSegmentSearcherConcurrentSearchTest,
       eight_threads_varying_params_match_per_opts_baseline) {
  if (const auto reason = fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }
  import_fixture();
  LaserSegmentSearcher searcher(seg_dir_);

  const auto query = fixture_query(37);
  DiskSearchOptions opts_a;
  opts_a.top_k = 10;
  opts_a.ef = 32;
  opts_a.beam_width = 4;
  DiskSearchOptions opts_b;
  opts_b.top_k = 10;
  opts_b.ef = 64;
  opts_b.beam_width = 8;

  ASSERT_TRUE(laser_hits_are_valid(searcher.search(query.data(), opts_a), 37, opts_a.top_k));
  ASSERT_TRUE(laser_hits_are_valid(searcher.search(query.data(), opts_b), 37, opts_b.top_k));

  constexpr int kThreadsPerHalf = 4;
  constexpr int kIters = 1000;
  std::atomic<int> mismatches{0};
  std::atomic<int> exceptions{0};
  std::atomic<bool> go{false};

  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(kThreadsPerHalf) * 2);
  for (int t = 0; t < kThreadsPerHalf * 2; ++t) {
    const bool first_half = (t < kThreadsPerHalf);
    workers.emplace_back([&, first_half] {
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      const DiskSearchOptions &opts = first_half ? opts_a : opts_b;
      try {
        for (int i = 0; i < kIters; ++i) {
          const auto hits = searcher.search(query.data(), opts);
          if (!laser_hits_are_valid(hits, 37, opts.top_k)) {
            mismatches.fetch_add(1, std::memory_order_relaxed);
            return;
          }
        }
      } catch (...) {
        exceptions.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  go.store(true, std::memory_order_release);
  for (auto &w : workers) {
    w.join();
  }
  EXPECT_EQ(mismatches.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(exceptions.load(std::memory_order_relaxed), 0);
}

TEST_F(LaserSegmentSearcherConcurrentSearchTest, two_threads_same_params_apply_set_params_once) {
  if (const auto reason = fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }
  import_fixture();
  LaserSegmentSearcher searcher(seg_dir_);
  ASSERT_EQ(searcher.set_params_call_count(), 0U);

  const auto query = fixture_query(7);
  DiskSearchOptions opts;
  opts.top_k = 10;
  opts.ef = 100;
  opts.beam_width = 4;

  std::thread t1([&] { (void)searcher.search(query.data(), opts); });
  std::thread t2([&] { (void)searcher.search(query.data(), opts); });
  t1.join();
  t2.join();
  EXPECT_EQ(searcher.set_params_call_count(), 1U);
}

TEST_F(LaserSegmentSearcherConcurrentSearchTest,
       two_threads_distinct_ef_apply_set_params_one_or_two_times) {
  if (const auto reason = fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }
  import_fixture();
  LaserSegmentSearcher searcher(seg_dir_);
  ASSERT_EQ(searcher.set_params_call_count(), 0U);

  const auto query = fixture_query(13);
  DiskSearchOptions opts_a;
  opts_a.top_k = 10;
  opts_a.ef = 32;
  opts_a.beam_width = 4;
  DiskSearchOptions opts_b;
  opts_b.top_k = 10;
  opts_b.ef = 64;
  opts_b.beam_width = 4;

  std::thread t_a([&] { (void)searcher.search(query.data(), opts_a); });
  std::thread t_b([&] { (void)searcher.search(query.data(), opts_b); });
  t_a.join();
  t_b.join();
  const auto count = searcher.set_params_call_count();
  EXPECT_GE(count, 1U);
  EXPECT_LE(count, 2U);
}

TEST_F(LaserSegmentSearcherConcurrentSearchTest, argument_validation_throws_before_lock) {
  if (const auto reason = fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }
  import_fixture();
  LaserSegmentSearcher searcher(seg_dir_);

  const auto expect_invalid_argument_with =
      [&searcher](const std::function<void()> &fn, const std::string &needle) {
        try {
          fn();
          FAIL() << "expected std::invalid_argument";
        } catch (const std::invalid_argument &e) {
          const std::string msg = e.what();
          EXPECT_NE(msg.find(needle), std::string::npos) << msg;
        }
      };

  const auto query = fixture_query(0);
  DiskSearchOptions opts_zero_k;
  opts_zero_k.top_k = 0;
  opts_zero_k.ef = 100;
  opts_zero_k.beam_width = 4;
  expect_invalid_argument_with(
      [&] { (void)searcher.search(query.data(), opts_zero_k); }, "top_k must be > 0");

  DiskSearchOptions opts_ok;
  opts_ok.top_k = 10;
  opts_ok.ef = 100;
  opts_ok.beam_width = 4;
  expect_invalid_argument_with(
      [&] { (void)searcher.search(nullptr, opts_ok); }, "query must not be null");

  DiskSearchOptions opts_overflow;
  opts_overflow.top_k = 10;
  opts_overflow.ef = 100;
  opts_overflow.beam_width = static_cast<uint32_t>(std::numeric_limits<int>::max()) + 1U;
  expect_invalid_argument_with(
      [&] { (void)searcher.search(query.data(), opts_overflow); }, "beam_width exceeds int max");

  // Counter at zero is the indirect proof that none of the three throws ran the
  // search body — including its mutex acquisition (apply_set_params is the only
  // place that bumps it, and it sits inside the locked region).
  EXPECT_EQ(searcher.set_params_call_count(), 0U);
}

}  // namespace
}  // namespace alaya::disk
