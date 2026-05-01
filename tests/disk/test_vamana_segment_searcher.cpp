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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <future>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "index/disk/vamana_segment_builder.hpp"
#include "index/disk/vamana_segment_searcher.hpp"
#include "simd/distance_l2.hpp"
#include "utils/metric_type.hpp"

namespace alaya::disk {
namespace {

class VamanaSegmentSearcherTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    tmp_root_ = std::filesystem::temp_directory_path() /
                ("alaya_vamana_searcher_" + std::to_string(::getpid()) + "_" + test_name);
    std::filesystem::remove_all(tmp_root_);
    seg_parent_ = tmp_root_ / "segments";
    std::filesystem::create_directories(seg_parent_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_root_, ec);
  }

  static auto make_vectors(uint64_t n, uint32_t dim, uint32_t seed = 42)
      -> std::vector<float> {
    std::vector<float> out(n * dim);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto &v : out) {
      v = dist(rng);
    }
    return out;
  }

  static auto labels(uint64_t n, uint64_t base = 1000) -> std::vector<uint64_t> {
    std::vector<uint64_t> out(n);
    std::iota(out.begin(), out.end(), base);
    return out;
  }

  static auto params() -> VamanaSegmentBuildParams {
    VamanaSegmentBuildParams p;
    p.R = 16;
    p.L = 64;
    p.num_threads = 1;
    return p;
  }

  auto build_segment(uint64_t n,
                     uint32_t dim,
                     uint32_t seed,
                     uint64_t label_base = 1000) -> std::filesystem::path {
    vectors_ = make_vectors(n, dim, seed);
    labels_ = labels(n, label_base);
    const auto seg_dir = seg_parent_ / "seg_00000001";
    VamanaSegmentBuilder builder(dim, MetricType::L2, params());
    builder.add_batch(vectors_.data(), labels_.data(), n);
    builder.finish(seg_dir);
    return seg_dir;
  }

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

  std::filesystem::path tmp_root_;
  std::filesystem::path seg_parent_;
  std::vector<float> vectors_;
  std::vector<uint64_t> labels_;
};

TEST_F(VamanaSegmentSearcherTest, loads_segment_round_trip) {
  const auto seg_dir = build_segment(512, 16, 1);
  VamanaSegmentSearcher searcher(seg_dir);
  EXPECT_EQ(searcher.dim(), 16u);
  EXPECT_EQ(searcher.size(), 512u);
  EXPECT_EQ(searcher.type(), DiskIndexType::Vamana);
}

TEST_F(VamanaSegmentSearcherTest, smoke_search_l2) {
  constexpr uint32_t kDim = 16;
  const auto seg_dir = build_segment(512, kDim, 2);
  VamanaSegmentSearcher searcher(seg_dir);
  DiskSearchOptions opts;
  opts.top_k = 10;
  opts.ef = 64;
  const auto hits = searcher.search(vectors_.data(), opts);
  ASSERT_EQ(hits.size(), 10u);
  std::unordered_set<uint64_t> label_set(labels_.begin(), labels_.end());
  for (const auto &hit : hits) {
    EXPECT_TRUE(label_set.contains(hit.label));
    EXPECT_TRUE(std::isfinite(hit.distance));
    EXPECT_GE(hit.distance, 0.0F);
  }
  for (size_t i = 1; i < hits.size(); ++i) {
    EXPECT_LE(hits[i - 1].distance, hits[i].distance);
  }
}

TEST_F(VamanaSegmentSearcherTest, top_k_larger_than_count_caps_before_greedy_search) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 32;
  const auto seg_dir = build_segment(kN, kDim, 12);
  VamanaSegmentSearcher searcher(seg_dir);
  DiskSearchOptions opts;
  opts.top_k = 200;
  opts.ef = 10;

  const auto hits = searcher.search(vectors_.data(), opts);
  EXPECT_EQ(hits.size(), kN);
}

TEST_F(VamanaSegmentSearcherTest, non_finite_query_throws) {
  constexpr uint32_t kDim = 8;
  const auto seg_dir = build_segment(128, kDim, 13);
  VamanaSegmentSearcher searcher(seg_dir);
  DiskSearchOptions opts;
  opts.top_k = 10;
  opts.ef = 64;
  std::vector<float> query(vectors_.begin(), vectors_.begin() + kDim);
  query[3] = std::numeric_limits<float>::quiet_NaN();

  EXPECT_THROW((void)searcher.search(query.data(), opts), std::invalid_argument);
}

TEST_F(VamanaSegmentSearcherTest, concurrent_search_results_are_stable) {
  constexpr uint32_t kDim = 16;
  const auto seg_dir = build_segment(512, kDim, 14);
  VamanaSegmentSearcher searcher(seg_dir);
  DiskSearchOptions opts;
  opts.top_k = 10;
  opts.ef = 64;
  const auto query = std::vector<float>(vectors_.begin() + 7 * kDim,
                                        vectors_.begin() + 8 * kDim);
  const auto baseline = searcher.search(query.data(), opts);

  std::vector<std::future<std::vector<DiskSearchHit>>> futures;
  futures.reserve(32);
  for (uint32_t i = 0; i < 32; ++i) {
    futures.push_back(std::async(std::launch::async, [&] {
      return searcher.search(query.data(), opts);
    }));
  }

  for (auto &future : futures) {
    const auto hits = future.get();
    ASSERT_EQ(hits.size(), baseline.size());
    for (size_t i = 0; i < hits.size(); ++i) {
      EXPECT_EQ(hits[i].label, baseline[i].label);
      EXPECT_FLOAT_EQ(hits[i].distance, baseline[i].distance);
    }
  }
}

TEST_F(VamanaSegmentSearcherTest, external_label_mapping) {
  constexpr uint32_t kDim = 8;
  const auto seg_dir = build_segment(256, kDim, 3, 1000);
  for (uint64_t i = 0; i < labels_.size(); ++i) {
    labels_[i] = 1000 + 1000 * i;
  }
  {
    VamanaSegmentBuilder builder(kDim, MetricType::L2, params());
    const auto seg2 = seg_parent_ / "seg_00000002";
    builder.add_batch(vectors_.data(), labels_.data(), labels_.size());
    builder.finish(seg2);
    VamanaSegmentSearcher searcher(seg2);
    DiskSearchOptions opts;
    opts.top_k = 5;
    opts.ef = 64;
    auto hits = searcher.search(vectors_.data(), opts);
    std::unordered_set<uint64_t> label_set(labels_.begin(), labels_.end());
    for (const auto &hit : hits) {
      EXPECT_TRUE(label_set.contains(hit.label));
      EXPECT_GE(hit.label, 1000u);
    }
  }
  (void)seg_dir;
}

TEST_F(VamanaSegmentSearcherTest, reopen_results_stable) {
  constexpr uint32_t kDim = 16;
  const auto seg_dir = build_segment(512, kDim, 4);
  DiskSearchOptions opts;
  opts.top_k = 10;
  opts.ef = 64;
  const auto query = std::vector<float>(vectors_.begin() + 17 * kDim,
                                        vectors_.begin() + 18 * kDim);
  VamanaSegmentSearcher first(seg_dir);
  const auto hits1 = first.search(query.data(), opts);
  VamanaSegmentSearcher second(seg_dir);
  const auto hits2 = second.search(query.data(), opts);
  ASSERT_EQ(hits1.size(), hits2.size());
  for (size_t i = 0; i < hits1.size(); ++i) {
    EXPECT_EQ(hits1[i].label, hits2[i].label);
    EXPECT_FLOAT_EQ(hits1[i].distance, hits2[i].distance);
  }
}

TEST_F(VamanaSegmentSearcherTest, recall_against_brute_force_l2) {
  constexpr uint32_t kDim = 16;
  constexpr uint32_t kN = 2048;
  const auto seg_dir = build_segment(kN, kDim, 5);
  VamanaSegmentSearcher searcher(seg_dir);
  DiskSearchOptions opts;
  opts.top_k = 10;
  opts.ef = 64;

  std::mt19937 rng(43);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  double recall_sum = 0.0;
  constexpr uint32_t kQueries = 50;
  for (uint32_t q = 0; q < kQueries; ++q) {
    std::vector<float> query(kDim);
    for (auto &v : query) {
      v = dist(rng);
    }

    std::vector<std::pair<float, uint64_t>> truth;
    truth.reserve(kN);
    for (uint32_t i = 0; i < kN; ++i) {
      const float d = alaya::simd::l2_sqr<float, float>(
          query.data(), vectors_.data() + static_cast<size_t>(i) * kDim, kDim);
      truth.emplace_back(d, labels_[i]);
    }
    std::partial_sort(truth.begin(), truth.begin() + opts.top_k, truth.end());
    std::unordered_set<uint64_t> top_truth;
    for (uint32_t i = 0; i < opts.top_k; ++i) {
      top_truth.insert(truth[i].second);
    }

    const auto hits = searcher.search(query.data(), opts);
    ASSERT_EQ(hits.size(), opts.top_k);
    uint32_t matched = 0;
    for (const auto &hit : hits) {
      matched += top_truth.contains(hit.label) ? 1U : 0U;
    }
    recall_sum += static_cast<double>(matched) / opts.top_k;
  }
  EXPECT_GE(recall_sum / kQueries, 0.7);
}

TEST_F(VamanaSegmentSearcherTest, invalid_graph_file_throws) {
  const auto seg_dir = build_segment(128, 8, 6);
  std::filesystem::resize_file(seg_dir / "graph.index", 12);
  expect_runtime_message_contains([&] { VamanaSegmentSearcher searcher(seg_dir); },
                                  {"VamanaReader"});
}

TEST_F(VamanaSegmentSearcherTest, ids_file_size_mismatch_throws) {
  const auto seg_dir = build_segment(128, 8, 7);
  const auto ids_path = seg_dir / "ids.u64.bin";
  std::filesystem::resize_file(ids_path, std::filesystem::file_size(ids_path) - 8);
  expect_runtime_message_contains([&] { VamanaSegmentSearcher searcher(seg_dir); },
                                  {seg_dir.string(), "ids file size mismatch"});
}

TEST_F(VamanaSegmentSearcherTest, vectors_file_size_mismatch_throws) {
  const auto seg_dir = build_segment(128, 8, 8);
  const auto vec_path = seg_dir / "vectors.f32.bin";
  std::filesystem::resize_file(vec_path, std::filesystem::file_size(vec_path) - 4);
  expect_runtime_message_contains([&] { VamanaSegmentSearcher searcher(seg_dir); },
                                  {seg_dir.string(), "vectors file size mismatch"});
}

TEST_F(VamanaSegmentSearcherTest, x_graph_file_missing_throws) {
  const auto seg_dir = build_segment(128, 8, 9);
  auto manifest = SegmentManifest::load(seg_dir / "manifest.txt");
  manifest.x_extras.erase("x_graph_file");
  manifest.save(seg_dir / "manifest.txt");
  expect_runtime_message_contains([&] { VamanaSegmentSearcher searcher(seg_dir); },
                                  {"x_graph_file missing", seg_dir.string()});
}

TEST_F(VamanaSegmentSearcherTest, manifest_count_disagrees_with_graph_throws) {
  constexpr uint32_t kDim = 8;
  const auto seg_dir = build_segment(128, kDim, 10);
  auto manifest = SegmentManifest::load(seg_dir / "manifest.txt");
  manifest.count = 127;
  manifest.save(seg_dir / "manifest.txt");
  std::filesystem::resize_file(seg_dir / "ids.u64.bin", 127 * sizeof(uint64_t));
  std::filesystem::resize_file(seg_dir / "vectors.f32.bin", 127 * kDim * sizeof(float));
  expect_runtime_message_contains([&] { VamanaSegmentSearcher searcher(seg_dir); },
                                  {"127", "128", seg_dir.string()});
}

TEST_F(VamanaSegmentSearcherTest, cos_metric_rejected) {
  const auto seg_dir = build_segment(128, 8, 11);
  auto manifest = SegmentManifest::load(seg_dir / "manifest.txt");
  manifest.metric = MetricType::COS;
  manifest.save(seg_dir / "manifest.txt");
  expect_runtime_message_contains([&] { VamanaSegmentSearcher searcher(seg_dir); },
                                  {"cos", "not implemented in v1"});
}

}  // namespace
}  // namespace alaya::disk
