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

#include "index/disk/disk_collection.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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

constexpr uint64_t kSentinelLabel = std::numeric_limits<uint64_t>::max();

// Bit-pattern NaN check: the project compiles with -Ofast / -ffast-math, so
// std::isnan is folded to a constant (per `project_ofast_finiteness_check`).
auto is_nan_f32(float v) -> bool {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(v));
  std::memcpy(&bits, &v, sizeof(v));
  return (bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0;
}

// Bit-exact float equality: `a == b` would return false on NaN-vs-NaN, and
// `-Ofast` may also re-associate; reading the bits is the only reliable way
// to assert "C++ wrote the same float bytes as the baseline".
auto bits_equal_f32(float a, float b) -> bool {
  uint32_t bits_a = 0;
  uint32_t bits_b = 0;
  std::memcpy(&bits_a, &a, sizeof(a));
  std::memcpy(&bits_b, &b, sizeof(b));
  return bits_a == bits_b;
}

auto allocate_label_buffer(uint64_t n_queries, uint32_t top_k) -> std::vector<uint64_t> {
  return std::vector<uint64_t>(n_queries * top_k, kSentinelLabel);
}

auto allocate_distance_buffer(uint64_t n_queries, uint32_t top_k) -> std::vector<float> {
  return std::vector<float>(n_queries * top_k, std::numeric_limits<float>::quiet_NaN());
}

class BatchSearchTestBase : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    tmp_root_ = std::filesystem::temp_directory_path() /
                ("alaya_disk_batch_search_" + std::to_string(::getpid()) + "_" +
                 info->test_suite_name() + "_" + info->name());
    std::filesystem::remove_all(tmp_root_);
    std::filesystem::create_directories(tmp_root_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_root_, ec);
  }

  static auto make_vectors(uint64_t n, uint32_t dim, uint32_t seed) -> std::vector<float> {
    std::vector<float> out(n * dim);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto &v : out) {
      v = dist(rng);
    }
    return out;
  }

  static auto sequential_labels(uint64_t n, uint64_t base) -> std::vector<uint64_t> {
    std::vector<uint64_t> out(n);
    std::iota(out.begin(), out.end(), base);
    return out;
  }

  // Computes the per-query serial baseline by calling search() once per query
  // and writing the results into a `(n_queries, top_k)` row-major buffer
  // pre-filled with the spec sentinels. This is the exact comparison the
  // multi-thread batch_search must reproduce element-by-element.
  static void serial_baseline(const DiskCollection &col,
                              const float *queries,
                              uint64_t n_queries,
                              const DiskSearchOptions &opts,
                              uint32_t dim,
                              std::vector<uint64_t> &labels_out,
                              std::vector<float> &distances_out) {
    labels_out.assign(n_queries * opts.top_k, kSentinelLabel);
    distances_out.assign(n_queries * opts.top_k, std::numeric_limits<float>::quiet_NaN());
    for (uint64_t i = 0; i < n_queries; ++i) {
      const auto hits = col.search(queries + i * dim, opts);
      for (size_t j = 0; j < hits.size(); ++j) {
        labels_out[i * opts.top_k + j] = hits[j].label;
        distances_out[i * opts.top_k + j] = hits[j].distance;
      }
    }
  }

  std::filesystem::path tmp_root_;
};

// ---------------------------------------------------------------------------
// disk_flat fixture
// ---------------------------------------------------------------------------

class BatchFlatTest : public BatchSearchTestBase {
 protected:
  static constexpr uint32_t kDim = 16;
  static constexpr uint32_t kTopK = 10;
  static constexpr uint32_t kEf = 100;
  static constexpr uint32_t kBeamWidth = 4;

  static auto default_opts() -> DiskSearchOptions {
    DiskSearchOptions opts;
    opts.top_k = kTopK;
    opts.ef = kEf;
    opts.beam_width = kBeamWidth;
    return opts;
  }

  // Build a 2-segment Flat collection with `kPerSegment` random vectors per
  // segment, returning the open-able path. The collection is closed before
  // the path is returned so callers can DiskCollection::open() it without
  // colliding on the writer lock.
  auto build_two_segment_collection(uint64_t per_segment = 500) -> std::filesystem::path {
    const auto path = tmp_root_ / "coll";
    DiskCollection col(path, kDim, MetricType::L2, DiskIndexType::Flat);
    auto v1 = make_vectors(per_segment, kDim, 1);
    auto l1 = sequential_labels(per_segment, 0);
    col.add_batch(v1.data(), l1.data(), per_segment);
    col.flush();
    auto v2 = make_vectors(per_segment, kDim, 2);
    auto l2 = sequential_labels(per_segment, per_segment);
    col.add_batch(v2.data(), l2.data(), per_segment);
    col.flush();
    return path;
  }
};

TEST_F(BatchFlatTest, SingleQueryEquivalence) {
  auto col = DiskCollection::open(build_two_segment_collection());
  const auto opts = default_opts();
  const auto query = make_vectors(1, kDim, 99);

  const auto baseline = col.search(query.data(), opts);
  ASSERT_GT(baseline.size(), 0u);

  auto out_labels = allocate_label_buffer(1, opts.top_k);
  auto out_distances = allocate_distance_buffer(1, opts.top_k);
  col.batch_search(query.data(), 1, opts, /*num_threads=*/1, out_labels.data(),
                   out_distances.data());

  for (size_t j = 0; j < baseline.size(); ++j) {
    EXPECT_EQ(out_labels[j], baseline[j].label);
    EXPECT_TRUE(bits_equal_f32(out_distances[j], baseline[j].distance));
  }
  for (size_t j = baseline.size(); j < opts.top_k; ++j) {
    EXPECT_EQ(out_labels[j], kSentinelLabel);
    EXPECT_TRUE(is_nan_f32(out_distances[j]));
  }
}

TEST_F(BatchFlatTest, PaddingSentinels) {
  // Build a collection with exactly 3 vectors; request top_k = 10 so trailing
  // [3, 10) slots remain at the caller-pre-filled sentinels.
  const auto path = tmp_root_ / "coll";
  DiskCollection col(path, kDim, MetricType::L2, DiskIndexType::Flat);
  const auto vectors = make_vectors(3, kDim, 11);
  const auto ids = sequential_labels(3, 100);
  col.add_batch(vectors.data(), ids.data(), 3);
  col.flush();

  auto opts = default_opts();
  opts.ef = 32;

  const auto query = make_vectors(1, kDim, 7);
  auto out_labels = allocate_label_buffer(1, opts.top_k);
  auto out_distances = allocate_distance_buffer(1, opts.top_k);
  col.batch_search(query.data(), 1, opts, /*num_threads=*/1, out_labels.data(),
                   out_distances.data());

  std::vector<uint64_t> seen{out_labels[0], out_labels[1], out_labels[2]};
  std::sort(seen.begin(), seen.end());
  EXPECT_EQ(seen[0], 100u);
  EXPECT_EQ(seen[1], 101u);
  EXPECT_EQ(seen[2], 102u);
  for (size_t j = 0; j < 3; ++j) {
    EXPECT_FALSE(is_nan_f32(out_distances[j]));
  }
  for (size_t j = 3; j < opts.top_k; ++j) {
    EXPECT_EQ(out_labels[j], kSentinelLabel);
    EXPECT_TRUE(is_nan_f32(out_distances[j]));
  }
}

TEST_F(BatchFlatTest, MultiQueryAgreesWithSerial) {
  auto col = DiskCollection::open(build_two_segment_collection());
  const auto opts = default_opts();

  constexpr uint64_t kN = 64;
  const auto queries = make_vectors(kN, kDim, 100);

  std::vector<uint64_t> baseline_labels;
  std::vector<float> baseline_distances;
  serial_baseline(col, queries.data(), kN, opts, kDim, baseline_labels, baseline_distances);

  auto out_labels = allocate_label_buffer(kN, opts.top_k);
  auto out_distances = allocate_distance_buffer(kN, opts.top_k);
  col.batch_search(queries.data(), kN, opts, /*num_threads=*/8, out_labels.data(),
                   out_distances.data());

  EXPECT_EQ(out_labels, baseline_labels);
  ASSERT_EQ(out_distances.size(), baseline_distances.size());
  for (size_t i = 0; i < out_distances.size(); ++i) {
    if (is_nan_f32(baseline_distances[i])) {
      EXPECT_TRUE(is_nan_f32(out_distances[i])) << "row " << i;
    } else {
      EXPECT_TRUE(bits_equal_f32(out_distances[i], baseline_distances[i])) << "row " << i;
    }
  }
}

TEST_F(BatchFlatTest, EmptyQueriesNoop) {
  auto col = DiskCollection::open(build_two_segment_collection());
  const auto opts = default_opts();
  EXPECT_NO_THROW(col.batch_search(nullptr, 0, opts, /*num_threads=*/4, nullptr, nullptr));
}

TEST_F(BatchFlatTest, EmptyCollectionAllSentinels) {
  // Fresh collection with zero segments: spec contract 7 keeps every output
  // slot at the caller-pre-filled sentinel.
  const auto path = tmp_root_ / "coll";
  DiskCollection col(path, kDim, MetricType::L2, DiskIndexType::Flat);

  const auto opts = default_opts();
  constexpr uint64_t kN = 4;
  const auto queries = make_vectors(kN, kDim, 1);
  auto out_labels = allocate_label_buffer(kN, opts.top_k);
  auto out_distances = allocate_distance_buffer(kN, opts.top_k);
  col.batch_search(queries.data(), kN, opts, /*num_threads=*/4, out_labels.data(),
                   out_distances.data());
  for (auto label : out_labels) {
    EXPECT_EQ(label, kSentinelLabel);
  }
  for (auto dist : out_distances) {
    EXPECT_TRUE(is_nan_f32(dist));
  }
}

TEST_F(BatchFlatTest, NumThreadsZeroThrows) {
  auto col = DiskCollection::open(build_two_segment_collection());
  const auto opts = default_opts();
  const auto queries = make_vectors(4, kDim, 1);
  auto out_labels = allocate_label_buffer(4, opts.top_k);
  auto out_distances = allocate_distance_buffer(4, opts.top_k);

  try {
    col.batch_search(queries.data(), 4, opts, /*num_threads=*/0, out_labels.data(),
                     out_distances.data());
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument &e) {
    EXPECT_NE(std::string(e.what()).find("num_threads"), std::string::npos);
  }
}

TEST_F(BatchFlatTest, TopKZeroThrows) {
  auto col = DiskCollection::open(build_two_segment_collection());
  DiskSearchOptions opts;
  opts.top_k = 0;
  opts.ef = kEf;
  opts.beam_width = kBeamWidth;
  const auto queries = make_vectors(4, kDim, 1);
  std::vector<uint64_t> out_labels(1, kSentinelLabel);
  std::vector<float> out_distances(1, std::numeric_limits<float>::quiet_NaN());
  EXPECT_THROW(col.batch_search(queries.data(), 4, opts, /*num_threads=*/1, out_labels.data(),
                                out_distances.data()),
               std::invalid_argument);
}

TEST_F(BatchFlatTest, OutDistancesNullFastPath) {
  auto col = DiskCollection::open(build_two_segment_collection());
  const auto opts = default_opts();
  constexpr uint64_t kN = 4;
  const auto queries = make_vectors(kN, kDim, 5);

  auto labels_with_dist = allocate_label_buffer(kN, opts.top_k);
  auto distances_with = allocate_distance_buffer(kN, opts.top_k);
  col.batch_search(queries.data(), kN, opts, /*num_threads=*/4, labels_with_dist.data(),
                   distances_with.data());

  auto labels_no_dist = allocate_label_buffer(kN, opts.top_k);
  col.batch_search(queries.data(), kN, opts, /*num_threads=*/4, labels_no_dist.data(),
                   /*out_distances=*/nullptr);

  EXPECT_EQ(labels_with_dist, labels_no_dist);
}

TEST_F(BatchFlatTest, ConcurrentBatchSearch_NoCorruption) {
  auto col = DiskCollection::open(build_two_segment_collection());
  const auto opts = default_opts();
  constexpr uint64_t kN = 256;
  const auto queries = make_vectors(kN, kDim, 200);

  std::vector<uint64_t> baseline_labels;
  std::vector<float> baseline_distances;
  serial_baseline(col, queries.data(), kN, opts, kDim, baseline_labels, baseline_distances);

  auto out_labels = allocate_label_buffer(kN, opts.top_k);
  col.batch_search(queries.data(), kN, opts, /*num_threads=*/8, out_labels.data(),
                   /*out_distances=*/nullptr);
  EXPECT_EQ(out_labels, baseline_labels);
}

// ---------------------------------------------------------------------------
// disk_vamana fixture
// ---------------------------------------------------------------------------

class BatchVamanaTest : public BatchSearchTestBase {
 protected:
  static constexpr uint32_t kDim = 16;
  static constexpr uint32_t kTopK = 10;
  static constexpr uint32_t kEf = 100;
  static constexpr uint32_t kBeamWidth = 4;

  static auto default_opts() -> DiskSearchOptions {
    DiskSearchOptions opts;
    opts.top_k = kTopK;
    opts.ef = kEf;
    opts.beam_width = kBeamWidth;
    return opts;
  }

  static auto vamana_params() -> VamanaSegmentBuildParams {
    VamanaSegmentBuildParams params;
    params.R = 32;
    params.L = 64;
    params.alpha = 1.2F;
    params.seed = 1234;
    params.num_threads = 1;
    return params;
  }

  auto build_two_segment_collection(uint64_t per_segment = 500) -> std::filesystem::path {
    const auto path = tmp_root_ / "coll";
    DiskCollection col(path,
                       kDim,
                       MetricType::L2,
                       DiskIndexType::Vamana,
                       DiskCollection::kDefaultMaxPendingBytes,
                       vamana_params());
    auto v1 = make_vectors(per_segment, kDim, 1);
    auto l1 = sequential_labels(per_segment, 0);
    col.add_batch(v1.data(), l1.data(), per_segment);
    col.flush();
    auto v2 = make_vectors(per_segment, kDim, 2);
    auto l2 = sequential_labels(per_segment, per_segment);
    col.add_batch(v2.data(), l2.data(), per_segment);
    col.flush();
    return path;
  }
};

TEST_F(BatchVamanaTest, SingleQueryEquivalence) {
  auto col = DiskCollection::open(build_two_segment_collection());
  const auto opts = default_opts();
  const auto query = make_vectors(1, kDim, 99);

  const auto baseline = col.search(query.data(), opts);
  ASSERT_GT(baseline.size(), 0u);

  auto out_labels = allocate_label_buffer(1, opts.top_k);
  auto out_distances = allocate_distance_buffer(1, opts.top_k);
  col.batch_search(query.data(), 1, opts, /*num_threads=*/1, out_labels.data(),
                   out_distances.data());

  for (size_t j = 0; j < baseline.size(); ++j) {
    EXPECT_EQ(out_labels[j], baseline[j].label);
    EXPECT_TRUE(bits_equal_f32(out_distances[j], baseline[j].distance));
  }
  for (size_t j = baseline.size(); j < opts.top_k; ++j) {
    EXPECT_EQ(out_labels[j], kSentinelLabel);
    EXPECT_TRUE(is_nan_f32(out_distances[j]));
  }
}

TEST_F(BatchVamanaTest, PaddingSentinels) {
  // Vamana cannot flush a 3-row segment, so build a normal-sized collection
  // and request a top_k larger than the row count to drive the trailing
  // sentinel path.
  auto col = DiskCollection::open(build_two_segment_collection(/*per_segment=*/8));
  auto opts = default_opts();
  opts.top_k = 32;  // > total rows (16) so trailing slots stay sentinel
  opts.ef = 64;

  const auto query = make_vectors(1, kDim, 23);
  auto out_labels = allocate_label_buffer(1, opts.top_k);
  auto out_distances = allocate_distance_buffer(1, opts.top_k);
  col.batch_search(query.data(), 1, opts, /*num_threads=*/1, out_labels.data(),
                   out_distances.data());

  // The first `col.size()` slots have real labels; the rest are sentinels.
  const auto total_rows = static_cast<size_t>(col.size());
  ASSERT_LE(total_rows, opts.top_k);
  for (size_t j = 0; j < total_rows; ++j) {
    EXPECT_NE(out_labels[j], kSentinelLabel);
    EXPECT_FALSE(is_nan_f32(out_distances[j]));
  }
  for (size_t j = total_rows; j < opts.top_k; ++j) {
    EXPECT_EQ(out_labels[j], kSentinelLabel);
    EXPECT_TRUE(is_nan_f32(out_distances[j]));
  }
}

TEST_F(BatchVamanaTest, MultiQueryAgreesWithSerial) {
  auto col = DiskCollection::open(build_two_segment_collection());
  const auto opts = default_opts();
  constexpr uint64_t kN = 64;
  const auto queries = make_vectors(kN, kDim, 100);

  std::vector<uint64_t> baseline_labels;
  std::vector<float> baseline_distances;
  serial_baseline(col, queries.data(), kN, opts, kDim, baseline_labels, baseline_distances);

  auto out_labels = allocate_label_buffer(kN, opts.top_k);
  auto out_distances = allocate_distance_buffer(kN, opts.top_k);
  col.batch_search(queries.data(), kN, opts, /*num_threads=*/8, out_labels.data(),
                   out_distances.data());

  EXPECT_EQ(out_labels, baseline_labels);
  ASSERT_EQ(out_distances.size(), baseline_distances.size());
  for (size_t i = 0; i < out_distances.size(); ++i) {
    if (is_nan_f32(baseline_distances[i])) {
      EXPECT_TRUE(is_nan_f32(out_distances[i])) << "row " << i;
    } else {
      EXPECT_TRUE(bits_equal_f32(out_distances[i], baseline_distances[i])) << "row " << i;
    }
  }
}

TEST_F(BatchVamanaTest, EmptyQueriesNoop) {
  auto col = DiskCollection::open(build_two_segment_collection());
  const auto opts = default_opts();
  EXPECT_NO_THROW(col.batch_search(nullptr, 0, opts, /*num_threads=*/4, nullptr, nullptr));
}

TEST_F(BatchVamanaTest, EmptyCollectionAllSentinels) {
  const auto path = tmp_root_ / "coll";
  DiskCollection col(path,
                     kDim,
                     MetricType::L2,
                     DiskIndexType::Vamana,
                     DiskCollection::kDefaultMaxPendingBytes,
                     vamana_params());
  const auto opts = default_opts();
  constexpr uint64_t kN = 4;
  const auto queries = make_vectors(kN, kDim, 1);
  auto out_labels = allocate_label_buffer(kN, opts.top_k);
  auto out_distances = allocate_distance_buffer(kN, opts.top_k);
  col.batch_search(queries.data(), kN, opts, /*num_threads=*/4, out_labels.data(),
                   out_distances.data());
  for (auto label : out_labels) {
    EXPECT_EQ(label, kSentinelLabel);
  }
  for (auto dist : out_distances) {
    EXPECT_TRUE(is_nan_f32(dist));
  }
}

TEST_F(BatchVamanaTest, NumThreadsZeroThrows) {
  auto col = DiskCollection::open(build_two_segment_collection());
  const auto opts = default_opts();
  const auto queries = make_vectors(4, kDim, 1);
  auto out_labels = allocate_label_buffer(4, opts.top_k);
  auto out_distances = allocate_distance_buffer(4, opts.top_k);
  EXPECT_THROW(col.batch_search(queries.data(), 4, opts, /*num_threads=*/0, out_labels.data(),
                                out_distances.data()),
               std::invalid_argument);
}

TEST_F(BatchVamanaTest, TopKZeroThrows) {
  auto col = DiskCollection::open(build_two_segment_collection());
  DiskSearchOptions opts;
  opts.top_k = 0;
  opts.ef = kEf;
  opts.beam_width = kBeamWidth;
  const auto queries = make_vectors(4, kDim, 1);
  std::vector<uint64_t> out_labels(1, kSentinelLabel);
  std::vector<float> out_distances(1, std::numeric_limits<float>::quiet_NaN());
  EXPECT_THROW(col.batch_search(queries.data(), 4, opts, /*num_threads=*/1, out_labels.data(),
                                out_distances.data()),
               std::invalid_argument);
}

TEST_F(BatchVamanaTest, OutDistancesNullFastPath) {
  auto col = DiskCollection::open(build_two_segment_collection());
  const auto opts = default_opts();
  constexpr uint64_t kN = 4;
  const auto queries = make_vectors(kN, kDim, 5);
  auto labels_with_dist = allocate_label_buffer(kN, opts.top_k);
  auto distances_with = allocate_distance_buffer(kN, opts.top_k);
  col.batch_search(queries.data(), kN, opts, /*num_threads=*/4, labels_with_dist.data(),
                   distances_with.data());
  auto labels_no_dist = allocate_label_buffer(kN, opts.top_k);
  col.batch_search(queries.data(), kN, opts, /*num_threads=*/4, labels_no_dist.data(),
                   /*out_distances=*/nullptr);
  EXPECT_EQ(labels_with_dist, labels_no_dist);
}

TEST_F(BatchVamanaTest, ConcurrentBatchSearch_NoCorruption) {
  auto col = DiskCollection::open(build_two_segment_collection());
  const auto opts = default_opts();
  constexpr uint64_t kN = 256;
  const auto queries = make_vectors(kN, kDim, 200);

  std::vector<uint64_t> baseline_labels;
  std::vector<float> baseline_distances;
  serial_baseline(col, queries.data(), kN, opts, kDim, baseline_labels, baseline_distances);

  auto out_labels = allocate_label_buffer(kN, opts.top_k);
  col.batch_search(queries.data(), kN, opts, /*num_threads=*/8, out_labels.data(),
                   /*out_distances=*/nullptr);
  EXPECT_EQ(out_labels, baseline_labels);
}

// ---------------------------------------------------------------------------
// disk_laser fixture (gated on ALAYA_ENABLE_LASER)
// ---------------------------------------------------------------------------

#if ALAYA_ENABLE_LASER

constexpr uint64_t kLaserFixtureCount = 2048;
constexpr uint32_t kLaserFixtureDim = 128;
constexpr uint32_t kLaserTopK = 10;
constexpr uint32_t kLaserEf = 100;
constexpr uint32_t kLaserBeamWidth = 4;

class BatchLaserTest : public BatchSearchTestBase {
 protected:
  static auto fixture_dir() -> std::filesystem::path {
    return std::filesystem::path(ALAYA_LASER_FIXTURE_DIR);
  }

  static auto fixture_prefix() -> std::string { return std::string(ALAYA_LASER_FIXTURE_PREFIX); }

  static auto fixture_has_required_files(const std::filesystem::path &dir,
                                         const std::string &prefix) -> bool {
    if (dir.empty()) {
      return false;
    }
    const auto index = dir / (prefix + "_R64_MD128.index");
    const std::vector<std::filesystem::path> required{
        dir / (prefix + "_input.fbin"),
        index,
        std::filesystem::path(index.string() + "_rotator"),
        std::filesystem::path(index.string() + "_cache_ids"),
        std::filesystem::path(index.string() + "_cache_nodes"),
    };
    std::error_code ec;
    return std::all_of(required.begin(), required.end(), [&](const auto &path) {
      const bool ok = std::filesystem::is_regular_file(path, ec) && !ec &&
                      std::filesystem::file_size(path, ec) > 0 && !ec;
      ec.clear();
      return ok;
    });
  }

  static void require_fixture_available() {
    if (!fixture_has_required_files(fixture_dir(), fixture_prefix())) {
      GTEST_SKIP() << "LASER fixture is missing or incomplete under " << fixture_dir();
    }
  }

  static auto read_fixture_vectors() -> std::vector<float> {
    const auto path = fixture_dir() / (fixture_prefix() + "_input.fbin");
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("failed to open LASER fixture vectors: " + path.string());
    }
    int32_t count = 0;
    int32_t dim = 0;
    input.read(reinterpret_cast<char *>(&count), sizeof(count));
    input.read(reinterpret_cast<char *>(&dim), sizeof(dim));
    if (count != static_cast<int32_t>(kLaserFixtureCount) ||
        dim != static_cast<int32_t>(kLaserFixtureDim)) {
      throw std::runtime_error("unexpected LASER fixture vector header in " + path.string());
    }
    std::vector<float> out(static_cast<size_t>(count) * static_cast<size_t>(dim));
    input.read(reinterpret_cast<char *>(out.data()),
               static_cast<std::streamsize>(out.size() * sizeof(float)));
    if (!input) {
      throw std::runtime_error("short LASER fixture vector read: " + path.string());
    }
    return out;
  }

  static auto fixture_query(const std::vector<float> &all_vectors, uint64_t row)
      -> std::vector<float> {
    if (row >= kLaserFixtureCount) {
      throw std::runtime_error("LASER fixture query row out of range");
    }
    std::vector<float> q(kLaserFixtureDim);
    std::copy_n(all_vectors.data() + static_cast<size_t>(row) * kLaserFixtureDim, kLaserFixtureDim,
                q.data());
    return q;
  }

  static auto default_opts() -> DiskSearchOptions {
    DiskSearchOptions opts;
    opts.top_k = kLaserTopK;
    opts.ef = kLaserEf;
    opts.beam_width = kLaserBeamWidth;
    return opts;
  }

  // Build a single-segment Laser collection populated from the deterministic
  // build-time fixture; returns an open collection (the import takes the
  // writer lock so callers don't reopen).
  auto build_one_segment_collection() -> DiskCollection {
    const auto path = tmp_root_ / "coll";
    DiskCollection col(path, kLaserFixtureDim, MetricType::L2, DiskIndexType::Laser);
    auto labels = sequential_labels(kLaserFixtureCount, 0);
    col.import_laser_segment(fixture_dir(), labels.data(), labels.size());
    return col;
  }
};

TEST_F(BatchLaserTest, SingleQueryEquivalence) {
  require_fixture_available();
  auto col = build_one_segment_collection();
  const auto opts = default_opts();
  const auto vectors = read_fixture_vectors();
  const auto query = fixture_query(vectors, 11);

  const auto baseline = col.search(query.data(), opts);
  ASSERT_GT(baseline.size(), 0u);

  auto out_labels = allocate_label_buffer(1, opts.top_k);
  auto out_distances = allocate_distance_buffer(1, opts.top_k);
  col.batch_search(query.data(), 1, opts, /*num_threads=*/1, out_labels.data(),
                   out_distances.data());

  for (size_t j = 0; j < baseline.size(); ++j) {
    EXPECT_EQ(out_labels[j], baseline[j].label);
    // Laser distance contract: every overwritten distance is NaN.
    EXPECT_TRUE(is_nan_f32(out_distances[j])) << "row " << j;
  }
  for (size_t j = baseline.size(); j < opts.top_k; ++j) {
    EXPECT_EQ(out_labels[j], kSentinelLabel);
    EXPECT_TRUE(is_nan_f32(out_distances[j]));
  }
}

TEST_F(BatchLaserTest, PaddingSentinels) {
  // The fixture has 2048 rows; a top_k larger than the fixture count drives
  // the trailing sentinel slots without requiring a new (smaller) fixture.
  require_fixture_available();
  auto col = build_one_segment_collection();
  auto opts = default_opts();
  opts.top_k = static_cast<uint32_t>(kLaserFixtureCount) + 16;
  opts.ef = 32;

  const auto vectors = read_fixture_vectors();
  const auto query = fixture_query(vectors, 5);
  auto out_labels = allocate_label_buffer(1, opts.top_k);
  auto out_distances = allocate_distance_buffer(1, opts.top_k);
  col.batch_search(query.data(), 1, opts, /*num_threads=*/1, out_labels.data(),
                   out_distances.data());

  // The Laser searcher returns at most some number of unique hits; whatever
  // the count, every NON-sentinel distance must still be NaN, and every
  // trailing slot beyond the returned hit count must remain at the sentinel.
  const auto baseline = col.search(query.data(), opts);
  for (size_t j = 0; j < baseline.size(); ++j) {
    EXPECT_NE(out_labels[j], kSentinelLabel);
    EXPECT_TRUE(is_nan_f32(out_distances[j]));
  }
  for (size_t j = baseline.size(); j < opts.top_k; ++j) {
    EXPECT_EQ(out_labels[j], kSentinelLabel);
    EXPECT_TRUE(is_nan_f32(out_distances[j]));
  }
}

TEST_F(BatchLaserTest, MultiQueryAgreesWithSerial) {
  require_fixture_available();
  auto col = build_one_segment_collection();
  const auto opts = default_opts();
  const auto vectors = read_fixture_vectors();

  constexpr uint64_t kN = 32;
  std::vector<float> queries(kN * kLaserFixtureDim);
  for (uint64_t i = 0; i < kN; ++i) {
    const auto q = fixture_query(vectors, i * 13 + 1);
    std::copy(q.begin(), q.end(), queries.begin() + i * kLaserFixtureDim);
  }

  std::vector<uint64_t> baseline_labels;
  std::vector<float> baseline_distances;
  serial_baseline(col, queries.data(), kN, opts, kLaserFixtureDim, baseline_labels,
                  baseline_distances);

  auto out_labels = allocate_label_buffer(kN, opts.top_k);
  auto out_distances = allocate_distance_buffer(kN, opts.top_k);
  col.batch_search(queries.data(), kN, opts, /*num_threads=*/8, out_labels.data(),
                   out_distances.data());

  EXPECT_EQ(out_labels, baseline_labels);
  // Laser: every overwritten distance slot is NaN.
  for (size_t i = 0; i < out_distances.size(); ++i) {
    if (out_labels[i] == kSentinelLabel) {
      EXPECT_TRUE(is_nan_f32(out_distances[i])) << "padding row " << i;
    } else {
      EXPECT_TRUE(is_nan_f32(out_distances[i])) << "data row " << i;
    }
  }
}

TEST_F(BatchLaserTest, EmptyQueriesNoop) {
  require_fixture_available();
  auto col = build_one_segment_collection();
  const auto opts = default_opts();
  EXPECT_NO_THROW(col.batch_search(nullptr, 0, opts, /*num_threads=*/4, nullptr, nullptr));
}

TEST_F(BatchLaserTest, EmptyCollectionAllSentinels) {
  // Construct a Laser collection but skip import: segments_.empty() is true.
  const auto path = tmp_root_ / "coll";
  DiskCollection col(path, kLaserFixtureDim, MetricType::L2, DiskIndexType::Laser);
  const auto opts = default_opts();
  constexpr uint64_t kN = 4;
  const auto queries = make_vectors(kN, kLaserFixtureDim, 1);
  auto out_labels = allocate_label_buffer(kN, opts.top_k);
  auto out_distances = allocate_distance_buffer(kN, opts.top_k);
  col.batch_search(queries.data(), kN, opts, /*num_threads=*/4, out_labels.data(),
                   out_distances.data());
  for (auto label : out_labels) {
    EXPECT_EQ(label, kSentinelLabel);
  }
  for (auto dist : out_distances) {
    EXPECT_TRUE(is_nan_f32(dist));
  }
}

TEST_F(BatchLaserTest, NumThreadsZeroThrows) {
  require_fixture_available();
  auto col = build_one_segment_collection();
  const auto opts = default_opts();
  const auto vectors = read_fixture_vectors();
  const auto query = fixture_query(vectors, 0);
  auto out_labels = allocate_label_buffer(1, opts.top_k);
  auto out_distances = allocate_distance_buffer(1, opts.top_k);
  EXPECT_THROW(col.batch_search(query.data(), 1, opts, /*num_threads=*/0, out_labels.data(),
                                out_distances.data()),
               std::invalid_argument);
}

TEST_F(BatchLaserTest, TopKZeroThrows) {
  require_fixture_available();
  auto col = build_one_segment_collection();
  DiskSearchOptions opts;
  opts.top_k = 0;
  opts.ef = kLaserEf;
  opts.beam_width = kLaserBeamWidth;
  const auto vectors = read_fixture_vectors();
  const auto query = fixture_query(vectors, 0);
  std::vector<uint64_t> out_labels(1, kSentinelLabel);
  std::vector<float> out_distances(1, std::numeric_limits<float>::quiet_NaN());
  EXPECT_THROW(col.batch_search(query.data(), 1, opts, /*num_threads=*/1, out_labels.data(),
                                out_distances.data()),
               std::invalid_argument);
}

TEST_F(BatchLaserTest, OutDistancesNullFastPath) {
  require_fixture_available();
  auto col = build_one_segment_collection();
  const auto opts = default_opts();
  const auto vectors = read_fixture_vectors();
  constexpr uint64_t kN = 4;
  std::vector<float> queries(kN * kLaserFixtureDim);
  for (uint64_t i = 0; i < kN; ++i) {
    const auto q = fixture_query(vectors, i * 7);
    std::copy(q.begin(), q.end(), queries.begin() + i * kLaserFixtureDim);
  }

  auto labels_with_dist = allocate_label_buffer(kN, opts.top_k);
  auto distances_with = allocate_distance_buffer(kN, opts.top_k);
  col.batch_search(queries.data(), kN, opts, /*num_threads=*/2, labels_with_dist.data(),
                   distances_with.data());

  auto labels_no_dist = allocate_label_buffer(kN, opts.top_k);
  col.batch_search(queries.data(), kN, opts, /*num_threads=*/2, labels_no_dist.data(),
                   /*out_distances=*/nullptr);
  EXPECT_EQ(labels_with_dist, labels_no_dist);
}

TEST_F(BatchLaserTest, ConcurrentBatchSearch_NoCorruption) {
  require_fixture_available();
  auto col = build_one_segment_collection();
  const auto opts = default_opts();
  const auto vectors = read_fixture_vectors();

  constexpr uint64_t kN = 256;
  std::vector<float> queries(kN * kLaserFixtureDim);
  for (uint64_t i = 0; i < kN; ++i) {
    const auto q = fixture_query(vectors, i % kLaserFixtureCount);
    std::copy(q.begin(), q.end(), queries.begin() + i * kLaserFixtureDim);
  }

  std::vector<uint64_t> baseline_labels;
  std::vector<float> baseline_distances;
  serial_baseline(col, queries.data(), kN, opts, kLaserFixtureDim, baseline_labels,
                  baseline_distances);

  auto out_labels = allocate_label_buffer(kN, opts.top_k);
  col.batch_search(queries.data(), kN, opts, /*num_threads=*/8, out_labels.data(),
                   /*out_distances=*/nullptr);
  EXPECT_EQ(out_labels, baseline_labels);
}

#endif  // ALAYA_ENABLE_LASER

}  // namespace
}  // namespace alaya::disk
