// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
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
#include "index/disk/disk_flat_builder.hpp"
#include "index/disk/disk_flat_searcher.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "utils/metric_type.hpp"

namespace alaya::disk {

namespace {

class DiskFlatSearcherTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto pid_str = std::to_string(static_cast<long long>(::getpid()));
    auto base = std::filesystem::temp_directory_path() /
                ("alaya_flat_searcher_" + pid_str + "_" +
                 ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::remove_all(base);
    std::filesystem::create_directories(base);
    tmp_root_ = base;
    seg_parent_ = tmp_root_ / "segments";
    std::filesystem::create_directories(seg_parent_);
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

  static auto sequential_labels(uint64_t n, uint64_t base = 1000) -> std::vector<uint64_t> {
    std::vector<uint64_t> out(n);
    std::iota(out.begin(), out.end(), base);
    return out;
  }

  static auto fnv1a64(const void *data, size_t bytes) -> uint64_t {
    const auto *p = static_cast<const uint8_t *>(data);
    uint64_t h = 0xCBF29CE484222325ULL;
    for (size_t i = 0; i < bytes; ++i) {
      h ^= p[i];
      h *= 0x100000001B3ULL;
    }
    return h;
  }

  // Builds a segment via DiskFlatBuilder and returns its directory.
  auto build_segment(MetricType metric, const std::vector<float> &vectors,
                     const std::vector<uint64_t> &labels, uint32_t dim,
                     const std::string &seg_name) const -> std::filesystem::path {
    auto seg_dir = seg_parent_ / seg_name;
    DiskFlatBuilder b(dim, metric);
    b.add_batch(vectors.data(), labels.data(), labels.size());
    b.finish(seg_dir);
    return seg_dir;
  }

  std::filesystem::path tmp_root_;
  std::filesystem::path seg_parent_;
};

// Brute-force reference for L2 squared distance (smaller = closer).
static auto bf_l2_topk(const std::vector<float> &vectors, const std::vector<uint64_t> &labels,
                       const std::vector<float> &query, uint32_t dim, uint64_t k)
    -> std::vector<DiskSearchHit> {
  const uint64_t n = labels.size();
  std::vector<DiskSearchHit> all(n);
  for (uint64_t i = 0; i < n; ++i) {
    float s = 0.0F;
    for (uint32_t c = 0; c < dim; ++c) {
      const float diff = query[c] - vectors[i * dim + c];
      s += diff * diff;
    }
    all[i] = DiskSearchHit{labels[i], s};
  }
  std::sort(all.begin(), all.end(), [](const DiskSearchHit &a, const DiskSearchHit &b) {
    if (a.distance != b.distance) return a.distance < b.distance;
    return a.label < b.label;
  });
  all.resize(std::min<uint64_t>(k, all.size()));
  return all;
}

// Brute-force reference for IP (negative inner product; smaller = closer).
static auto bf_ip_topk(const std::vector<float> &vectors, const std::vector<uint64_t> &labels,
                       const std::vector<float> &query, uint32_t dim, uint64_t k)
    -> std::vector<DiskSearchHit> {
  const uint64_t n = labels.size();
  std::vector<DiskSearchHit> all(n);
  for (uint64_t i = 0; i < n; ++i) {
    float s = 0.0F;
    for (uint32_t c = 0; c < dim; ++c) {
      s += query[c] * vectors[i * dim + c];
    }
    all[i] = DiskSearchHit{labels[i], -s};
  }
  std::sort(all.begin(), all.end(), [](const DiskSearchHit &a, const DiskSearchHit &b) {
    if (a.distance != b.distance) return a.distance < b.distance;
    return a.label < b.label;
  });
  all.resize(std::min<uint64_t>(k, all.size()));
  return all;
}

static auto bf_cos_topk(const std::vector<float> &vectors, const std::vector<uint64_t> &labels,
                        const std::vector<float> &query, uint32_t dim, uint64_t k)
    -> std::vector<DiskSearchHit> {
  // Normalize vectors and query.
  auto normalize = [&](std::vector<float> &v, uint64_t rows) {
    for (uint64_t i = 0; i < rows; ++i) {
      double s = 0.0;
      for (uint32_t c = 0; c < dim; ++c) {
        const double x = v[i * dim + c];
        s += x * x;
      }
      const double inv = 1.0 / std::sqrt(s);
      for (uint32_t c = 0; c < dim; ++c) {
        v[i * dim + c] = static_cast<float>(static_cast<double>(v[i * dim + c]) * inv);
      }
    }
  };
  std::vector<float> norm_vecs = vectors;
  std::vector<float> norm_q = query;
  normalize(norm_vecs, labels.size());
  normalize(norm_q, 1);
  return bf_ip_topk(norm_vecs, labels, norm_q, dim, k);
}

TEST_F(DiskFlatSearcherTest, L2MatchesBruteforce) {
  constexpr uint32_t kDim = 32;
  constexpr uint64_t kN = 1000;
  auto vectors = make_random_vectors(kN, kDim, 1);
  auto labels = sequential_labels(kN);
  auto seg_dir = build_segment(MetricType::L2, vectors, labels, kDim, "seg_00000001");

  DiskFlatSegmentSearcher s(seg_dir);
  ASSERT_EQ(s.size(), kN);
  ASSERT_EQ(s.dim(), kDim);
  ASSERT_EQ(s.type(), DiskIndexType::Flat);

  auto query = make_random_vectors(1, kDim, 7);
  DiskSearchOptions opts;
  opts.top_k = 10;
  auto hits = s.search(query.data(), opts);
  auto expected = bf_l2_topk(vectors, labels, query, kDim, 10);
  ASSERT_EQ(hits.size(), expected.size());
  for (size_t i = 0; i < hits.size(); ++i) {
    EXPECT_EQ(hits[i].label, expected[i].label) << "rank " << i;
    EXPECT_NEAR(hits[i].distance, expected[i].distance, 1e-3F)
        << "rank " << i << " dist=" << hits[i].distance << " expected=" << expected[i].distance;
  }
}

TEST_F(DiskFlatSearcherTest, IpMatchesBruteforce) {
  constexpr uint32_t kDim = 32;
  constexpr uint64_t kN = 1000;
  auto vectors = make_random_vectors(kN, kDim, 2);
  auto labels = sequential_labels(kN);
  auto seg_dir = build_segment(MetricType::IP, vectors, labels, kDim, "seg_00000001");

  DiskFlatSegmentSearcher s(seg_dir);
  auto query = make_random_vectors(1, kDim, 8);
  DiskSearchOptions opts;
  opts.top_k = 10;
  auto hits = s.search(query.data(), opts);
  auto expected = bf_ip_topk(vectors, labels, query, kDim, 10);
  ASSERT_EQ(hits.size(), expected.size());
  for (size_t i = 0; i < hits.size(); ++i) {
    EXPECT_EQ(hits[i].label, expected[i].label) << "rank " << i;
    EXPECT_NEAR(hits[i].distance, expected[i].distance, 1e-3F);
  }
}

TEST_F(DiskFlatSearcherTest, CosMatchesBruteforce) {
  constexpr uint32_t kDim = 32;
  constexpr uint64_t kN = 1000;
  auto vectors = make_random_vectors(kN, kDim, 3);
  auto labels = sequential_labels(kN);
  auto seg_dir = build_segment(MetricType::COS, vectors, labels, kDim, "seg_00000001");

  DiskFlatSegmentSearcher s(seg_dir);
  auto query = make_random_vectors(1, kDim, 9);
  DiskSearchOptions opts;
  opts.top_k = 10;
  auto hits = s.search(query.data(), opts);
  auto expected = bf_cos_topk(vectors, labels, query, kDim, 10);
  ASSERT_EQ(hits.size(), expected.size());
  for (size_t i = 0; i < hits.size(); ++i) {
    EXPECT_EQ(hits[i].label, expected[i].label) << "rank " << i;
    EXPECT_NEAR(hits[i].distance, expected[i].distance, 1e-3F);
  }
}

TEST_F(DiskFlatSearcherTest, L2ExactMatch) {
  constexpr uint32_t kDim = 16;
  constexpr uint64_t kN = 50;
  auto vectors = make_random_vectors(kN, kDim, 11);
  auto labels = sequential_labels(kN);
  auto seg_dir = build_segment(MetricType::L2, vectors, labels, kDim, "seg_00000001");

  DiskFlatSegmentSearcher s(seg_dir);
  std::vector<float> query(vectors.begin() + 7 * kDim, vectors.begin() + 8 * kDim);
  DiskSearchOptions opts;
  opts.top_k = 1;
  auto hits = s.search(query.data(), opts);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].label, labels[7]);
  EXPECT_FLOAT_EQ(hits[0].distance, 0.0F);
}

TEST_F(DiskFlatSearcherTest, IpExactMatch) {
  constexpr uint32_t kDim = 16;
  constexpr uint64_t kN = 50;
  auto vectors = make_random_vectors(kN, kDim, 12);
  auto labels = sequential_labels(kN);
  auto seg_dir = build_segment(MetricType::IP, vectors, labels, kDim, "seg_00000001");

  DiskFlatSegmentSearcher s(seg_dir);
  std::vector<float> query(vectors.begin() + 13 * kDim, vectors.begin() + 14 * kDim);
  DiskSearchOptions opts;
  opts.top_k = 1;
  auto hits = s.search(query.data(), opts);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].label, labels[13]);
  float expected_neg_norm_sq = 0.0F;
  for (uint32_t c = 0; c < kDim; ++c) {
    expected_neg_norm_sq -= query[c] * query[c];
  }
  EXPECT_NEAR(hits[0].distance, expected_neg_norm_sq, 1e-3F);
}

TEST_F(DiskFlatSearcherTest, CallerBufferNotMutatedSearchL2) {
  constexpr uint32_t kDim = 8;
  auto vectors = make_random_vectors(20, kDim, 21);
  auto labels = sequential_labels(20);
  auto seg_dir = build_segment(MetricType::L2, vectors, labels, kDim, "seg_00000001");

  DiskFlatSegmentSearcher s(seg_dir);
  auto query = make_random_vectors(1, kDim, 99);
  const auto h0 = fnv1a64(query.data(), query.size() * sizeof(float));
  DiskSearchOptions opts;
  opts.top_k = 5;
  (void)s.search(query.data(), opts);
  const auto h1 = fnv1a64(query.data(), query.size() * sizeof(float));
  EXPECT_EQ(h0, h1);
}

TEST_F(DiskFlatSearcherTest, CallerBufferNotMutatedSearchIp) {
  constexpr uint32_t kDim = 8;
  auto vectors = make_random_vectors(20, kDim, 22);
  auto labels = sequential_labels(20);
  auto seg_dir = build_segment(MetricType::IP, vectors, labels, kDim, "seg_00000001");

  DiskFlatSegmentSearcher s(seg_dir);
  auto query = make_random_vectors(1, kDim, 100);
  const auto h0 = fnv1a64(query.data(), query.size() * sizeof(float));
  DiskSearchOptions opts;
  opts.top_k = 5;
  (void)s.search(query.data(), opts);
  EXPECT_EQ(fnv1a64(query.data(), query.size() * sizeof(float)), h0);
}

TEST_F(DiskFlatSearcherTest, CallerBufferNotMutatedSearchCos) {
  constexpr uint32_t kDim = 8;
  auto vectors = make_random_vectors(20, kDim, 23);
  auto labels = sequential_labels(20);
  auto seg_dir = build_segment(MetricType::COS, vectors, labels, kDim, "seg_00000001");

  DiskFlatSegmentSearcher s(seg_dir);
  auto query = make_random_vectors(1, kDim, 101);
  const auto h0 = fnv1a64(query.data(), query.size() * sizeof(float));
  DiskSearchOptions opts;
  opts.top_k = 5;
  (void)s.search(query.data(), opts);
  EXPECT_EQ(fnv1a64(query.data(), query.size() * sizeof(float)), h0);
}

TEST_F(DiskFlatSearcherTest, ZeroQueryThrowsSearchCos) {
  constexpr uint32_t kDim = 8;
  auto vectors = make_random_vectors(10, kDim, 31);
  auto labels = sequential_labels(10);
  auto seg_dir = build_segment(MetricType::COS, vectors, labels, kDim, "seg_00000001");

  DiskFlatSegmentSearcher s(seg_dir);
  std::vector<float> query(kDim, 0.0F);
  DiskSearchOptions opts;
  opts.top_k = 5;
  EXPECT_THROW(s.search(query.data(), opts), std::invalid_argument);
}

TEST_F(DiskFlatSearcherTest, ZeroQueryL2Succeeds) {
  constexpr uint32_t kDim = 8;
  auto vectors = make_random_vectors(10, kDim, 32);
  auto labels = sequential_labels(10);
  auto seg_dir = build_segment(MetricType::L2, vectors, labels, kDim, "seg_00000001");

  DiskFlatSegmentSearcher s(seg_dir);
  std::vector<float> query(kDim, 0.0F);
  DiskSearchOptions opts;
  opts.top_k = 3;
  auto hits = s.search(query.data(), opts);
  ASSERT_EQ(hits.size(), 3u);
  EXPECT_LE(hits[0].distance, hits[1].distance);
  EXPECT_LE(hits[1].distance, hits[2].distance);
}

class NonFiniteQueryParam : public DiskFlatSearcherTest,
                             public ::testing::WithParamInterface<MetricType> {};

TEST_P(NonFiniteQueryParam, NaNQueryThrows) {
  constexpr uint32_t kDim = 8;
  auto vectors = make_random_vectors(10, kDim, 41);
  auto labels = sequential_labels(10);
  auto seg_dir = build_segment(GetParam(), vectors, labels, kDim, "seg_00000001");

  DiskFlatSegmentSearcher s(seg_dir);
  auto query = make_random_vectors(1, kDim, 102);
  query[5] = std::numeric_limits<float>::quiet_NaN();
  DiskSearchOptions opts;
  opts.top_k = 3;
  try {
    (void)s.search(query.data(), opts);
    FAIL() << "expected throw on NaN query";
  } catch (const std::exception &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("5"), std::string::npos) << "msg should mention pos 5: " << msg;
    EXPECT_NE(msg.find("NaN"), std::string::npos) << "msg should mention NaN: " << msg;
  }
}

TEST_P(NonFiniteQueryParam, PosInfQueryThrows) {
  constexpr uint32_t kDim = 8;
  auto vectors = make_random_vectors(10, kDim, 42);
  auto labels = sequential_labels(10);
  auto seg_dir = build_segment(GetParam(), vectors, labels, kDim, "seg_00000001");

  DiskFlatSegmentSearcher s(seg_dir);
  auto query = make_random_vectors(1, kDim, 103);
  query[5] = std::numeric_limits<float>::infinity();
  DiskSearchOptions opts;
  opts.top_k = 3;
  EXPECT_THROW(s.search(query.data(), opts), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(AllMetrics, NonFiniteQueryParam,
                         ::testing::Values(MetricType::L2, MetricType::IP, MetricType::COS));

TEST_F(DiskFlatSearcherTest, TopKZeroThrows) {
  constexpr uint32_t kDim = 4;
  auto vectors = make_random_vectors(10, kDim, 51);
  auto labels = sequential_labels(10);
  auto seg_dir = build_segment(MetricType::L2, vectors, labels, kDim, "seg_00000001");

  DiskFlatSegmentSearcher s(seg_dir);
  auto query = make_random_vectors(1, kDim, 200);
  DiskSearchOptions opts;
  opts.top_k = 0;
  EXPECT_THROW(s.search(query.data(), opts), std::invalid_argument);
}

TEST_F(DiskFlatSearcherTest, TopKExceedsCountCaps) {
  constexpr uint32_t kDim = 4;
  constexpr uint64_t kN = 10;
  auto vectors = make_random_vectors(kN, kDim, 52);
  auto labels = sequential_labels(kN);
  auto seg_dir = build_segment(MetricType::L2, vectors, labels, kDim, "seg_00000001");

  DiskFlatSegmentSearcher s(seg_dir);
  auto query = make_random_vectors(1, kDim, 201);
  DiskSearchOptions opts;
  opts.top_k = 100;
  auto hits = s.search(query.data(), opts);
  EXPECT_EQ(hits.size(), kN);
}

TEST_F(DiskFlatSearcherTest, FileSizeMismatchThrowsVectors) {
  constexpr uint32_t kDim = 8;
  auto vectors = make_random_vectors(10, kDim, 61);
  auto labels = sequential_labels(10);
  auto seg_dir = build_segment(MetricType::L2, vectors, labels, kDim, "seg_00000001");

  // Truncate vectors.f32.bin by 1 byte.
  auto vec_path = seg_dir / "vectors.f32.bin";
  const auto orig_size = std::filesystem::file_size(vec_path);
  std::filesystem::resize_file(vec_path, orig_size - 1);
  EXPECT_THROW(DiskFlatSegmentSearcher{seg_dir}, std::runtime_error);
}

TEST_F(DiskFlatSearcherTest, FileSizeMismatchThrowsIds) {
  constexpr uint32_t kDim = 8;
  auto vectors = make_random_vectors(10, kDim, 62);
  auto labels = sequential_labels(10);
  auto seg_dir = build_segment(MetricType::L2, vectors, labels, kDim, "seg_00000001");

  auto ids_path = seg_dir / "ids.u64.bin";
  const auto orig_size = std::filesystem::file_size(ids_path);
  std::filesystem::resize_file(ids_path, orig_size - 1);
  EXPECT_THROW(DiskFlatSegmentSearcher{seg_dir}, std::runtime_error);
}

TEST_F(DiskFlatSearcherTest, DimCountOverflowRejected) {
  // Synthesize a manifest where dim*count*4 overflows uint64.
  auto seg_dir = seg_parent_ / "seg_00000001";
  std::filesystem::create_directories(seg_dir);
  // dim=2^32, count=2^33 ⇒ dim*count = 2^65 (overflow)
  // 2^32 = 4294967296; 2^33 = 8589934592
  std::ofstream ofs(seg_dir / "manifest.txt", std::ios::binary | std::ios::trunc);
  ofs << "version=1\n"
      << "segment_id=seg_00000001\n"
      << "index_type=disk_flat\n"
      << "metric=L2\n"
      << "dim=4294967296\n"
      << "count=8589934592\n"
      << "ids_file=ids.u64.bin\n"
      << "vectors_file=vectors.f32.bin\n";
  ofs.close();
  // Create empty ids and vectors files; we expect the overflow check to fire
  // before the size comparison anyway.
  { std::ofstream(seg_dir / "ids.u64.bin", std::ios::binary); }
  { std::ofstream(seg_dir / "vectors.f32.bin", std::ios::binary); }

  try {
    DiskFlatSegmentSearcher s(seg_dir);
    FAIL() << "expected runtime_error on overflowing dim*count";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("overflow"), std::string::npos)
        << "message must contain 'overflow', got: " << msg;
  }
}

}  // namespace
}  // namespace alaya::disk
