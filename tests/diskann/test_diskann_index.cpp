// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/graph/diskann/diskann_index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "simd/distance_l2.hpp"

namespace {

using alaya::diskann::DiskANNBuildParams;
using alaya::diskann::DiskANNIndex;
using alaya::diskann::DiskANNLoadParams;
using alaya::diskann::DiskANNSearchParams;

std::vector<float> make_vectors(uint64_t n, uint64_t dim, uint32_t seed = 123) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n * dim);
  for (auto &x : v) {
    x = dist(rng);
  }
  return v;
}

// External labels deliberately offset from internal ids to exercise the map.
std::vector<uint64_t> make_labels(uint64_t n) {
  std::vector<uint64_t> labels(n);
  for (uint64_t i = 0; i < n; ++i) {
    labels[i] = 1000 + i;
  }
  return labels;
}

std::vector<uint32_t> brute_force_ids(const std::vector<float> &v, uint64_t n, uint64_t dim,
                                      const float *q, uint32_t k) {
  std::vector<std::pair<float, uint32_t>> all;
  for (uint64_t i = 0; i < n; ++i) {
    all.emplace_back(alaya::simd::l2_sqr<float, float>(q, v.data() + i * dim, dim),
                     static_cast<uint32_t>(i));
  }
  std::sort(all.begin(), all.end());
  std::vector<uint32_t> ids;
  for (uint32_t i = 0; i < k && i < all.size(); ++i) {
    ids.push_back(all[i].second);
  }
  return ids;
}

class DiskANNIndexTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<uint64_t> counter{0};
    dir_ = std::filesystem::temp_directory_path() /
           ("diskann_idx_" + std::to_string(counter.fetch_add(1)));
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  std::string dir() const { return dir_.string(); }
  std::filesystem::path dir_;
};

// ----------------------------- build --------------------------------------

TEST_F(DiskANNIndexTest, BuildWithoutPQProducesExpectedFiles) {
  const uint64_t n = 200, dim = 32;
  const auto v = make_vectors(n, dim);
  const auto labels = make_labels(n);
  DiskANNBuildParams bp;
  bp.R = 32;
  bp.pq_n_chunks = 0;
  DiskANNIndex::build(dir(), v.data(), labels.data(), n, dim, bp);

  namespace fs = std::filesystem;
  EXPECT_TRUE(fs::exists(dir_ / "meta.bin"));
  EXPECT_TRUE(fs::exists(dir_ / "diskann.index"));
  EXPECT_TRUE(fs::exists(dir_ / "ids.bin"));
  EXPECT_TRUE(fs::exists(dir_ / "cache_ids.bin"));
  EXPECT_TRUE(fs::exists(dir_ / "cache_nodes.bin"));
  EXPECT_FALSE(fs::exists(dir_ / "pq_pivots.bin"));
  EXPECT_FALSE(fs::exists(dir_ / "pq_compressed.bin"));
}

TEST_F(DiskANNIndexTest, BuildWithPQProducesPQFiles) {
  const uint64_t n = 300, dim = 32;
  const auto v = make_vectors(n, dim);
  const auto labels = make_labels(n);
  DiskANNBuildParams bp;
  bp.R = 32;
  bp.pq_n_chunks = 8;
  DiskANNIndex::build(dir(), v.data(), labels.data(), n, dim, bp);

  namespace fs = std::filesystem;
  EXPECT_TRUE(fs::exists(dir_ / "pq_pivots.bin"));
  EXPECT_TRUE(fs::exists(dir_ / "pq_compressed.bin"));
}

TEST_F(DiskANNIndexTest, BuildRejectsZeroDim) {
  const auto v = make_vectors(10, 4);
  const auto labels = make_labels(10);
  EXPECT_THROW(DiskANNIndex::build(dir(), v.data(), labels.data(), 10, /*dim=*/0, {}),
               std::invalid_argument);
}

TEST_F(DiskANNIndexTest, BuildRejectsEmptyInput) {
  const auto v = make_vectors(1, 4);
  const auto labels = make_labels(1);
  EXPECT_THROW(DiskANNIndex::build(dir(), v.data(), labels.data(), /*n=*/0, 4, {}),
               std::invalid_argument);
}

TEST_F(DiskANNIndexTest, BuildRejectsExistingDirectoryBeforeWriting) {
  std::filesystem::create_directories(dir_);
  const auto v = make_vectors(10, 4);
  const auto labels = make_labels(10);
  EXPECT_THROW(DiskANNIndex::build(dir(), v.data(), labels.data(), 10, 4, {}), std::runtime_error);
  // Nothing should have been written into the pre-existing directory.
  EXPECT_FALSE(std::filesystem::exists(dir_ / "meta.bin"));
}

// ----------------------------- load ---------------------------------------

TEST_F(DiskANNIndexTest, LoadNoPQHasNoPQAndSearches) {
  const uint64_t n = 200, dim = 32;
  const auto v = make_vectors(n, dim);
  const auto labels = make_labels(n);
  DiskANNBuildParams bp;
  bp.R = 32;
  bp.pq_n_chunks = 0;
  DiskANNIndex::build(dir(), v.data(), labels.data(), n, dim, bp);

  DiskANNIndex idx;
  idx.load(dir(), {/*num_threads=*/2, /*beam_width=*/4});
  EXPECT_FALSE(idx.has_pq());
  EXPECT_EQ(idx.size(), n);
  EXPECT_EQ(idx.dim(), dim);

  std::vector<uint64_t> out_l(10);
  std::vector<float> out_d(10);
  const uint32_t cnt = idx.search(v.data(), 10, out_l.data(), out_d.data(),
                                  {/*L=*/64, /*use_pq=*/false, /*rerank=*/false});
  EXPECT_EQ(cnt, 10u);
}

TEST_F(DiskANNIndexTest, LoadPQHasPQ) {
  const uint64_t n = 300, dim = 32;
  const auto v = make_vectors(n, dim);
  const auto labels = make_labels(n);
  DiskANNBuildParams bp;
  bp.R = 32;
  bp.pq_n_chunks = 8;
  DiskANNIndex::build(dir(), v.data(), labels.data(), n, dim, bp);

  DiskANNIndex idx;
  idx.load(dir());
  EXPECT_TRUE(idx.has_pq());
}

TEST_F(DiskANNIndexTest, LoadRejectsMissingDirectory) {
  DiskANNIndex idx;
  EXPECT_THROW(idx.load((dir_ / "does_not_exist").string()), std::runtime_error);
}

TEST_F(DiskANNIndexTest, LoadRejectsTruncatedMeta) {
  const uint64_t n = 100, dim = 16;
  const auto v = make_vectors(n, dim);
  const auto labels = make_labels(n);
  DiskANNIndex::build(dir(), v.data(), labels.data(), n, dim, {});
  // Truncate meta.bin to a few bytes.
  {
    std::ofstream out((dir_ / "meta.bin").string(), std::ios::binary | std::ios::trunc);
    const char junk[4] = {1, 2, 3, 4};
    out.write(junk, sizeof(junk));
  }
  DiskANNIndex idx;
  EXPECT_THROW(idx.load(dir()), std::runtime_error);
}

// ----------------------------- search -------------------------------------

TEST_F(DiskANNIndexTest, SearchReturnsSortedValidTopK) {
  const uint64_t n = 400, dim = 32;
  const auto v = make_vectors(n, dim);
  const auto labels = make_labels(n);
  DiskANNBuildParams bp;
  bp.R = 32;
  bp.pq_n_chunks = 8;
  DiskANNIndex::build(dir(), v.data(), labels.data(), n, dim, bp);

  DiskANNIndex idx;
  idx.load(dir(), {2, 4});
  const float *q = v.data() + 17 * dim;
  std::vector<uint64_t> out_l(10);
  std::vector<float> out_d(10);
  // deterministic=true: this asserts the exact nearest neighbour is found, an
  // exactness guarantee the reproducible barrier path provides; the default
  // async path's recall is covered statistically by the e2e tests.
  const uint32_t cnt = idx.search(q, 10, out_l.data(), out_d.data(),
                                  {/*L=*/96, /*use_pq=*/true, /*rerank=*/true,
                                   /*rerank_count=*/0, /*deterministic=*/true});
  ASSERT_EQ(cnt, 10u);
  for (uint32_t i = 0; i < cnt; ++i) {
    EXPECT_GE(out_d[i], 0.0f);
    EXPECT_GE(out_l[i], 1000u);  // labels are 1000+id
    if (i > 0) {
      EXPECT_LE(out_d[i - 1], out_d[i]);  // ascending
    }
    // Reported distance must equal the exact L2 to that label's vector.
    const uint32_t internal = static_cast<uint32_t>(out_l[i] - 1000);
    const float exact = alaya::simd::l2_sqr<float, float>(q, v.data() + internal * dim, dim);
    EXPECT_NEAR(out_d[i], exact, 1e-2f * (1.0f + exact)) << "i=" << i;
  }
  // The nearest neighbour (the query is point 17) should be found.
  const auto truth = brute_force_ids(v, n, dim, q, 1);
  EXPECT_EQ(out_l[0] - 1000, truth[0]);
}

TEST_F(DiskANNIndexTest, SearchTopKExceedingSizeReturnsAll) {
  const uint64_t n = 50, dim = 16;
  const auto v = make_vectors(n, dim);
  const auto labels = make_labels(n);
  DiskANNBuildParams bp;
  bp.R = 24;
  DiskANNIndex::build(dir(), v.data(), labels.data(), n, dim, bp);
  DiskANNIndex idx;
  idx.load(dir());
  std::vector<uint64_t> out_l(200);
  std::vector<float> out_d(200);
  const uint32_t cnt = idx.search(v.data(), 200, out_l.data(), out_d.data(), {/*L=*/200});
  EXPECT_EQ(cnt, n);
}

TEST_F(DiskANNIndexTest, SearchRejectsNullQueryAndZeroTopK) {
  const uint64_t n = 60, dim = 16;
  const auto v = make_vectors(n, dim);
  const auto labels = make_labels(n);
  DiskANNIndex::build(dir(), v.data(), labels.data(), n, dim, {});
  DiskANNIndex idx;
  idx.load(dir());
  std::vector<uint64_t> out_l(10);
  std::vector<float> out_d(10);
  EXPECT_THROW(idx.search(nullptr, 10, out_l.data(), out_d.data()), std::invalid_argument);
  EXPECT_THROW(idx.search(v.data(), 0, out_l.data(), out_d.data()), std::invalid_argument);
}

// ----------------------------- batch --------------------------------------

TEST_F(DiskANNIndexTest, BatchSearchMatchesSequential) {
  const uint64_t n = 400, dim = 32;
  const uint32_t nq = 40, k = 10;
  const auto v = make_vectors(n, dim);
  const auto labels = make_labels(n);
  const auto queries = make_vectors(nq, dim, /*seed=*/777);
  DiskANNBuildParams bp;
  bp.R = 32;
  bp.pq_n_chunks = 8;
  DiskANNIndex::build(dir(), v.data(), labels.data(), n, dim, bp);

  DiskANNIndex idx;
  idx.load(dir(), {/*num_threads=*/4, /*beam_width=*/4});
  // deterministic=true: asserts byte-exact batch==sequential (the barrier's job).
  const DiskANNSearchParams sp{/*L=*/96, /*use_pq=*/true, /*rerank=*/true,
                               /*rerank_count=*/0, /*deterministic=*/true};

  std::vector<uint64_t> seq_l(nq * k);
  std::vector<float> seq_d(nq * k);
  for (uint32_t qi = 0; qi < nq; ++qi) {
    idx.search(queries.data() + qi * dim, k, seq_l.data() + qi * k, seq_d.data() + qi * k, sp);
  }

  std::vector<uint64_t> bat_l(nq * k);
  std::vector<float> bat_d(nq * k);
  idx.batch_search(queries.data(), nq, k, bat_l.data(), bat_d.data(), /*num_threads=*/4, sp);

  EXPECT_EQ(seq_l, bat_l);
  for (size_t i = 0; i < seq_d.size(); ++i) {
    EXPECT_FLOAT_EQ(seq_d[i], bat_d[i]) << "i=" << i;
  }
}

TEST_F(DiskANNIndexTest, NoPQBatchDeterministicMatchesSequential) {
  // No-PQ greedy search: with deterministic = true the batched per-expansion
  // barrier must reproduce sequential search() byte for byte across threads
  // (the async default may reorder equally-distant ties, so it is not asserted
  // here — its recall equivalence is covered by the e2e suite).
  const uint64_t n = 400, dim = 32;
  const uint32_t nq = 40, k = 10;
  const auto v = make_vectors(n, dim);
  const auto labels = make_labels(n);
  const auto queries = make_vectors(nq, dim, /*seed=*/777);
  DiskANNBuildParams bp;
  bp.R = 32;
  bp.pq_n_chunks = 0;  // No-PQ index
  DiskANNIndex::build(dir(), v.data(), labels.data(), n, dim, bp);

  DiskANNIndex idx;
  idx.load(dir(), {/*num_threads=*/4, /*beam_width=*/4});
  const DiskANNSearchParams sp{/*L=*/96, /*use_pq=*/false, /*rerank=*/false,
                               /*rerank_count=*/0, /*deterministic=*/true};

  std::vector<uint64_t> seq_l(nq * k);
  std::vector<float> seq_d(nq * k);
  for (uint32_t qi = 0; qi < nq; ++qi) {
    idx.search(queries.data() + qi * dim, k, seq_l.data() + qi * k, seq_d.data() + qi * k, sp);
  }

  std::vector<uint64_t> bat_l(nq * k);
  std::vector<float> bat_d(nq * k);
  idx.batch_search(queries.data(), nq, k, bat_l.data(), bat_d.data(), /*num_threads=*/4, sp);

  EXPECT_EQ(seq_l, bat_l);
  for (size_t i = 0; i < seq_d.size(); ++i) {
    EXPECT_FLOAT_EQ(seq_d[i], bat_d[i]) << "i=" << i;
  }
}

TEST_F(DiskANNIndexTest, BatchSearchSingleThread) {
  const uint64_t n = 200, dim = 16;
  const uint32_t nq = 8, k = 5;
  const auto v = make_vectors(n, dim);
  const auto labels = make_labels(n);
  const auto queries = make_vectors(nq, dim, /*seed=*/9);
  DiskANNIndex::build(dir(), v.data(), labels.data(), n, dim, {});
  DiskANNIndex idx;
  idx.load(dir(), {/*num_threads=*/2, /*beam_width=*/4});
  std::vector<uint64_t> out_l(nq * k);
  std::vector<float> out_d(nq * k);
  idx.batch_search(queries.data(), nq, k, out_l.data(), out_d.data(), /*num_threads=*/1,
                   {/*L=*/64, /*use_pq=*/false, /*rerank=*/false});
  for (uint32_t qi = 0; qi < nq; ++qi) {
    EXPECT_GE(out_l[qi * k], 1000u);  // each query produced results
  }
}

}  // namespace
