// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/graph/diskann/beam_search.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/diskann/node_cache.hpp"
#include "index/graph/diskann/pq_table.hpp"
#include "index/graph/diskann/search_scratch.hpp"
#include "index/graph/laser/utils/aligned_file_reader_factory.hpp"
#include "simd/distance_l2.hpp"

namespace {

using alaya::diskann::cached_beam_search;
using alaya::diskann::DiskLayoutGeometry;
using alaya::diskann::NodeCache;
using alaya::diskann::PQTable;
using alaya::diskann::scan_and_insert_neighbors;
using alaya::diskann::SearchContext;
using alaya::diskann::SearchParams;
using alaya::diskann::SearchStats;
using alaya::diskann::ThreadData;
using alaya::diskann::write_disk_layout;
using alaya::vamana::Neighbor;
using alaya::vamana::NeighborPriorityQueue;

std::vector<float> make_vectors(uint64_t n, uint64_t dim, uint32_t seed = 31) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n * dim);
  for (auto &x : v) {
    x = dist(rng);
  }
  return v;
}

// Complete graph capped at R: each node points to the first R other ids.
std::vector<std::vector<uint32_t>> make_complete_graph(uint64_t n, uint32_t r) {
  std::vector<std::vector<uint32_t>> g(n);
  for (uint64_t i = 0; i < n; ++i) {
    for (uint64_t j = 0; j < n && g[i].size() < r; ++j) {
      if (j != i) {
        g[i].push_back(static_cast<uint32_t>(j));
      }
    }
  }
  return g;
}

uint32_t medoid_of(const std::vector<float> &v, uint64_t n, uint64_t dim) {
  std::vector<float> centroid(dim, 0.0f);
  for (uint64_t i = 0; i < n; ++i) {
    for (uint64_t d = 0; d < dim; ++d) {
      centroid[d] += v[i * dim + d];
    }
  }
  for (auto &c : centroid) {
    c /= static_cast<float>(n);
  }
  uint32_t best = 0;
  float best_d = std::numeric_limits<float>::max();
  for (uint64_t i = 0; i < n; ++i) {
    const float d = alaya::simd::l2_sqr<float, float>(centroid.data(), v.data() + i * dim, dim);
    if (d < best_d) {
      best_d = d;
      best = static_cast<uint32_t>(i);
    }
  }
  return best;
}

std::vector<std::pair<uint32_t, float>> brute_force(const std::vector<float> &v, uint64_t n,
                                                    uint64_t dim, const float *q, uint32_t k) {
  std::vector<std::pair<uint32_t, float>> all;
  all.reserve(n);
  for (uint64_t i = 0; i < n; ++i) {
    all.emplace_back(static_cast<uint32_t>(i),
                     alaya::simd::l2_sqr<float, float>(q, v.data() + i * dim, dim));
  }
  std::sort(all.begin(), all.end(), [](const auto &a, const auto &b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  });
  if (all.size() > k) {
    all.resize(k);
  }
  return all;
}

// ---- Pure neighbor-processing tests (no disk) ------------------------------

TEST(BeamScanInsert, SkipsVisitedAndInsertsRest) {
  // n <= 256 => lossless PQ; here we only check the visited/dedup bookkeeping.
  const uint64_t n = 50, dim = 8;
  const uint32_t n_chunks = 2;
  const auto v = make_vectors(n, dim);
  PQTable pq;
  pq.train(v.data(), n, dim, n_chunks);
  pq.encode(v.data(), n);
  std::vector<float> table(static_cast<size_t>(n_chunks) * 256);
  pq.preprocess_query(v.data(), table.data());

  NeighborPriorityQueue retset(100);
  std::unordered_set<uint32_t> visited;
  for (uint32_t id = 0; id < 10; ++id) {
    visited.insert(id);  // 10 already visited
  }
  std::vector<uint32_t> nbrs(32);
  for (uint32_t i = 0; i < 32; ++i) {
    nbrs[i] = i;  // 0..31; 0..9 are visited
  }
  scan_and_insert_neighbors(retset, visited, nbrs.data(), 32, pq, table.data(), /*num_points=*/n);
  EXPECT_EQ(retset.size(), 22u);  // 32 - 10 already visited
}

TEST(BeamScanInsert, PQPrunesUncompetitiveNeighbors) {
  const uint64_t n = 10, dim = 8;
  const uint32_t n_chunks = 2;
  const auto v = make_vectors(n, dim);
  PQTable pq;  // n <= 256 => lossless: pq_distance == exact L2
  pq.train(v.data(), n, dim, n_chunks);
  pq.encode(v.data(), n);

  std::vector<float> table(static_cast<size_t>(n_chunks) * 256);
  pq.preprocess_query(v.data() + 0 * dim, table.data());  // query == point 0

  NeighborPriorityQueue retset(4);  // capacity 4
  std::unordered_set<uint32_t> visited;
  std::vector<uint32_t> nbrs = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  scan_and_insert_neighbors(retset, visited, nbrs.data(),
                            static_cast<uint32_t>(nbrs.size()), pq, table.data(), n);
  EXPECT_EQ(retset.size(), 4u);  // bounded
  // The globally farthest point from point 0 must have been pruned.
  const auto bf = brute_force(v, n, dim, v.data(), n);  // sorted ascending
  const uint32_t farthest = bf.back().first;
  bool present = false;
  for (size_t i = 0; i < retset.size(); ++i) {
    if (retset[i].id == farthest) {
      present = true;
    }
  }
  EXPECT_FALSE(present) << "farthest id=" << farthest << " should be pruned";
}

// ---- Beam search over a real (tiny) on-disk index --------------------------

class BeamSearchTest : public ::testing::Test {
 protected:
  void build(uint64_t n, uint64_t dim, uint32_t r, uint32_t n_chunks, double cache_ratio,
             uint32_t beam = 4) {
    n_ = n;
    dim_ = dim;
    r_ = r;
    has_pq_ = n_chunks > 0;
    vecs_ = make_vectors(n, dim);
    graph_ = make_complete_graph(n, r);
    medoid_ = medoid_of(vecs_, n, dim);
    geom_ = DiskLayoutGeometry::compute(dim, r);

    index_path_ = std::filesystem::temp_directory_path() /
                  ("diskann_beam_" + std::to_string(counter_.fetch_add(1)) + ".index");
    write_disk_layout(index_path_.string(), vecs_.data(), graph_, {n, dim, r, medoid_});

    cache_.generate(graph_, vecs_.data(), medoid_, n, dim, r, cache_ratio);
    if (has_pq_) {
      pq_.train(vecs_.data(), n, dim, n_chunks);
      pq_.encode(vecs_.data(), n);
    }

    reader_ = make_aligned_file_reader();
    reader_->open(index_path_.string());
    reader_->register_thread();
    td_.ctx_ = reader_->get_ctx();
    td_.alloc_scratch(2u * beam, geom_.page_size, has_pq_ ? n_chunks * 256u : 0u);
    beam_ = beam;
  }

  void TearDown() override {
    td_.free_scratch();
    if (reader_) {
      reader_->close();
      reader_->deregister_all_threads();
    }
    std::error_code ec;
    std::filesystem::remove(index_path_, ec);
  }

  std::vector<std::pair<uint32_t, float>> search(const float *q, uint32_t top_k, bool use_pq,
                                                 bool rerank, uint32_t L = 50) {
    SearchContext ctx;
    ctx.reader = reader_.get();
    ctx.geom = &geom_;
    ctx.cache = &cache_;
    ctx.pq = has_pq_ ? &pq_ : nullptr;
    ctx.medoid = medoid_;
    ctx.num_points = n_;
    SearchParams p;
    p.search_list_size = L;
    p.beam_width = beam_;
    p.use_pq = use_pq;
    p.rerank = rerank;
    last_stats_ = SearchStats{};
    return cached_beam_search(ctx, q, top_k, p, td_, &last_stats_);
  }

  static std::atomic<uint64_t> counter_;
  uint64_t n_ = 0, dim_ = 0;
  uint32_t r_ = 0, medoid_ = 0, beam_ = 4;
  bool has_pq_ = false;
  std::vector<float> vecs_;
  std::vector<std::vector<uint32_t>> graph_;
  DiskLayoutGeometry geom_;
  NodeCache cache_;
  PQTable pq_;
  std::filesystem::path index_path_;
  std::unique_ptr<AlignedFileReader> reader_;
  ThreadData td_;
  SearchStats last_stats_;
};
std::atomic<uint64_t> BeamSearchTest::counter_{0};

TEST_F(BeamSearchTest, MedoidProcessedFirst) {
  build(/*n=*/20, /*dim=*/8, /*r=*/19, /*n_chunks=*/0, /*cache_ratio=*/0.0);
  const auto res = search(vecs_.data() + 3 * dim_, /*top_k=*/5, /*use_pq=*/false, /*rerank=*/false);
  ASSERT_FALSE(last_stats_.read_order.empty());
  EXPECT_EQ(last_stats_.read_order[0], medoid_);
}

TEST_F(BeamSearchTest, NoDuplicateReads) {
  build(20, 8, 19, 0, 0.0);
  (void)search(vecs_.data() + 1 * dim_, 5, false, false);
  std::unordered_set<uint32_t> seen;
  for (uint32_t id : last_stats_.read_order) {
    EXPECT_TRUE(seen.insert(id).second) << "duplicate read of id=" << id;
  }
}

TEST_F(BeamSearchTest, NoPQFindsNearestNeighbor) {
  build(20, 8, 19, 0, 0.0);  // complete graph => fully reachable
  for (uint32_t qi = 0; qi < 20; ++qi) {
    const float *q = vecs_.data() + qi * dim_;
    const auto res = search(q, /*top_k=*/1, /*use_pq=*/false, /*rerank=*/false, /*L=*/40);
    ASSERT_EQ(res.size(), 1u);
    const auto bf = brute_force(vecs_, n_, dim_, q, 1);
    EXPECT_EQ(res[0].first, bf[0].first) << "qi=" << qi;
    EXPECT_NEAR(res[0].second, bf[0].second, 1e-4f);
  }
}

TEST_F(BeamSearchTest, PQRerankGivesExactTopK) {
  // n <= 256 => lossless PQ; with rerank the results are exact top-k.
  build(20, 8, 19, /*n_chunks=*/4, /*cache_ratio=*/0.0);
  const float *q = vecs_.data() + 7 * dim_;
  const auto res = search(q, /*top_k=*/5, /*use_pq=*/true, /*rerank=*/true, /*L=*/40);
  const auto bf = brute_force(vecs_, n_, dim_, q, 5);
  ASSERT_EQ(res.size(), bf.size());
  for (size_t i = 0; i < res.size(); ++i) {
    EXPECT_EQ(res[i].first, bf[i].first) << "i=" << i;
    EXPECT_NEAR(res[i].second, bf[i].second, 1e-3f) << "i=" << i;
  }
}

TEST_F(BeamSearchTest, PQNoRerankReturnsSortedResults) {
  build(20, 8, 19, 4, 0.0);
  const float *q = vecs_.data() + 2 * dim_;
  const auto res = search(q, /*top_k=*/5, /*use_pq=*/true, /*rerank=*/false, /*L=*/40);
  ASSERT_EQ(res.size(), 5u);
  for (size_t i = 1; i < res.size(); ++i) {
    EXPECT_LE(res[i - 1].second, res[i].second);
  }
}

TEST_F(BeamSearchTest, CacheHitPathServesAllFromMemory) {
  build(20, 8, 19, /*n_chunks=*/0, /*cache_ratio=*/1.0);  // cache everything
  const float *q = vecs_.data() + 4 * dim_;
  const auto res = search(q, /*top_k=*/3, /*use_pq=*/false, /*rerank=*/false, /*L=*/40);
  EXPECT_GT(last_stats_.n_cache_hits, 0u);
  EXPECT_EQ(last_stats_.n_ios, 0u);  // nothing read from disk
  const auto bf = brute_force(vecs_, n_, dim_, q, 3);
  ASSERT_EQ(res.size(), 3u);
  EXPECT_EQ(res[0].first, bf[0].first);
}

TEST_F(BeamSearchTest, TopKExceedingIndexSizeReturnsAll) {
  build(12, 8, 11, 0, 0.0);
  const auto res = search(vecs_.data(), /*top_k=*/50, false, false, /*L=*/60);
  EXPECT_EQ(res.size(), 12u);  // only 12 points exist
}

}  // namespace
