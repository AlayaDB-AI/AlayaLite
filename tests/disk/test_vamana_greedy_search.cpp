// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_greedy_search.hpp"
#include "index/graph/vamana/vamana_reader.hpp"
#include "index/graph/vamana/vamana_writer.hpp"
#include "simd/distance_l2.hpp"

namespace {

std::atomic<uint64_t> g_path_counter{0};

std::filesystem::path unique_temp_path(const std::string &stem) {
  const uint64_t n = g_path_counter.fetch_add(1, std::memory_order_relaxed);
  std::string name = stem + "_pid" + std::to_string(::getpid()) + "_" + std::to_string(n) +
                     ".index";
  return std::filesystem::temp_directory_path() / name;
}

std::vector<float> make_data(uint32_t n, uint32_t dim, uint64_t seed) {
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> out(static_cast<size_t>(n) * dim);
  for (auto &v : out) {
    v = dist(rng);
  }
  return out;
}

struct BuiltDataset {
  std::filesystem::path path;
  std::vector<float> data;
  uint32_t medoid;
};

BuiltDataset build_dataset(uint32_t n,
                           uint32_t dim,
                           uint32_t R,
                           uint32_t L,
                           float alpha,
                           uint64_t seed,
                           const std::string &stem) {
  BuiltDataset bd;
  bd.data = make_data(n, dim, seed);
  alaya::vamana::VamanaBuildParams params{};
  params.R = R;
  params.L = L;
  params.alpha = alpha;
  params.num_threads = 1;
  params.seed = seed;
  alaya::vamana::VamanaBuilder builder(bd.data.data(), n, dim, params);
  builder.build();
  bd.medoid = builder.medoid();
  bd.path = unique_temp_path(stem);
  alaya::vamana::save_graph(builder.graph(), bd.path, R, bd.medoid, /*frozen_pts=*/0);
  return bd;
}

// Byte-wise FNV-1a hash for buffer-immutability checks. Independent of
// `std::hash` so a future stdlib change can't accidentally make the
// before/after comparison succeed without the bytes being equal.
uint64_t buffer_hash(const void *data, size_t bytes) {
  const auto *p = static_cast<const unsigned char *>(data);
  uint64_t h = 14695981039346656037ULL;  // FNV offset basis
  for (size_t i = 0; i < bytes; ++i) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= 1099511628211ULL;  // FNV prime
  }
  return h;
}

}  // namespace

class VamanaGreedySearchTest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (const auto &p : owned_paths_) {
      std::error_code ec;
      std::filesystem::remove(p, ec);
    }
  }

  void track(const std::filesystem::path &p) { owned_paths_.push_back(p); }

  std::vector<std::filesystem::path> owned_paths_;
};

// 10.1 — smoke test: shape, range, monotonic.
TEST_F(VamanaGreedySearchTest, Smoke) {
  const uint32_t N = 256, dim = 16;
  BuiltDataset bd = build_dataset(N, dim, 16, 64, 1.2f, 42, "vgs_smoke");
  track(bd.path);

  alaya::vamana::VamanaReader reader{bd.path};
  alaya::vamana::VamanaGreedySearch search(reader, bd.data.data(), dim);

  std::mt19937 rng(101);
  std::uniform_real_distribution<float> qd(-1.0f, 1.0f);
  std::vector<float> query(dim);
  for (auto &v : query) {
    v = qd(rng);
  }

  auto hits = search.search(query.data(), /*top_k=*/10, /*search_list_size=*/64);
  ASSERT_EQ(hits.size(), 10u);
  for (const auto &h : hits) {
    EXPECT_LT(h.id, N);
  }
  for (size_t i = 1; i < hits.size(); ++i) {
    EXPECT_LE(hits[i - 1].distance, hits[i].distance) << "result " << i << " out of order";
  }
}

// 10.2 — first inserted candidate is always the medoid.
TEST_F(VamanaGreedySearchTest, StartsFromMedoid) {
  const uint32_t N = 128, dim = 8;
  BuiltDataset bd = build_dataset(N, dim, 16, 64, 1.2f, 42, "vgs_medoid");
  track(bd.path);

  alaya::vamana::VamanaReader reader{bd.path};
  alaya::vamana::VamanaGreedySearch search(reader, bd.data.data(), dim);

  std::vector<float> query(dim, 0.0f);
  (void)search.search(query.data(), /*top_k=*/5, /*search_list_size=*/32);

  ASSERT_GE(search.last_visited_order().size(), 1u);
  EXPECT_EQ(search.last_visited_order()[0], reader.start());
}

// 10.3 — top_k = 0 is rejected.
TEST_F(VamanaGreedySearchTest, RejectsTopKZero) {
  const uint32_t N = 32, dim = 4;
  BuiltDataset bd = build_dataset(N, dim, 8, 32, 1.2f, 42, "vgs_topk0");
  track(bd.path);

  alaya::vamana::VamanaReader reader{bd.path};
  alaya::vamana::VamanaGreedySearch search(reader, bd.data.data(), dim);
  std::vector<float> query(dim, 0.0f);

  EXPECT_THROW(search.search(query.data(), /*top_k=*/0, /*search_list_size=*/16),
               std::runtime_error);
}

// 10.4 — search_list_size = 0 is rejected.
TEST_F(VamanaGreedySearchTest, RejectsSearchListSizeZero) {
  const uint32_t N = 32, dim = 4;
  BuiltDataset bd = build_dataset(N, dim, 8, 32, 1.2f, 42, "vgs_sls0");
  track(bd.path);

  alaya::vamana::VamanaReader reader{bd.path};
  alaya::vamana::VamanaGreedySearch search(reader, bd.data.data(), dim);
  std::vector<float> query(dim, 0.0f);

  EXPECT_THROW(search.search(query.data(), /*top_k=*/5, /*search_list_size=*/0),
               std::runtime_error);
}

// 10.5 — search_list_size < top_k is rejected.
TEST_F(VamanaGreedySearchTest, RejectsSearchListSmallerThanTopK) {
  const uint32_t N = 32, dim = 4;
  BuiltDataset bd = build_dataset(N, dim, 8, 32, 1.2f, 42, "vgs_sls_lt");
  track(bd.path);

  alaya::vamana::VamanaReader reader{bd.path};
  alaya::vamana::VamanaGreedySearch search(reader, bd.data.data(), dim);
  std::vector<float> query(dim, 0.0f);

  EXPECT_THROW(search.search(query.data(), /*top_k=*/10, /*search_list_size=*/5),
               std::runtime_error);
}

// 10.6 — top_k > num_nodes returns num_nodes results.
TEST_F(VamanaGreedySearchTest, TopKCapsToNumNodes) {
  const uint32_t N = 10, dim = 4;
  BuiltDataset bd = build_dataset(N, dim, /*R=*/4, /*L=*/16, 1.2f, 42, "vgs_topk_cap");
  track(bd.path);

  alaya::vamana::VamanaReader reader{bd.path};
  alaya::vamana::VamanaGreedySearch search(reader, bd.data.data(), dim);
  std::vector<float> query(dim, 0.0f);

  auto hits = search.search(query.data(), /*top_k=*/100, /*search_list_size=*/100);
  EXPECT_EQ(hits.size(), static_cast<size_t>(N));
  std::unordered_set<uint32_t> seen;
  for (const auto &h : hits) {
    seen.insert(h.id);
  }
  EXPECT_EQ(seen.size(), static_cast<size_t>(N));
}

// 10.7 — query buffer is not mutated.
TEST_F(VamanaGreedySearchTest, QueryBufferNotMutated) {
  const uint32_t N = 128, dim = 8;
  BuiltDataset bd = build_dataset(N, dim, 16, 64, 1.2f, 42, "vgs_query_immut");
  track(bd.path);

  alaya::vamana::VamanaReader reader{bd.path};
  alaya::vamana::VamanaGreedySearch search(reader, bd.data.data(), dim);

  std::vector<float> query(dim);
  for (uint32_t i = 0; i < dim; ++i) {
    query[i] = 0.123f * static_cast<float>(i + 1);
  }

  const uint64_t hash_before = buffer_hash(query.data(), query.size() * sizeof(float));
  (void)search.search(query.data(), /*top_k=*/10, /*search_list_size=*/64);
  const uint64_t hash_after = buffer_hash(query.data(), query.size() * sizeof(float));
  EXPECT_EQ(hash_before, hash_after);
}

// 10.8 — vectors buffer is not mutated.
TEST_F(VamanaGreedySearchTest, VectorsBufferNotMutated) {
  const uint32_t N = 128, dim = 8;
  BuiltDataset bd = build_dataset(N, dim, 16, 64, 1.2f, 42, "vgs_vec_immut");
  track(bd.path);

  alaya::vamana::VamanaReader reader{bd.path};
  alaya::vamana::VamanaGreedySearch search(reader, bd.data.data(), dim);

  std::vector<float> query(dim, 0.0f);
  const uint64_t hash_before = buffer_hash(bd.data.data(), bd.data.size() * sizeof(float));
  (void)search.search(query.data(), /*top_k=*/10, /*search_list_size=*/64);
  const uint64_t hash_after = buffer_hash(bd.data.data(), bd.data.size() * sizeof(float));
  EXPECT_EQ(hash_before, hash_after);
}

// 10.9 — graph is not mutated.
TEST_F(VamanaGreedySearchTest, GraphNotMutated) {
  const uint32_t N = 64, dim = 8;
  BuiltDataset bd = build_dataset(N, dim, 16, 64, 1.2f, 42, "vgs_graph_immut");
  track(bd.path);

  alaya::vamana::VamanaReader reader{bd.path};
  std::vector<std::vector<uint32_t>> snapshot = reader.graph();

  alaya::vamana::VamanaGreedySearch search(reader, bd.data.data(), dim);
  std::vector<float> query(dim, 0.0f);
  (void)search.search(query.data(), /*top_k=*/10, /*search_list_size=*/64);

  ASSERT_EQ(reader.graph().size(), snapshot.size());
  for (size_t i = 0; i < snapshot.size(); ++i) {
    EXPECT_EQ(reader.graph()[i], snapshot[i]) << "node " << i;
  }
}

// 10.10 — recall@10 ≥ 0.7 against brute-force ground truth.
TEST_F(VamanaGreedySearchTest, RecallAgainstBruteforceL2) {
  const uint32_t N = 2048, dim = 16;
  BuiltDataset bd = build_dataset(N, dim, /*R=*/16, /*L=*/64, 1.2f, /*seed=*/42,
                                  "vgs_recall");
  track(bd.path);

  alaya::vamana::VamanaReader reader{bd.path};
  alaya::vamana::VamanaGreedySearch search(reader, bd.data.data(), dim);

  std::mt19937 query_rng(43);
  std::uniform_real_distribution<float> qd(-1.0f, 1.0f);
  const uint32_t num_queries = 50;
  const uint32_t top_k = 10;
  const uint32_t search_list_size = 64;

  double recall_sum = 0.0;
  for (uint32_t q = 0; q < num_queries; ++q) {
    std::vector<float> query(dim);
    for (auto &v : query) {
      v = qd(query_rng);
    }

    // Brute-force ground truth using the same kernel.
    std::vector<std::pair<float, uint32_t>> bf;
    bf.reserve(N);
    for (uint32_t i = 0; i < N; ++i) {
      const float d = alaya::simd::l2_sqr<float, float>(
          query.data(), bd.data.data() + static_cast<size_t>(i) * dim, dim);
      bf.emplace_back(d, i);
    }
    std::partial_sort(bf.begin(), bf.begin() + top_k, bf.end());

    std::unordered_set<uint32_t> truth;
    for (uint32_t i = 0; i < top_k; ++i) {
      truth.insert(bf[i].second);
    }

    auto hits = search.search(query.data(), top_k, search_list_size);
    ASSERT_EQ(hits.size(), top_k);
    uint32_t matched = 0;
    for (const auto &h : hits) {
      if (truth.count(h.id) > 0) {
        matched++;
      }
    }
    recall_sum += static_cast<double>(matched) / static_cast<double>(top_k);
  }

  const double avg_recall = recall_sum / static_cast<double>(num_queries);
  EXPECT_GE(avg_recall, 0.7) << "avg recall@10 = " << avg_recall;
}

// 10.11 — duplicate vectors produce bit-identical distances; tie-break
// orders by ascending internal id.
//
// This test hand-crafts a minimal 2-node graph via `save_graph` directly
// (bypassing VamanaBuilder) so the assertion does not depend on
// builder-level reachability decisions. The graph is simply
// {0 -> 1, 1 -> 0}, vectors are zero-vectors so every distance is 0,
// and the search must visit both ids.
TEST_F(VamanaGreedySearchTest, TieBreakIsAscendingId) {
  const auto path = unique_temp_path("vgs_tie");
  track(path);

  std::vector<std::vector<uint32_t>> graph = {{1u}, {0u}};
  alaya::vamana::save_graph(graph, path, /*max_degree=*/1u, /*start=*/0u,
                            /*frozen_pts=*/0u);

  const uint32_t dim = 2;
  std::vector<float> vectors(2u * dim, 0.0f);
  std::vector<float> query(dim, 0.0f);

  alaya::vamana::VamanaReader reader{path};
  alaya::vamana::VamanaGreedySearch search(reader, vectors.data(), dim);

  auto hits = search.search(query.data(), /*top_k=*/2, /*search_list_size=*/2);
  ASSERT_EQ(hits.size(), 2u);
  EXPECT_EQ(hits[0].id, 0u) << "ascending-id tie-break violated";
  EXPECT_EQ(hits[1].id, 1u);
  EXPECT_EQ(hits[0].distance, hits[1].distance);
  EXPECT_FLOAT_EQ(hits[0].distance, 0.0f);
}
