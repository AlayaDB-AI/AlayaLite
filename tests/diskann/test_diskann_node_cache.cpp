// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/graph/diskann/node_cache.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

#include "index/graph/diskann/disk_layout.hpp"

namespace {

using alaya::diskann::DiskLayoutGeometry;
using alaya::diskann::NodeCache;
using alaya::diskann::NodeRecordView;

std::vector<float> make_vectors(uint64_t n, uint64_t dim, uint32_t seed = 19) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n * dim);
  for (auto &x : v) {
    x = dist(rng);
  }
  return v;
}

// Connected graph: ring forward edges + a few deterministic random edges.
std::vector<std::vector<uint32_t>> make_connected_graph(uint64_t n, uint32_t degree,
                                                        uint32_t seed = 5) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint32_t> pick(0, static_cast<uint32_t>(n - 1));
  std::vector<std::vector<uint32_t>> g(n);
  for (uint64_t i = 0; i < n; ++i) {
    std::unordered_set<uint32_t> nb;
    nb.insert(static_cast<uint32_t>((i + 1) % n));  // keep it connected
    while (nb.size() < degree && nb.size() < n - 1) {
      uint32_t v = pick(rng);
      if (v != i) {
        nb.insert(v);
      }
    }
    g[i].assign(nb.begin(), nb.end());
  }
  return g;
}

class NodeCacheTest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (const auto &p : owned_) {
      std::error_code ec;
      std::filesystem::remove(p, ec);
    }
  }
  std::filesystem::path temp_path(const std::string &tag) {
    static std::atomic<uint64_t> counter{0};
    auto p = std::filesystem::temp_directory_path() /
             ("diskann_cache_" + tag + "_" + std::to_string(counter.fetch_add(1)) + ".bin");
    owned_.push_back(p);
    return p;
  }
  std::vector<std::filesystem::path> owned_;
};

TEST_F(NodeCacheTest, CacheIncludesMedoidFirst) {
  const uint64_t n = 1000, dim = 16;
  const uint32_t r = 16, medoid = 421;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_connected_graph(n, r);
  NodeCache cache;
  cache.generate(graph, vecs.data(), medoid, n, dim, r, /*cache_ratio=*/0.05);
  ASSERT_GT(cache.size(), 0u);
  EXPECT_EQ(cache.ids()[0], medoid);
  EXPECT_NE(cache.lookup(medoid), nullptr);
}

TEST_F(NodeCacheTest, CacheRespectsRatio) {
  const uint64_t n = 10000, dim = 8;
  const uint32_t r = 8, medoid = 0;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_connected_graph(n, r);
  NodeCache cache;
  cache.generate(graph, vecs.data(), medoid, n, dim, r, /*cache_ratio=*/0.05);
  const uint64_t expected = static_cast<uint64_t>(std::ceil(0.05 * n));  // 500
  EXPECT_LE(cache.size(), expected);
  EXPECT_EQ(cache.size(), expected);  // ring keeps the graph reachable
}

TEST_F(NodeCacheTest, CacheRatioZeroIsEmpty) {
  const uint64_t n = 100, dim = 8;
  const uint32_t r = 8;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_connected_graph(n, r);
  NodeCache cache;
  cache.generate(graph, vecs.data(), 0, n, dim, r, /*cache_ratio=*/0.0);
  EXPECT_EQ(cache.size(), 0u);
  EXPECT_EQ(cache.lookup(0), nullptr);
}

TEST_F(NodeCacheTest, CacheRatioCappedAtN) {
  const uint64_t n = 200, dim = 8;
  const uint32_t r = 8;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_connected_graph(n, r);
  NodeCache cache;
  cache.generate(graph, vecs.data(), 3, n, dim, r, /*cache_ratio=*/2.0);  // >1 caps at n
  EXPECT_EQ(cache.size(), n);
}

TEST_F(NodeCacheTest, LookupReturnsCorrectNodeData) {
  const uint64_t n = 500, dim = 24;
  const uint32_t r = 12, medoid = 100;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_connected_graph(n, r);
  NodeCache cache;
  cache.generate(graph, vecs.data(), medoid, n, dim, r, /*cache_ratio=*/0.2);

  for (const uint32_t id : cache.ids()) {
    const char *rec = cache.lookup(id);
    ASSERT_NE(rec, nullptr) << "id=" << id;
    NodeRecordView view{rec, dim};
    for (uint64_t d = 0; d < dim; ++d) {
      EXPECT_FLOAT_EQ(view.coords()[d], vecs[id * dim + d]) << "id=" << id << " d=" << d;
    }
    const uint32_t expect_nbrs =
        static_cast<uint32_t>(graph[id].size() > r ? r : graph[id].size());
    ASSERT_EQ(view.n_nbrs(), expect_nbrs) << "id=" << id;
    for (uint32_t k = 0; k < expect_nbrs; ++k) {
      EXPECT_EQ(view.nbrs()[k], graph[id][k]) << "id=" << id << " k=" << k;
    }
  }
}

TEST_F(NodeCacheTest, CacheMissReturnsNullptr) {
  const uint64_t n = 5000, dim = 8;
  const uint32_t r = 6, medoid = 0;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_connected_graph(n, r);
  NodeCache cache;
  cache.generate(graph, vecs.data(), medoid, n, dim, r, /*cache_ratio=*/0.02);  // 100 nodes

  std::unordered_set<uint32_t> cached(cache.ids().begin(), cache.ids().end());
  uint32_t uncached = 0;
  for (uint32_t id = 0; id < n; ++id) {
    if (cached.find(id) == cached.end()) {
      uncached = id;
      break;
    }
  }
  EXPECT_EQ(cache.lookup(uncached), nullptr);
}

TEST_F(NodeCacheTest, FileRoundtrip) {
  const uint64_t n = 800, dim = 16;
  const uint32_t r = 10, medoid = 55;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_connected_graph(n, r);
  NodeCache cache;
  cache.generate(graph, vecs.data(), medoid, n, dim, r, /*cache_ratio=*/0.15);

  const auto ids_path = temp_path("ids");
  const auto nodes_path = temp_path("nodes");
  cache.save(ids_path.string(), nodes_path.string());

  NodeCache loaded;
  loaded.load(ids_path.string(), nodes_path.string());

  ASSERT_EQ(loaded.size(), cache.size());
  EXPECT_EQ(loaded.node_len(), cache.node_len());
  EXPECT_EQ(loaded.ids(), cache.ids());
  for (const uint32_t id : cache.ids()) {
    const char *a = cache.lookup(id);
    const char *b = loaded.lookup(id);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr) << "id=" << id;
    EXPECT_EQ(std::memcmp(a, b, cache.node_len()), 0) << "id=" << id;
  }
}

TEST_F(NodeCacheTest, ConcurrentReadSafety) {
  const uint64_t n = 2000, dim = 32;
  const uint32_t r = 16, medoid = 7;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_connected_graph(n, r);
  NodeCache cache;
  cache.generate(graph, vecs.data(), medoid, n, dim, r, /*cache_ratio=*/0.25);

  const std::vector<uint32_t> ids = cache.ids();
  std::atomic<int> mismatches{0};
  auto worker = [&]() {
    for (int rep = 0; rep < 50; ++rep) {
      for (const uint32_t id : ids) {
        const char *rec = cache.lookup(id);
        if (rec == nullptr) {
          mismatches.fetch_add(1);
          continue;
        }
        NodeRecordView view{rec, dim};
        // Spot-check a few coordinates against the ground truth.
        if (view.coords()[0] != vecs[id * dim + 0] ||
            view.coords()[dim - 1] != vecs[id * dim + dim - 1]) {
          mismatches.fetch_add(1);
        }
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(4);
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back(worker);
  }
  for (auto &th : threads) {
    th.join();
  }
  EXPECT_EQ(mismatches.load(), 0);
}

}  // namespace
