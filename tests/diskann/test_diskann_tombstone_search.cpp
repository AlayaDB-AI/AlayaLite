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
#include <vector>

#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/diskann/node_cache.hpp"
#include "index/graph/diskann/search_scratch.hpp"
#include "index/graph/diskann/tombstone_bitmap.hpp"
#include "index/graph/laser/utils/aligned_file_reader_factory.hpp"

namespace {

using alaya::diskann::disk_greedy_search;
using alaya::diskann::DiskLayoutGeometry;
using alaya::diskann::NodeCache;
using alaya::diskann::SearchContext;
using alaya::diskann::SearchParams;
using alaya::diskann::SearchStats;
using alaya::diskann::ThreadData;
using alaya::diskann::TombstoneBitmap;
using alaya::diskann::write_disk_layout;

// Engineered 6-node layout (dim=2) where node 4 is reachable ONLY through
// node 1: medoid 0 -> 1 -> 4.
//
//   id : position        neighbors
//    0 : (0, 0)  medoid   {1, 2, 3, 5}
//    1 : (1, 0)           {0, 4}
//    2 : (0, 1)           {0}
//    3 : (0,-1)           {0}
//    4 : (2, 0)  hidden   {1}
//    5 : (-1,0)           {0}
struct Scenario {
  std::vector<float> vecs = {0, 0, 1, 0, 0, 1, 0, -1, 2, 0, -1, 0};
  std::vector<std::vector<uint32_t>> graph = {{1, 2, 3, 5}, {0, 4}, {0}, {0}, {1}, {0}};
  uint32_t medoid = 0;
  uint64_t dim = 2;
  uint32_t r = 4;
};

class TombstoneSearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<uint64_t> counter{0};
    index_path_ = std::filesystem::temp_directory_path() /
                  ("diskann_tomb_" + std::to_string(counter.fetch_add(1)) + ".index");
    n_ = scn_.graph.size();
    geom_ = DiskLayoutGeometry::compute(scn_.dim, scn_.r);
    write_disk_layout(index_path_.string(),
                      scn_.vecs.data(),
                      scn_.graph,
                      {n_, scn_.dim, scn_.r, scn_.medoid});
    cache_.generate(scn_.graph,
                    scn_.vecs.data(),
                    scn_.medoid,
                    n_,
                    scn_.dim,
                    scn_.r,
                    /*cache_ratio=*/0.0);
    reader_ = make_aligned_file_reader();
    reader_->open(index_path_.string());
    reader_->register_thread();
    td_.ctx_ = reader_->get_ctx();
    td_.alloc_scratch(/*n_page_slots=*/8, geom_.page_size, /*pq_table_entries=*/0);
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

  std::vector<std::pair<uint32_t, float>> search(const float *q,
                                                 uint32_t top_k,
                                                 const TombstoneBitmap *tomb,
                                                 bool deterministic,
                                                 uint32_t l = 50) {
    SearchContext ctx;
    ctx.reader = reader_.get();
    ctx.geom = &geom_;
    ctx.cache = &cache_;
    ctx.pq = nullptr;
    ctx.medoid = scn_.medoid;
    ctx.num_points = n_;
    ctx.tombstone = tomb;
    SearchParams p;
    p.search_list_size = l;
    p.beam_width = 4;
    p.use_pq = false;
    p.rerank = false;
    p.deterministic = deterministic;
    SearchStats stats;
    return disk_greedy_search(ctx, q, top_k, p, td_, &stats);
  }

  static bool contains(const std::vector<std::pair<uint32_t, float>> &res, uint32_t id) {
    return std::any_of(res.begin(), res.end(), [id](const auto &p) {
      return p.first == id;
    });
  }

  Scenario scn_;
  uint64_t n_ = 0;
  DiskLayoutGeometry geom_;
  NodeCache cache_;
  std::filesystem::path index_path_;
  std::unique_ptr<AlignedFileReader> reader_;
  ThreadData td_;
};

const std::vector<float> kQuery = {2.0f, 0.0f};

TEST_F(TombstoneSearchTest, NullTombstoneFindsHiddenNeighborUnchanged) {
  for (bool det : {true, false}) {
    const auto res = search(kQuery.data(), 1, /*tomb=*/nullptr, det);
    ASSERT_EQ(res.size(), 1u);
    EXPECT_EQ(res[0].first, 4u) << "deterministic=" << det;
  }
}

// IP-DiskANN: deleting the bridge node makes node 4 unreachable.
// Graph repair is done at delete time (by DiskANNIndex::remove), not at search time.
TEST_F(TombstoneSearchTest, DeletedBridgeMakesNodeUnreachable) {
  TombstoneBitmap tomb;
  tomb.set(1);
  for (bool det : {true, false}) {
    const auto res = search(kQuery.data(), 1, &tomb, det);
    ASSERT_EQ(res.size(), 1u);
    EXPECT_FALSE(contains(res, 4u)) << "node 4 unreachable; deterministic=" << det;
    EXPECT_FALSE(contains(res, 1u));
  }
}

TEST_F(TombstoneSearchTest, DeletedNearestNeighborIsNotReturned) {
  TombstoneBitmap tomb;
  tomb.set(4);
  for (bool det : {true, false}) {
    const auto res = search(kQuery.data(), 3, &tomb, det);
    EXPECT_FALSE(contains(res, 4u)) << "deleted NN must not appear; deterministic=" << det;
    ASSERT_FALSE(res.empty());
    EXPECT_EQ(res[0].first, 1u);
  }
}

TEST_F(TombstoneSearchTest, ResultCountReflectsLiveNodesOnly) {
  TombstoneBitmap tomb;
  for (uint32_t id : {1u, 2u, 3u, 4u, 5u}) {
    tomb.set(id);
  }
  const auto res = search(kQuery.data(), /*top_k=*/3, &tomb, /*deterministic=*/true);
  ASSERT_EQ(res.size(), 1u) << "only the medoid (node 0) is live";
  EXPECT_EQ(res[0].first, 0u);
}

}  // namespace
