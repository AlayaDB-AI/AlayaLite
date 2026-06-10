// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/graph/diskann/disk_page_io.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/diskann/diskann_index.hpp"

namespace {

#if defined(__linux__)

using alaya::diskann::DiskANNBuildParams;
using alaya::diskann::DiskANNIndex;
using alaya::diskann::DiskLayoutGeometry;
using alaya::diskann::DiskPageIO;

std::vector<float> make_vectors(uint64_t n, uint64_t dim, uint32_t seed = 123) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n * dim);
  for (auto &x : v) {
    x = dist(rng);
  }
  return v;
}

std::vector<uint64_t> make_labels(uint64_t n) {
  std::vector<uint64_t> labels(n);
  for (uint64_t i = 0; i < n; ++i) {
    labels[i] = 1000 + i;
  }
  return labels;
}

class DiskPageIOTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<uint64_t> counter{0};
    dir_ = std::filesystem::temp_directory_path() /
           ("diskann_pageio_" + std::to_string(counter.fetch_add(1)));
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }

  void build(uint64_t n, uint64_t dim, uint32_t r) {
    n_ = n;
    dim_ = dim;
    r_ = r;
    v_ = make_vectors(n, dim);
    labels_ = make_labels(n);
    DiskANNBuildParams bp;
    bp.R = r;
    bp.pq_n_chunks = 0;  // No-PQ (updates are No-PQ only)
    DiskANNIndex::build(dir_.string(), v_.data(), labels_.data(), n, dim, bp);
    geom_ = DiskLayoutGeometry::compute(dim, r);
  }

  std::string index_path() const { return (dir_ / "diskann.index").string(); }

  std::filesystem::path dir_;
  std::vector<float> v_;
  std::vector<uint64_t> labels_;
  uint64_t n_ = 0;
  uint64_t dim_ = 0;
  uint32_t r_ = 0;
  DiskLayoutGeometry geom_;
};

TEST_F(DiskPageIOTest, ReadNodeMatchesBuild) {
  build(200, 32, 32);
  DiskPageIO io(index_path(), geom_);
  for (uint32_t id : {0u, 1u, 50u, 199u}) {
    const auto nd = io.read_node(id);
    ASSERT_EQ(nd.coords.size(), dim_);
    for (uint64_t d = 0; d < dim_; ++d) {
      EXPECT_FLOAT_EQ(nd.coords[d], v_[id * dim_ + d]) << "id=" << id << " d=" << d;
    }
    EXPECT_LE(nd.nbrs.size(), r_);
    for (const auto nb : nd.nbrs) {
      EXPECT_LT(nb, n_);  // build neighbors are in range
    }
  }
}

TEST_F(DiskPageIOTest, WriteNodeRoundTrip) {
  build(200, 32, 32);
  DiskPageIO io(index_path(), geom_);
  std::vector<float> coords(dim_);
  for (uint64_t d = 0; d < dim_; ++d) {
    coords[d] = 0.25f + 0.01f * static_cast<float>(d);
  }
  const std::vector<uint32_t> nbrs = {3, 7, 11, 42, 100};
  io.write_node(5, coords.data(), static_cast<uint32_t>(nbrs.size()), nbrs.data());

  const auto nd = io.read_node(5);
  ASSERT_EQ(nd.coords.size(), dim_);
  for (uint64_t d = 0; d < dim_; ++d) {
    EXPECT_FLOAT_EQ(nd.coords[d], coords[d]) << "d=" << d;
  }
  ASSERT_EQ(nd.nbrs.size(), nbrs.size());
  for (size_t i = 0; i < nbrs.size(); ++i) {
    EXPECT_EQ(nd.nbrs[i], nbrs[i]) << "i=" << i;
  }
}

TEST_F(DiskPageIOTest, WriteNodeReuseDoesNotDisturbCoResidentNodes) {
  build(200, 32, 32);
  DiskPageIO io(index_path(), geom_);
  // Pick a node id that shares its page with neighbors; snapshot a co-resident.
  const uint32_t target = 16;
  const uint32_t sibling = target + 1;  // same sector page for nps>=2
  ASSERT_EQ(geom_.get_page_offset(target), geom_.get_page_offset(sibling));
  const auto sibling_before = io.read_node(sibling);

  std::vector<float> coords(dim_, 0.5f);
  const std::vector<uint32_t> nbrs = {1, 2};
  io.write_node(target, coords.data(), static_cast<uint32_t>(nbrs.size()), nbrs.data());

  const auto sibling_after = io.read_node(sibling);
  EXPECT_EQ(sibling_before.coords, sibling_after.coords);
  EXPECT_EQ(sibling_before.nbrs, sibling_after.nbrs);
}

TEST_F(DiskPageIOTest, WriteNodeNeighborsPreservesCoords) {
  build(200, 32, 32);
  DiskPageIO io(index_path(), geom_);
  const auto before = io.read_node(10);
  const std::vector<uint32_t> new_nbrs = {1, 2, 3};
  io.write_node_neighbors(10, static_cast<uint32_t>(new_nbrs.size()), new_nbrs.data());

  const auto after = io.read_node(10);
  ASSERT_EQ(after.coords.size(), before.coords.size());
  for (uint64_t d = 0; d < dim_; ++d) {
    EXPECT_FLOAT_EQ(after.coords[d], before.coords[d]) << "d=" << d;
  }
  ASSERT_EQ(after.nbrs.size(), new_nbrs.size());
  for (size_t i = 0; i < new_nbrs.size(); ++i) {
    EXPECT_EQ(after.nbrs[i], new_nbrs[i]) << "i=" << i;
  }
}

TEST_F(DiskPageIOTest, WriteNodeExtendsFileOnAppend) {
  build(200, 32, 32);
  DiskPageIO io(index_path(), geom_);
  const uint64_t before_size = io.file_size();
  // First id whose page does not yet exist (forces ftruncate extension).
  const uint32_t append_id = static_cast<uint32_t>(geom_.nodes_per_sector * geom_.num_pages(n_));
  ASSERT_GE(geom_.get_page_offset(append_id) + geom_.page_size, before_size);

  std::vector<float> coords(dim_, 1.0f);
  const std::vector<uint32_t> nbrs = {0, 1};
  io.write_node(append_id, coords.data(), static_cast<uint32_t>(nbrs.size()), nbrs.data());
  EXPECT_GT(io.file_size(), before_size);  // file was extended

  const auto nd = io.read_node(append_id);
  for (uint64_t d = 0; d < dim_; ++d) {
    EXPECT_FLOAT_EQ(nd.coords[d], coords[d]) << "d=" << d;
  }
  ASSERT_EQ(nd.nbrs.size(), nbrs.size());
}

TEST_F(DiskPageIOTest, CoordsCacheReturnsConsistentVectors) {
  build(100, 16, 16);
  DiskPageIO io(index_path(), geom_);
  const std::vector<float> first = io.read_coords_cached(7);  // value copy
  const auto &cached = io.read_coords_cached(7);              // cache hit
  EXPECT_EQ(first, cached);
  ASSERT_EQ(cached.size(), dim_);
  for (uint64_t d = 0; d < dim_; ++d) {
    EXPECT_FLOAT_EQ(cached[d], v_[7 * dim_ + d]) << "d=" << d;
  }
}

#else  // !__linux__

TEST(DiskPageIOTest, SkippedOnNonLinux) {
  GTEST_SKIP() << "DiskPageIO in-place updates require Linux O_DIRECT";
}

#endif  // __linux__

}  // namespace
