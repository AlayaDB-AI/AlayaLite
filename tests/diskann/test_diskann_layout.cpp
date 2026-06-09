// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/graph/diskann/disk_layout.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

using alaya::diskann::DiskLayoutGeometry;
using alaya::diskann::DiskLayoutHeader;
using alaya::diskann::kSectorLen;
using alaya::diskann::read_disk_layout_header;
using alaya::diskann::write_disk_layout;
using alaya::diskann::WriteDiskLayoutParams;

// Generate deterministic row-major vectors.
std::vector<float> make_vectors(uint64_t n, uint64_t dim, uint32_t seed = 7) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n * dim);
  for (auto &x : v) {
    x = dist(rng);
  }
  return v;
}

// A simple ring graph so every node has a deterministic, in-range neighbor set.
std::vector<std::vector<uint32_t>> make_ring_graph(uint64_t n, uint32_t degree) {
  std::vector<std::vector<uint32_t>> g(n);
  for (uint64_t i = 0; i < n; ++i) {
    for (uint32_t d = 1; d <= degree && d < n; ++d) {
      g[i].push_back(static_cast<uint32_t>((i + d) % n));
    }
  }
  return g;
}

class DiskLayoutTest : public ::testing::Test {
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
             ("diskann_layout_" + tag + "_" + std::to_string(counter.fetch_add(1)) + ".index");
    owned_.push_back(p);
    return p;
  }

  std::vector<std::filesystem::path> owned_;
};

// --- Geometry math ----------------------------------------------------------

TEST_F(DiskLayoutTest, GeometryMultiNodePerSector) {
  const auto g = DiskLayoutGeometry::compute(/*dim=*/128, /*max_degree=*/64);
  EXPECT_EQ(g.node_len, 128u * 4 + 4 + 64u * 4);  // 772
  EXPECT_EQ(g.node_len, 772u);
  EXPECT_EQ(g.nodes_per_sector, 5u);  // floor(4096 / 772)
  EXPECT_EQ(g.page_size, 4096u);      // ceil(5*772 / 4096) * 4096
}

TEST_F(DiskLayoutTest, GeometryMultiSectorPerNode) {
  const auto g = DiskLayoutGeometry::compute(/*dim=*/1024, /*max_degree=*/64);
  EXPECT_EQ(g.node_len, 1024u * 4 + 4 + 64u * 4);  // 4356 > 4096
  EXPECT_EQ(g.node_len, 4356u);
  EXPECT_EQ(g.nodes_per_sector, 1u);  // max(1, 4096/4356)
  EXPECT_EQ(g.page_size, 8192u);      // ceil(4356 / 4096) * 4096
}

TEST_F(DiskLayoutTest, GeometryRejectsZeroDim) {
  EXPECT_THROW(DiskLayoutGeometry::compute(0, 64), std::invalid_argument);
}

// --- Offset calculation -----------------------------------------------------

TEST_F(DiskLayoutTest, OffsetFirstNodeAtSectorOne) {
  const auto g = DiskLayoutGeometry::compute(128, 64);
  EXPECT_EQ(g.file_offset(0), kSectorLen);  // immediately after header
  EXPECT_EQ(g.get_page_offset(0), kSectorLen);
  EXPECT_EQ(g.offset_to_node(0), 0u);
}

TEST_F(DiskLayoutTest, OffsetCrossPageBoundaryMultiNode) {
  const auto g = DiskLayoutGeometry::compute(128, 64);  // nodes_per_sector = 5
  // First node of the second page is node id == nodes_per_sector.
  const uint64_t first_of_page2 = g.nodes_per_sector;
  EXPECT_EQ(g.get_page_offset(first_of_page2), kSectorLen + g.page_size);
  EXPECT_EQ(g.offset_to_node(first_of_page2), 0u);
  EXPECT_EQ(g.file_offset(first_of_page2), kSectorLen + g.page_size);
  // Last node of the first page packs at slot (nodes_per_sector - 1).
  EXPECT_EQ(g.file_offset(g.nodes_per_sector - 1),
            kSectorLen + (g.nodes_per_sector - 1) * g.node_len);
}

TEST_F(DiskLayoutTest, OffsetCrossPageBoundaryMultiSector) {
  const auto g = DiskLayoutGeometry::compute(1024, 64);  // nodes_per_sector = 1, page = 8192
  EXPECT_EQ(g.file_offset(0), kSectorLen);
  EXPECT_EQ(g.file_offset(1), kSectorLen + g.page_size);
  EXPECT_EQ(g.file_offset(2), kSectorLen + 2 * g.page_size);
}

TEST_F(DiskLayoutTest, AllPageOffsetsAre512Aligned) {
  for (uint64_t dim : {2u, 16u, 100u, 128u, 200u, 960u, 1024u}) {
    for (uint32_t r : {8u, 32u, 64u}) {
      const auto g = DiskLayoutGeometry::compute(dim, r);
      EXPECT_EQ(g.page_size % 512, 0u) << "dim=" << dim << " r=" << r;
      // The page that holds any node id begins at a 512-aligned file offset.
      for (uint64_t id : {0u, 1u, 4u, 5u, 9u, 10u, 100u}) {
        EXPECT_EQ(g.get_page_offset(id) % 512, 0u) << "dim=" << dim << " r=" << r << " id=" << id;
      }
    }
  }
}

// --- Header roundtrip -------------------------------------------------------

TEST_F(DiskLayoutTest, HeaderRoundtripMultiNode) {
  const uint64_t n = 23, dim = 128;
  const uint32_t r = 64, medoid = 7;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_ring_graph(n, r);
  const auto path = temp_path("hdr_mn");

  write_disk_layout(path.string(), vecs.data(), graph, {n, dim, r, medoid});
  const DiskLayoutHeader h = read_disk_layout_header(path.string());

  EXPECT_EQ(h.num_points, n);
  EXPECT_EQ(h.dim, dim);
  EXPECT_EQ(h.medoid, medoid);
  EXPECT_EQ(h.max_degree, r);
  EXPECT_EQ(h.node_len, 772u);
  EXPECT_EQ(h.nodes_per_sector, 5u);

  const auto g = DiskLayoutGeometry::compute(dim, r);
  EXPECT_EQ(h.total_file_size, g.total_file_size(n));
}

TEST_F(DiskLayoutTest, HeaderRoundtripMultiSector) {
  const uint64_t n = 4, dim = 1024;
  const uint32_t r = 64, medoid = 2;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_ring_graph(n, r);
  const auto path = temp_path("hdr_ms");

  write_disk_layout(path.string(), vecs.data(), graph, {n, dim, r, medoid});
  const DiskLayoutHeader h = read_disk_layout_header(path.string());

  EXPECT_EQ(h.num_points, n);
  EXPECT_EQ(h.dim, dim);
  EXPECT_EQ(h.medoid, medoid);
  EXPECT_EQ(h.node_len, 4356u);
  EXPECT_EQ(h.nodes_per_sector, 1u);
}

TEST_F(DiskLayoutTest, TotalFileSizeMatchesActualOnDisk) {
  const uint64_t n = 17, dim = 64;
  const uint32_t r = 32;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_ring_graph(n, r);
  const auto path = temp_path("size");

  write_disk_layout(path.string(), vecs.data(), graph, {n, dim, r, 0});
  const DiskLayoutHeader h = read_disk_layout_header(path.string());

  const auto actual = std::filesystem::file_size(path);
  EXPECT_EQ(static_cast<uint64_t>(actual), h.total_file_size);
  EXPECT_EQ(actual % kSectorLen, 0u);  // whole file is sector-aligned
}

// --- Node data roundtrip (validates the actual packing) ---------------------

TEST_F(DiskLayoutTest, NodeDataRoundtripMultiNode) {
  const uint64_t n = 13, dim = 16;
  const uint32_t r = 8, medoid = 3;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_ring_graph(n, r);
  const auto path = temp_path("node_mn");

  write_disk_layout(path.string(), vecs.data(), graph, {n, dim, r, medoid});
  const auto g = DiskLayoutGeometry::compute(dim, r);

  std::ifstream in(path.string(), std::ios::binary);
  ASSERT_TRUE(in.good());
  std::vector<char> rec(g.node_len);
  const uint64_t coords_bytes = dim * sizeof(float);

  for (uint64_t id = 0; id < n; ++id) {
    in.seekg(static_cast<std::streamoff>(g.file_offset(id)));
    in.read(rec.data(), static_cast<std::streamsize>(g.node_len));
    ASSERT_EQ(in.gcount(), static_cast<std::streamsize>(g.node_len));

    // coords
    std::vector<float> coords(dim);
    std::memcpy(coords.data(), rec.data(), coords_bytes);
    for (uint64_t d = 0; d < dim; ++d) {
      EXPECT_FLOAT_EQ(coords[d], vecs[id * dim + d]) << "id=" << id << " d=" << d;
    }

    // n_nbrs
    uint32_t n_nbrs = 0;
    std::memcpy(&n_nbrs, rec.data() + coords_bytes, sizeof(n_nbrs));
    ASSERT_EQ(n_nbrs, graph[id].size()) << "id=" << id;

    // neighbor ids
    std::vector<uint32_t> nbrs(n_nbrs);
    std::memcpy(nbrs.data(), rec.data() + coords_bytes + sizeof(uint32_t),
                n_nbrs * sizeof(uint32_t));
    for (uint32_t k = 0; k < n_nbrs; ++k) {
      EXPECT_EQ(nbrs[k], graph[id][k]) << "id=" << id << " k=" << k;
    }
  }
}

TEST_F(DiskLayoutTest, NodeDataRoundtripMultiSector) {
  const uint64_t n = 5, dim = 1024;
  const uint32_t r = 64;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_ring_graph(n, r);
  const auto path = temp_path("node_ms");

  write_disk_layout(path.string(), vecs.data(), graph, {n, dim, r, 0});
  const auto g = DiskLayoutGeometry::compute(dim, r);

  std::ifstream in(path.string(), std::ios::binary);
  ASSERT_TRUE(in.good());
  std::vector<char> rec(g.node_len);
  const uint64_t coords_bytes = dim * sizeof(float);

  for (uint64_t id = 0; id < n; ++id) {
    in.seekg(static_cast<std::streamoff>(g.file_offset(id)));
    in.read(rec.data(), static_cast<std::streamsize>(g.node_len));
    ASSERT_EQ(in.gcount(), static_cast<std::streamsize>(g.node_len));
    std::vector<float> coords(dim);
    std::memcpy(coords.data(), rec.data(), coords_bytes);
    for (uint64_t d = 0; d < dim; ++d) {
      EXPECT_FLOAT_EQ(coords[d], vecs[id * dim + d]) << "id=" << id << " d=" << d;
    }
    uint32_t n_nbrs = 0;
    std::memcpy(&n_nbrs, rec.data() + coords_bytes, sizeof(n_nbrs));
    EXPECT_EQ(n_nbrs, graph[id].size()) << "id=" << id;
  }
}

// --- Fail-loudly edge cases -------------------------------------------------

TEST_F(DiskLayoutTest, WriteRejectsGraphSizeMismatch) {
  const uint64_t n = 10, dim = 8;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_ring_graph(n - 1, 4);  // wrong size
  const auto path = temp_path("bad_graph");
  EXPECT_THROW(write_disk_layout(path.string(), vecs.data(), graph, {n, dim, 4, 0}),
               std::invalid_argument);
}

TEST_F(DiskLayoutTest, WriteRejectsMedoidOutOfRange) {
  const uint64_t n = 10, dim = 8;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_ring_graph(n, 4);
  const auto path = temp_path("bad_medoid");
  EXPECT_THROW(write_disk_layout(path.string(), vecs.data(), graph, {n, dim, 4, /*medoid=*/n}),
               std::invalid_argument);
}

TEST_F(DiskLayoutTest, WriteRejectsDegreeOverMax) {
  const uint64_t n = 10, dim = 8;
  const uint32_t max_degree = 4;
  const auto vecs = make_vectors(n, dim);
  auto graph = make_ring_graph(n, max_degree);
  graph[0].push_back(5);  // node 0 now has max_degree + 1 neighbors
  const auto path = temp_path("over_degree");
  EXPECT_THROW(write_disk_layout(path.string(), vecs.data(), graph, {n, dim, max_degree, 0}),
               std::invalid_argument);
}

TEST_F(DiskLayoutTest, ReadRejectsTruncatedHeader) {
  const auto path = temp_path("truncated");
  {
    std::ofstream out(path.string(), std::ios::binary);
    const char tiny[16] = {0};
    out.write(tiny, sizeof(tiny));  // far smaller than a sector
  }
  EXPECT_THROW(read_disk_layout_header(path.string()), std::runtime_error);
}

TEST_F(DiskLayoutTest, ReadRejectsSizeMismatch) {
  const uint64_t n = 12, dim = 16;
  const uint32_t r = 8;
  const auto vecs = make_vectors(n, dim);
  const auto graph = make_ring_graph(n, r);
  const auto path = temp_path("appended");
  write_disk_layout(path.string(), vecs.data(), graph, {n, dim, r, 0});
  // Corrupt the file by appending bytes so on-disk size != header total_file_size.
  {
    std::ofstream out(path.string(), std::ios::binary | std::ios::app);
    std::vector<char> extra(4096, 1);
    out.write(extra.data(), static_cast<std::streamsize>(extra.size()));
  }
  EXPECT_THROW(read_disk_layout_header(path.string()), std::runtime_error);
}

}  // namespace
