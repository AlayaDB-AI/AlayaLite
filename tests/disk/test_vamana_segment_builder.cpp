// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "index/disk/vamana_segment_builder.hpp"
#include "index/graph/vamana/vamana_reader.hpp"
#include "utils/metric_type.hpp"

namespace alaya::disk {
namespace {

class VamanaSegmentBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    tmp_root_ = std::filesystem::temp_directory_path() /
                ("alaya_vamana_builder_" + std::to_string(::getpid()) + "_" + test_name);
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

  static auto read_floats(const std::filesystem::path &path) -> std::vector<float> {
    std::ifstream in(path, std::ios::binary);
    const auto bytes = std::filesystem::file_size(path);
    std::vector<float> out(bytes / sizeof(float));
    in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(bytes));
    return out;
  }

  static auto graph_start(const std::filesystem::path &path) -> uint32_t {
    std::ifstream in(path, std::ios::binary);
    uint64_t expected_size = 0;
    uint32_t max_degree = 0;
    uint32_t start = 0;
    in.read(reinterpret_cast<char *>(&expected_size), sizeof(expected_size));
    in.read(reinterpret_cast<char *>(&max_degree), sizeof(max_degree));
    in.read(reinterpret_cast<char *>(&start), sizeof(start));
    return start;
  }

  static auto default_params() -> VamanaSegmentBuildParams {
    VamanaSegmentBuildParams params;
    params.R = 16;
    params.L = 64;
    params.num_threads = 1;
    return params;
  }

  std::filesystem::path tmp_root_;
  std::filesystem::path seg_parent_;
};

TEST_F(VamanaSegmentBuilderTest, vamana_segment_builder_writes_expected_files) {
  constexpr uint32_t kDim = 16;
  constexpr uint64_t kN = 256;
  auto vectors = make_vectors(kN, kDim, 1);
  auto ids = labels(kN, 5000);
  const auto seg_dir = seg_parent_ / "seg_00000001";

  VamanaSegmentBuilder builder(kDim, MetricType::L2, default_params());
  builder.add_batch(vectors.data(), ids.data(), kN);
  auto manifest = builder.finish(seg_dir);

  EXPECT_EQ(manifest.index_type, DiskIndexType::Vamana);
  EXPECT_EQ(manifest.metric, MetricType::L2);
  EXPECT_EQ(manifest.dim, kDim);
  EXPECT_EQ(manifest.count, kN);
  EXPECT_EQ(manifest.ids_file, "ids.u64.bin");
  EXPECT_EQ(manifest.vectors_file, "vectors.f32.bin");

  const std::set<std::string> expected{"graph.index", "ids.u64.bin", "manifest.txt",
                                       "vectors.f32.bin"};
  std::set<std::string> actual;
  for (const auto &entry : std::filesystem::directory_iterator(seg_dir)) {
    actual.insert(entry.path().filename().string());
  }
  EXPECT_EQ(actual, expected);

  auto loaded = SegmentManifest::load(seg_dir / "manifest.txt");
  EXPECT_EQ(loaded.index_type, DiskIndexType::Vamana);
  EXPECT_EQ(loaded.x_extras.at("x_graph_file"), "graph.index");
  EXPECT_EQ(std::filesystem::file_size(seg_dir / "ids.u64.bin"), kN * sizeof(uint64_t));
  EXPECT_EQ(std::filesystem::file_size(seg_dir / "vectors.f32.bin"),
            kN * kDim * sizeof(float));

  alaya::vamana::VamanaReader reader(seg_dir / "graph.index");
  EXPECT_EQ(reader.num_nodes(), kN);

  for (const auto &entry : std::filesystem::directory_iterator(seg_parent_)) {
    EXPECT_FALSE(entry.path().filename().string().starts_with(".tmp_seg_"));
  }
}

TEST_F(VamanaSegmentBuilderTest, add_batch_no_pointer_retention) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 128;
  auto vectors = make_vectors(kN, kDim, 7);
  const auto original = vectors;
  auto ids = labels(kN, 9000);
  const auto seg_dir = seg_parent_ / "seg_00000002";

  VamanaSegmentBuilder builder(kDim, MetricType::L2, default_params());
  builder.add_batch(vectors.data(), ids.data(), kN);
  std::fill(vectors.begin(), vectors.end(), 0.0F);
  builder.finish(seg_dir);

  EXPECT_EQ(read_floats(seg_dir / "vectors.f32.bin"), original);
  alaya::vamana::VamanaReader reader(seg_dir / "graph.index");
  EXPECT_EQ(reader.num_nodes(), kN);
}

TEST_F(VamanaSegmentBuilderTest, finish_refuses_existing_seg_dir) {
  constexpr uint32_t kDim = 4;
  auto vectors = make_vectors(32, kDim, 8);
  auto ids = labels(32);
  const auto seg_dir = seg_parent_ / "seg_00000003";
  std::filesystem::create_directories(seg_dir);
  {
    std::ofstream marker(seg_dir / "marker.txt");
    marker << "keep";
  }

  VamanaSegmentBuilder builder(kDim, MetricType::L2, default_params());
  builder.add_batch(vectors.data(), ids.data(), ids.size());
  EXPECT_THROW(builder.finish(seg_dir), std::runtime_error);
  EXPECT_TRUE(std::filesystem::exists(seg_dir / "marker.txt"));
}

TEST_F(VamanaSegmentBuilderTest, finish_rejects_ip_metric) {
  constexpr uint32_t kDim = 4;
  auto vectors = make_vectors(32, kDim, 9);
  auto ids = labels(32);
  const auto seg_dir = seg_parent_ / "seg_00000004";

  VamanaSegmentBuilder builder(kDim, MetricType::IP, default_params());
  builder.add_batch(vectors.data(), ids.data(), ids.size());
  try {
    (void)builder.finish(seg_dir);
    FAIL() << "expected IP metric rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("ip"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not implemented in v1"), std::string::npos) << msg;
  }
  EXPECT_FALSE(std::filesystem::exists(seg_dir));
}

TEST_F(VamanaSegmentBuilderTest, finish_rejects_cos_metric) {
  constexpr uint32_t kDim = 4;
  auto vectors = make_vectors(32, kDim, 10);
  auto ids = labels(32);
  const auto seg_dir = seg_parent_ / "seg_00000005";

  VamanaSegmentBuilder builder(kDim, MetricType::COS, default_params());
  builder.add_batch(vectors.data(), ids.data(), ids.size());
  try {
    (void)builder.finish(seg_dir);
    FAIL() << "expected COS metric rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("cos"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not implemented in v1"), std::string::npos) << msg;
  }
  EXPECT_FALSE(std::filesystem::exists(seg_dir));
}

TEST_F(VamanaSegmentBuilderTest, manifest_records_build_params) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 128;
  auto vectors = make_vectors(kN, kDim, 11);
  auto ids = labels(kN);
  const auto seg_dir = seg_parent_ / "seg_00000006";

  VamanaSegmentBuildParams params;
  params.R = 32;
  params.L = 64;
  params.alpha = 1.1F;
  params.seed = 7;
  params.num_threads = 1;

  VamanaSegmentBuilder builder(kDim, MetricType::L2, params);
  builder.add_batch(vectors.data(), ids.data(), kN);
  builder.finish(seg_dir);

  auto manifest = SegmentManifest::load(seg_dir / "manifest.txt");
  EXPECT_EQ(manifest.x_extras.at("x_R"), "32");
  EXPECT_EQ(manifest.x_extras.at("x_L"), "64");
  EXPECT_NEAR(std::stof(manifest.x_extras.at("x_alpha")), 1.1F, 1e-6F);
  EXPECT_EQ(manifest.x_extras.at("x_seed"), "7");
  EXPECT_EQ(static_cast<uint32_t>(std::stoul(manifest.x_extras.at("x_medoid"))),
            graph_start(seg_dir / "graph.index"));
}

}  // namespace
}  // namespace alaya::disk
