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

#include <gtest/gtest.h>
#include <unistd.h>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "index/disk/disk_flat_builder.hpp"
#include "index/disk/segment_factory.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "utils/metric_type.hpp"

namespace alaya::disk {
namespace {

class SegmentFactoryVamanaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    tmp_root_ = std::filesystem::temp_directory_path() /
                ("alaya_factory_vamana_" + std::to_string(::getpid()) + "_" + test_name);
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

  static auto manifest(DiskIndexType type, uint64_t dim, MetricType metric = MetricType::L2)
      -> CollectionManifest {
    CollectionManifest m;
    m.version = kManifestVersion;
    m.dim = dim;
    m.metric = metric;
    m.index_type = type;
    m.next_segment_id = 1;
    return m;
  }

  static void write_laser_manifest(const std::filesystem::path &seg_dir) {
    std::filesystem::create_directories(seg_dir);
    std::ofstream ofs(seg_dir / "manifest.txt");
    ofs << "version=1\n"
        << "segment_id=" << seg_dir.filename().string() << "\n"
        << "index_type=disk_laser\n"
        << "metric=L2\n"
        << "dim=4\n"
        << "count=1\n"
        << "ids_file=ids.u64.bin\n"
        << "vectors_file=vectors.f32.bin\n";
  }

  std::filesystem::path tmp_root_;
  std::filesystem::path seg_parent_;
};

TEST_F(SegmentFactoryVamanaTest, engine_supported_v1_now_true) {
  EXPECT_TRUE(engine_supported_v1(DiskIndexType::Flat));
  EXPECT_TRUE(engine_supported_v1(DiskIndexType::Vamana));
  EXPECT_FALSE(engine_supported_v1(DiskIndexType::Laser));
  static_assert(engine_supported_v1(DiskIndexType::Flat));
  static_assert(engine_supported_v1(DiskIndexType::Vamana));
  static_assert(!engine_supported_v1(DiskIndexType::Laser));
}

TEST_F(SegmentFactoryVamanaTest, create_returns_vamana_searcher) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 128;
  auto vectors = make_vectors(kN, kDim, 1);
  auto ids = labels(kN);
  const auto seg_dir = seg_parent_ / "seg_00000001";

  auto searcher = create_segment_from_pending(seg_dir, manifest(DiskIndexType::Vamana, kDim),
                                              vectors.data(), ids.data(), kN);
  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->type(), DiskIndexType::Vamana);
  EXPECT_EQ(searcher->size(), kN);
  EXPECT_TRUE(std::filesystem::exists(seg_dir / "manifest.txt"));
  EXPECT_TRUE(std::filesystem::exists(seg_dir / "ids.u64.bin"));
  EXPECT_TRUE(std::filesystem::exists(seg_dir / "vectors.f32.bin"));
  EXPECT_TRUE(std::filesystem::exists(seg_dir / "graph.index"));
}

TEST_F(SegmentFactoryVamanaTest, load_returns_vamana_searcher) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 128;
  auto vectors = make_vectors(kN, kDim, 2);
  auto ids = labels(kN, 5000);
  const auto seg_dir = seg_parent_ / "seg_00000001";

  auto created = create_segment_from_pending(seg_dir, manifest(DiskIndexType::Vamana, kDim),
                                             vectors.data(), ids.data(), kN);
  auto loaded = load_segment_from_manifest(seg_dir);
  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->type(), DiskIndexType::Vamana);
  EXPECT_EQ(loaded->size(), kN);

  DiskSearchOptions opts;
  opts.top_k = 5;
  opts.ef = 64;
  auto h1 = created->search(vectors.data(), opts);
  auto h2 = loaded->search(vectors.data(), opts);
  ASSERT_EQ(h1.size(), h2.size());
  for (size_t i = 0; i < h1.size(); ++i) {
    EXPECT_EQ(h1[i].label, h2[i].label);
    EXPECT_FLOAT_EQ(h1[i].distance, h2[i].distance);
  }
}

TEST_F(SegmentFactoryVamanaTest, flat_unchanged) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 32;
  auto vectors = make_vectors(kN, kDim, 3);
  auto ids = labels(kN, 7000);
  const auto seg_dir = seg_parent_ / "seg_00000001";

  auto searcher = create_segment_from_pending(seg_dir, manifest(DiskIndexType::Flat, kDim),
                                              vectors.data(), ids.data(), kN);
  EXPECT_EQ(searcher->type(), DiskIndexType::Flat);
  EXPECT_TRUE(std::filesystem::exists(seg_dir / "manifest.txt"));
  EXPECT_FALSE(std::filesystem::exists(seg_dir / "graph.index"));
}

TEST_F(SegmentFactoryVamanaTest, laser_still_unsupported) {
  constexpr uint32_t kDim = 4;
  auto vectors = make_vectors(4, kDim, 4);
  auto ids = labels(4);
  const auto seg_dir = seg_parent_ / "seg_00000001";

  try {
    (void)create_segment_from_pending(seg_dir, manifest(DiskIndexType::Laser, kDim),
                                      vectors.data(), ids.data(), ids.size());
    FAIL() << "expected Laser create rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("disk_laser"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not implemented in v1"), std::string::npos) << msg;
  }
  EXPECT_FALSE(std::filesystem::exists(seg_dir));

  write_laser_manifest(seg_dir);
  try {
    (void)load_segment_from_manifest(seg_dir);
    FAIL() << "expected Laser load rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("disk_laser"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not implemented in v1"), std::string::npos) << msg;
  }
}

TEST_F(SegmentFactoryVamanaTest, vamana_ip_throws_through_engine) {
  constexpr uint32_t kDim = 4;
  auto vectors = make_vectors(32, kDim, 5);
  auto ids = labels(32);
  const auto seg_dir = seg_parent_ / "seg_00000001";

  try {
    (void)create_segment_from_pending(seg_dir,
                                      manifest(DiskIndexType::Vamana, kDim, MetricType::IP),
                                      vectors.data(), ids.data(), ids.size());
    FAIL() << "expected Vamana IP rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("ip"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not implemented in v1"), std::string::npos) << msg;
  }
  EXPECT_FALSE(std::filesystem::exists(seg_dir));
  for (const auto &entry : std::filesystem::directory_iterator(seg_parent_)) {
    EXPECT_FALSE(entry.path().filename().string().starts_with(".tmp_seg_"));
  }
}

}  // namespace
}  // namespace alaya::disk
