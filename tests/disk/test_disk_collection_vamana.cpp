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
#include <algorithm>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "index/disk/disk_collection.hpp"
#include "index/disk/segment_factory.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "utils/metric_type.hpp"

namespace alaya::disk {
namespace {

class DiskCollectionVamanaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    tmp_root_ = std::filesystem::temp_directory_path() /
                ("alaya_collection_vamana_" + std::to_string(::getpid()) + "_" + test_name);
    std::filesystem::remove_all(tmp_root_);
    std::filesystem::create_directories(tmp_root_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_root_, ec);
  }

  static auto make_vectors(uint64_t n, uint32_t dim, uint32_t seed = 42) -> std::vector<float> {
    std::vector<float> out(n * dim);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto &v : out) {
      v = dist(rng);
    }
    return out;
  }

  static auto labels(uint64_t n, uint64_t base = 0) -> std::vector<uint64_t> {
    std::vector<uint64_t> out(n);
    std::iota(out.begin(), out.end(), base);
    return out;
  }

  std::filesystem::path tmp_root_;
};

TEST_F(DiskCollectionVamanaTest, constructor_accepts_vamana) {
  const auto path = tmp_root_ / "coll";
  DiskCollection col(path, 128, MetricType::L2, DiskIndexType::Vamana);
  auto manifest = CollectionManifest::load(path / "collection_manifest.txt");
  EXPECT_EQ(manifest.index_type, DiskIndexType::Vamana);
  EXPECT_EQ(manifest.dim, 128u);
  EXPECT_TRUE(std::filesystem::is_directory(path / "segments"));
}

TEST_F(DiskCollectionVamanaTest, flush_writes_vamana_segment) {
  constexpr uint32_t kDim = 16;
  constexpr uint64_t kN = 512;
  const auto path = tmp_root_ / "coll";
  auto vectors = make_vectors(kN, kDim, 1);
  auto ids = labels(kN, 1000);

  {
    DiskCollection col(path, kDim, MetricType::L2, DiskIndexType::Vamana);
    col.add_batch(vectors.data(), ids.data(), kN);
    col.flush();
    EXPECT_EQ(col.size(), kN);
  }

  EXPECT_TRUE(std::filesystem::exists(path / "segments" / "seg_00000001" / "graph.index"));
  auto seg = load_segment_from_manifest(path / "segments" / "seg_00000001");
  EXPECT_EQ(seg->type(), DiskIndexType::Vamana);

  auto reopened = DiskCollection::open(path);
  EXPECT_EQ(reopened.size(), kN);
  DiskSearchOptions opts;
  opts.top_k = 1;
  opts.ef = 64;
  auto hits = reopened.search(vectors.data(), opts);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].label, ids[0]);
  EXPECT_GE(hits[0].distance, 0.0F);
}

TEST_F(DiskCollectionVamanaTest, multi_vamana_segment_search) {
  constexpr uint32_t kDim = 16;
  const auto path = tmp_root_ / "coll";
  auto query = make_vectors(1, kDim, 99);
  std::vector<DiskSearchHit> hits;
  DiskSearchOptions opts;
  opts.top_k = 10;
  opts.ef = 64;
  {
    DiskCollection col(path, kDim, MetricType::L2, DiskIndexType::Vamana);

    auto v1 = make_vectors(256, kDim, 2);
    auto l1 = labels(256, 0);
    col.add_batch(v1.data(), l1.data(), l1.size());
    col.flush();

    auto v2 = make_vectors(256, kDim, 3);
    auto l2 = labels(256, 1000);
    col.add_batch(v2.data(), l2.data(), l2.size());
    col.flush();

    hits = col.search(query.data(), opts);
  }
  ASSERT_EQ(hits.size(), 10u);
  for (size_t i = 1; i < hits.size(); ++i) {
    EXPECT_LE(hits[i - 1].distance, hits[i].distance);
  }

  auto reopened = DiskCollection::open(path);
  auto hits2 = reopened.search(query.data(), opts);
  ASSERT_EQ(hits.size(), hits2.size());
  for (size_t i = 0; i < hits.size(); ++i) {
    EXPECT_EQ(hits[i].label, hits2[i].label);
    EXPECT_FLOAT_EQ(hits[i].distance, hits2[i].distance);
  }
}

TEST_F(DiskCollectionVamanaTest, duplicate_label_across_vamana_segments_throws) {
  constexpr uint32_t kDim = 8;
  const auto path = tmp_root_ / "coll";
  DiskCollection col(path, kDim, MetricType::L2, DiskIndexType::Vamana);
  auto v1 = make_vectors(64, kDim, 4);
  auto l1 = labels(64, 10);
  col.add_batch(v1.data(), l1.data(), l1.size());
  col.flush();

  auto v2 = make_vectors(4, kDim, 5);
  std::vector<uint64_t> l2{42, 1000, 1001, 1002};
  col.add_batch(v2.data(), l2.data(), l2.size());
  EXPECT_THROW(col.flush(), std::invalid_argument);
  EXPECT_FALSE(std::filesystem::exists(path / "segments" / "seg_00000002"));
}

TEST_F(DiskCollectionVamanaTest, singleton_flush_rejected_before_publish) {
  constexpr uint32_t kDim = 4;
  const auto path = tmp_root_ / "coll";
  VamanaSegmentBuildParams params;
  params.R = 1;
  params.L = 1;
  params.num_threads = 1;
  DiskCollection col(path,
                     kDim,
                     MetricType::L2,
                     DiskIndexType::Vamana,
                     DiskCollection::kDefaultMaxPendingBytes,
                     params);
  std::vector<float> vectors(kDim, 1.0F);
  std::vector<uint64_t> ids{123};
  col.add_batch(vectors.data(), ids.data(), ids.size());

  EXPECT_THROW(col.flush(), std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(path / "segments" / "seg_00000001"));
}

TEST_F(DiskCollectionVamanaTest, open_rejects_invalid_vamana_params) {
  const auto path = tmp_root_ / "coll";
  std::filesystem::create_directories(path / "segments");
  std::ofstream manifest(path / "collection_manifest.txt");
  manifest << "version=1\n"
           << "dim=4\n"
           << "metric=L2\n"
           << "index_type=disk_vamana\n"
           << "next_segment_id=1\n"
           << "x_vamana_R=0\n";
  manifest.close();

  EXPECT_THROW((void)DiskCollection::open(path), std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(path / "segments" / "seg_00000001"));
}

TEST_F(DiskCollectionVamanaTest, open_rejects_unsupported_vamana_metric) {
  const auto path = tmp_root_ / "coll";
  std::filesystem::create_directories(path / "segments");
  std::ofstream manifest(path / "collection_manifest.txt");
  manifest << "version=1\n"
           << "dim=4\n"
           << "metric=IP\n"
           << "index_type=disk_vamana\n"
           << "next_segment_id=1\n";
  manifest.close();

  EXPECT_THROW((void)DiskCollection::open(path), std::runtime_error);
}

TEST_F(DiskCollectionVamanaTest, max_pending_bytes_survives_reopen) {
  constexpr uint32_t kDim = 4;
  const auto path = tmp_root_ / "coll";
  { DiskCollection col(path, kDim, MetricType::L2, DiskIndexType::Vamana, 100); }

  auto reopened = DiskCollection::open(path);
  std::vector<float> vectors(3 * kDim, 0.0F);
  std::vector<uint64_t> ids{1, 2, 3};
  EXPECT_THROW(reopened.add_batch(vectors.data(), ids.data(), ids.size()), std::runtime_error);
}

TEST_F(DiskCollectionVamanaTest, open_classifies_vamana_orphans) {
  constexpr uint32_t kDim = 8;
  const auto path = tmp_root_ / "coll";
  auto vectors = make_vectors(64, kDim, 6);
  auto ids = labels(64);
  {
    DiskCollection col(path, kDim, MetricType::L2, DiskIndexType::Vamana);
    col.add_batch(vectors.data(), ids.data(), ids.size());
    col.flush();
  }

  const auto listed = path / "segments" / "seg_00000001";
  const auto orphan = path / "segments" / "seg_00000003";
  std::filesystem::copy(listed, orphan, std::filesystem::copy_options::recursive);
  EXPECT_NO_THROW((void)DiskCollection::open(path));

  std::filesystem::resize_file(orphan / "graph.index",
                               std::filesystem::file_size(orphan / "graph.index") - 1);
  EXPECT_NO_THROW((void)DiskCollection::open(path));

  std::filesystem::remove(orphan / "manifest.txt");
  auto reopened = DiskCollection::open(path);
  EXPECT_EQ(reopened.size(), ids.size());

  auto v2 = make_vectors(2, kDim, 7);
  std::vector<uint64_t> l2{9000, 9001};
  reopened.add_batch(v2.data(), l2.data(), l2.size());
  reopened.flush();
  EXPECT_TRUE(std::filesystem::exists(path / "segments" / "seg_00000004"));
}

TEST_F(DiskCollectionVamanaTest, laser_still_rejected) {
  const auto path = tmp_root_ / "coll";
  try {
    DiskCollection col(path, 128, MetricType::L2, DiskIndexType::Laser);
    (void)col;
    FAIL() << "expected Laser rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("disk_laser"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not implemented in v1"), std::string::npos) << msg;
  }
}

TEST_F(DiskCollectionVamanaTest, vamana_ip_flush_rejected_by_engine) {
  constexpr uint32_t kDim = 8;
  const auto path = tmp_root_ / "coll";
  DiskCollection col(path, kDim, MetricType::IP, DiskIndexType::Vamana);
  auto vectors = make_vectors(32, kDim, 8);
  auto ids = labels(32);
  col.add_batch(vectors.data(), ids.data(), ids.size());

  try {
    col.flush();
    FAIL() << "expected Vamana IP rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("ip"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not implemented in v1"), std::string::npos) << msg;
  }
  EXPECT_FALSE(std::filesystem::exists(path / "segments" / "seg_00000001"));
}

}  // namespace
}  // namespace alaya::disk
