// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "index/disk/disk_flat_builder.hpp"
#include "index/disk/disk_flat_searcher.hpp"
#include "index/disk/segment_factory.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "utils/metric_type.hpp"

namespace alaya::disk {

namespace {

class SegmentFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto pid_str = std::to_string(static_cast<long long>(::getpid()));
    auto base = std::filesystem::temp_directory_path() /
                ("alaya_seg_factory_" + pid_str + "_" +
                 ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::remove_all(base);
    std::filesystem::create_directories(base);
    tmp_root_ = base;
    seg_parent_ = tmp_root_ / "segments";
    std::filesystem::create_directories(seg_parent_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_root_, ec);
  }

  static auto make_random_vectors(uint64_t n, uint32_t dim, uint32_t seed = 42)
      -> std::vector<float> {
    std::vector<float> out(n * dim);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto &v : out) {
      v = dist(rng);
    }
    return out;
  }

  static auto sequential_labels(uint64_t n, uint64_t base = 1000) -> std::vector<uint64_t> {
    std::vector<uint64_t> out(n);
    std::iota(out.begin(), out.end(), base);
    return out;
  }

  static auto make_collection_manifest(DiskIndexType engine, uint64_t dim,
                                       MetricType metric = MetricType::L2)
      -> CollectionManifest {
    CollectionManifest m;
    m.version = kManifestVersion;
    m.dim = dim;
    m.metric = metric;
    m.index_type = engine;
    m.next_segment_id = 1;
    return m;
  }

  std::filesystem::path tmp_root_;
  std::filesystem::path seg_parent_;
};

TEST_F(SegmentFactoryTest, segment_factory_opens_flat_segment) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 32;
  const auto seg_dir = seg_parent_ / "seg_00000001";

  // Build a flat segment directly via the existing builder path.
  auto vectors = make_random_vectors(kN, kDim, 1);
  auto labels = sequential_labels(kN, 100);
  {
    DiskFlatBuilder b(kDim, MetricType::L2);
    b.add_batch(vectors.data(), labels.data(), kN);
    b.finish(seg_dir);
  }

  // Open it through the factory.
  auto factory_searcher = load_segment_from_manifest(seg_dir);
  ASSERT_NE(factory_searcher, nullptr);
  EXPECT_EQ(factory_searcher->type(), DiskIndexType::Flat);
  EXPECT_EQ(factory_searcher->size(), kN);
  EXPECT_EQ(factory_searcher->dim(), kDim);

  // Open the same dir directly via DiskFlatSegmentSearcher and confirm hits
  // are byte-for-byte identical.
  DiskFlatSegmentSearcher direct(seg_dir);
  auto query = make_random_vectors(1, kDim, 99);
  DiskSearchOptions opts;
  opts.top_k = 5;
  auto hits_factory = factory_searcher->search(query.data(), opts);
  auto hits_direct = direct.search(query.data(), opts);
  ASSERT_EQ(hits_factory.size(), hits_direct.size());
  for (size_t i = 0; i < hits_factory.size(); ++i) {
    EXPECT_EQ(hits_factory[i].label, hits_direct[i].label);
    EXPECT_EQ(hits_factory[i].distance, hits_direct[i].distance);
  }
}

TEST_F(SegmentFactoryTest, segment_factory_creates_flat_segment_from_pending) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 16;
  const auto seg_dir = seg_parent_ / "seg_00000001";

  auto vectors = make_random_vectors(kN, kDim, 7);
  auto labels = sequential_labels(kN, 5000);
  auto col_manifest = make_collection_manifest(DiskIndexType::Flat, kDim);

  auto created = create_segment_from_pending(seg_dir, col_manifest, vectors.data(),
                                             labels.data(), kN);
  ASSERT_NE(created, nullptr);
  EXPECT_EQ(created->type(), DiskIndexType::Flat);
  EXPECT_EQ(created->size(), kN);
  EXPECT_EQ(created->dim(), kDim);

  // Reopen via factory and compare hits.
  auto reopened = load_segment_from_manifest(seg_dir);
  auto query = make_random_vectors(1, kDim, 13);
  DiskSearchOptions opts;
  opts.top_k = 4;
  auto hits_created = created->search(query.data(), opts);
  auto hits_reopened = reopened->search(query.data(), opts);
  ASSERT_EQ(hits_created.size(), hits_reopened.size());
  for (size_t i = 0; i < hits_created.size(); ++i) {
    EXPECT_EQ(hits_created[i].label, hits_reopened[i].label);
    EXPECT_EQ(hits_created[i].distance, hits_reopened[i].distance);
  }
}

TEST_F(SegmentFactoryTest, segment_factory_engine_supported_v1) {
  EXPECT_TRUE(engine_supported_v1(DiskIndexType::Flat));
  EXPECT_TRUE(engine_supported_v1(DiskIndexType::Vamana));
  EXPECT_FALSE(engine_supported_v1(DiskIndexType::Laser));
  // constexpr-ness check: usable in a constant expression.
  static_assert(engine_supported_v1(DiskIndexType::Flat));
  static_assert(engine_supported_v1(DiskIndexType::Vamana));
  static_assert(!engine_supported_v1(DiskIndexType::Laser));
}

TEST_F(SegmentFactoryTest, segment_factory_create_with_vamana_succeeds) {
  constexpr uint32_t kDim = 4;
  constexpr uint64_t kN = 32;
  const auto seg_dir = seg_parent_ / "seg_00000001";

  auto vectors = make_random_vectors(kN, kDim);
  auto labels = sequential_labels(kN);
  auto col_manifest = make_collection_manifest(DiskIndexType::Vamana, kDim);

  auto searcher = create_segment_from_pending(seg_dir, col_manifest, vectors.data(),
                                              labels.data(), kN);
  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->type(), DiskIndexType::Vamana);
  EXPECT_TRUE(std::filesystem::exists(seg_dir / "graph.index"));
}

TEST_F(SegmentFactoryTest, segment_factory_create_with_laser_throws_clear_error) {
  constexpr uint32_t kDim = 4;
  constexpr uint64_t kN = 2;
  const auto seg_dir = seg_parent_ / "seg_00000001";

  auto vectors = make_random_vectors(kN, kDim);
  auto labels = sequential_labels(kN);
  auto col_manifest = make_collection_manifest(DiskIndexType::Laser, kDim);

  try {
    (void)create_segment_from_pending(seg_dir, col_manifest, vectors.data(), labels.data(), kN);
    FAIL() << "expected throw on Laser create";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("disk_laser"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not implemented in v1"), std::string::npos) << msg;
  }
  EXPECT_FALSE(std::filesystem::exists(seg_dir));
}

namespace {

// Hand-crafts a minimally-valid SegmentManifest with `index_type` overridden
// to whatever the caller requests. The `count`-sized ids/vectors files are
// also created so the manifest is parseable; the test only cares that the
// factory rejects the engine before opening anything.
void hand_craft_segment_with_index_type(const std::filesystem::path &seg_dir,
                                        const std::string &engine_str) {
  std::filesystem::create_directories(seg_dir);
  // Tiny but valid ids file: count=1, dim=4, one row.
  const uint64_t one_id = 42;
  std::vector<float> one_vec(4, 1.0F);
  {
    std::ofstream ofs(seg_dir / "ids.u64.bin", std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(&one_id), sizeof(one_id));
  }
  {
    std::ofstream ofs(seg_dir / "vectors.f32.bin", std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(one_vec.data()),
              static_cast<std::streamsize>(one_vec.size() * sizeof(float)));
  }
  std::ofstream ofs(seg_dir / "manifest.txt");
  ofs << "version=1\n"
      << "segment_id=" << seg_dir.filename().string() << "\n"
      << "index_type=" << engine_str << "\n"
      << "metric=L2\n"
      << "dim=4\n"
      << "count=1\n"
      << "ids_file=ids.u64.bin\n"
      << "vectors_file=vectors.f32.bin\n";
}

}  // namespace

TEST_F(SegmentFactoryTest, segment_factory_load_with_vamana_manifest_requires_graph) {
  const auto seg_dir = seg_parent_ / "seg_00000001";
  hand_craft_segment_with_index_type(seg_dir, "disk_vamana");
  try {
    (void)load_segment_from_manifest(seg_dir);
    FAIL() << "expected malformed Vamana segment load to throw";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("x_graph_file"), std::string::npos) << msg;
  }
}

TEST_F(SegmentFactoryTest, segment_factory_load_with_laser_manifest_throws) {
  const auto seg_dir = seg_parent_ / "seg_00000001";
  hand_craft_segment_with_index_type(seg_dir, "disk_laser");
  try {
    (void)load_segment_from_manifest(seg_dir);
    FAIL() << "expected throw on disk_laser manifest load";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("disk_laser"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not implemented in v1"), std::string::npos) << msg;
  }
}

namespace {

// Mirror of the StubSearcher in disk_types_test — re-asserted here so that any
// future PR that quietly extends SegmentSearcher to satisfy a factory need
// (e.g. adding labels()) breaks this test as well as disk-types.
class StubSearcher : public SegmentSearcher {
 public:
  ~StubSearcher() override = default;
  auto search(const float * /*query*/, const DiskSearchOptions & /*opts*/) const
      -> std::vector<DiskSearchHit> override {
    return {};
  }
  auto size() const -> uint64_t override { return 0; }
  auto dim() const -> uint32_t override { return 0; }
  auto type() const -> DiskIndexType override { return DiskIndexType::Flat; }
};

}  // namespace

TEST_F(SegmentFactoryTest, segment_factory_does_not_add_virtuals_to_searcher) {
  static_assert(std::is_abstract_v<SegmentSearcher>,
                "SegmentSearcher must remain abstract");
  static_assert(std::has_virtual_destructor_v<SegmentSearcher>,
                "SegmentSearcher must keep a virtual destructor");
  // The StubSearcher implements EXACTLY the four documented overrides; if a
  // new pure virtual were added to SegmentSearcher, this stub would no longer
  // be instantiable and the line below would fail to compile.
  StubSearcher s;
  EXPECT_EQ(s.size(), 0u);
  EXPECT_EQ(s.dim(), 0u);
  EXPECT_EQ(s.type(), DiskIndexType::Flat);
}

}  // namespace
}  // namespace alaya::disk
