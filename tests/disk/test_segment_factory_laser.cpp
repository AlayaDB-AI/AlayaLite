// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <unistd.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "index/disk/segment_factory.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "utils/metric_type.hpp"

#ifndef ALAYA_LASER_FIXTURE_DIR
  #define ALAYA_LASER_FIXTURE_DIR ""
#endif

#ifndef ALAYA_LASER_FIXTURE_PREFIX
  #define ALAYA_LASER_FIXTURE_PREFIX "dsqg_seg_00000001"
#endif

namespace alaya::disk {
namespace {

#if defined(ALAYA_ENABLE_LASER) && (ALAYA_ENABLE_LASER + 0) != 0 && defined(__linux__)
constexpr bool kLaserFactorySupported = true;
#else
constexpr bool kLaserFactorySupported = false;
#endif

constexpr uint32_t kLaserFixtureDim = 128;
constexpr uint64_t kLaserFixtureCount = 2048;
constexpr uint32_t kLaserFixtureR = 64;
constexpr uint32_t kLaserTopK = 10;

class SegmentFactoryLaserTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    tmp_root_ = std::filesystem::temp_directory_path() /
                ("alaya_factory_laser_" + std::to_string(::getpid()) + "_" + info->name());
    std::filesystem::remove_all(tmp_root_);
    seg_parent_ = tmp_root_ / "segments";
    src_dir_ = tmp_root_ / "src";
    std::filesystem::create_directories(seg_parent_);
    std::filesystem::create_directories(src_dir_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_root_, ec);
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

  static auto vectors(uint64_t n, uint32_t dim, uint32_t seed = 42) -> std::vector<float> {
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

  static auto is_nan_bits(float value) -> bool {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return (bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0;
  }

  static void expect_contains(const std::runtime_error &e, const std::string &needle) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find(needle), std::string::npos) << msg;
  }

  static void write_minimal_laser_manifest(const std::filesystem::path &seg_dir) {
    std::filesystem::create_directories(seg_dir);
    std::ofstream ofs(seg_dir / "manifest.txt");
    ofs << "version=1\n"
        << "segment_id=" << seg_dir.filename().string() << "\n"
        << "index_type=disk_laser\n"
        << "metric=L2\n"
        << "dim=128\n"
        << "count=1\n"
        << "ids_file=ids.u64.bin\n"
        << "vectors_file=\n";
  }

  static auto laser_fixture_dir() -> std::filesystem::path {
    return std::filesystem::path(ALAYA_LASER_FIXTURE_DIR);
  }

  static auto laser_fixture_prefix() -> std::string {
    return std::string(ALAYA_LASER_FIXTURE_PREFIX);
  }

  static auto laser_fixture_index_name() -> std::string {
    return laser_fixture_prefix() + "_R" + std::to_string(kLaserFixtureR) + "_MD" +
           std::to_string(kLaserFixtureDim) + ".index";
  }

  static auto laser_fixture_required_keys() -> std::vector<std::string> {
    return {
        "x_laser_index_file",
        "x_laser_rotator_file",
        "x_laser_cache_ids_file",
        "x_laser_cache_nodes_file",
    };
  }

  static auto laser_fixture_required_files() -> std::vector<std::string> {
    const auto index = laser_fixture_index_name();
    return {
        index,
        index + "_rotator",
        index + "_cache_ids",
        index + "_cache_nodes",
        laser_fixture_prefix() + "_input.fbin",
    };
  }

  static auto laser_fixture_has_required_files(const std::filesystem::path &dir) -> bool {
    if (dir.empty()) {
      return false;
    }

    const auto required = laser_fixture_required_files();
    return std::all_of(required.begin(), required.end(), [&](const auto &name) {
      std::error_code ec;
      const auto path = dir / name;
      if (!std::filesystem::is_regular_file(path, ec) || ec) {
        return false;
      }
      const auto size = std::filesystem::file_size(path, ec);
      return !ec && size > 0;
    });
  }

  static auto laser_fixture_skip_reason() -> std::string {
    if (!engine_supported_v1(DiskIndexType::Laser)) {
      return "disk_laser is not registered in this build";
    }
    const auto dir = laser_fixture_dir();
    if (!laser_fixture_has_required_files(dir)) {
      return "LASER fixture is missing or incomplete under " + dir.string();
    }
    return {};
  }

  static auto read_laser_fixture_query(uint64_t row = 0) -> std::vector<float> {
    const auto path = laser_fixture_dir() / (laser_fixture_prefix() + "_input.fbin");
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("failed to open fixture vectors: " + path.string());
    }

    int32_t count = 0;
    int32_t dim = 0;
    input.read(reinterpret_cast<char *>(&count), sizeof(count));
    input.read(reinterpret_cast<char *>(&dim), sizeof(dim));
    if (count != static_cast<int32_t>(kLaserFixtureCount) ||
        dim != static_cast<int32_t>(kLaserFixtureDim)) {
      throw std::runtime_error("unexpected fixture vector header in " + path.string());
    }
    if (row >= kLaserFixtureCount) {
      throw std::runtime_error("fixture query row out of range");
    }

    std::vector<float> vectors(static_cast<size_t>(count) * static_cast<size_t>(dim));
    input.read(reinterpret_cast<char *>(vectors.data()),
               static_cast<std::streamsize>(vectors.size() * sizeof(float)));
    if (!input) {
      throw std::runtime_error("short fixture vector read: " + path.string());
    }

    std::vector<float> query(kLaserFixtureDim);
    std::copy_n(vectors.data() + static_cast<size_t>(row) * kLaserFixtureDim,
                kLaserFixtureDim,
                query.data());
    return query;
  }

  auto import_fixture_segment(const std::filesystem::path &seg_dir,
                              const std::vector<uint64_t> &ids) const
      -> std::shared_ptr<SegmentSearcher> {
    return import_segment_from_artifacts(seg_dir,
                                         manifest(DiskIndexType::Laser, kLaserFixtureDim),
                                         laser_fixture_dir(),
                                         ids.data(),
                                         ids.size());
  }

  std::filesystem::path tmp_root_;
  std::filesystem::path seg_parent_;
  std::filesystem::path src_dir_;
};

TEST_F(SegmentFactoryLaserTest, engine_supported_matches_build_gate) {
  EXPECT_TRUE(engine_supported_v1(DiskIndexType::Flat));
  EXPECT_TRUE(engine_supported_v1(DiskIndexType::Vamana));
  EXPECT_EQ(engine_supported_v1(DiskIndexType::Laser), kLaserFactorySupported);
  static_assert(engine_supported_v1(DiskIndexType::Flat));
  static_assert(engine_supported_v1(DiskIndexType::Vamana));
#if defined(ALAYA_ENABLE_LASER) && (ALAYA_ENABLE_LASER + 0) != 0 && defined(__linux__)
  static_assert(engine_supported_v1(DiskIndexType::Laser));
#else
  static_assert(!engine_supported_v1(DiskIndexType::Laser));
#endif
}

TEST_F(SegmentFactoryLaserTest, create_from_pending_always_throws_with_import_pointer) {
  constexpr uint32_t kDim = 128;
  auto data = vectors(2, kDim);
  auto ids = labels(2);
  const auto seg_dir = seg_parent_ / "seg_00000001";

  try {
    (void)create_segment_from_pending(seg_dir,
                                      manifest(DiskIndexType::Laser, kDim),
                                      data.data(),
                                      ids.data(),
                                      ids.size());
    FAIL() << "expected Laser create rejection";
  } catch (const std::runtime_error &e) {
    expect_contains(e, "disk_laser");
    expect_contains(e, "not implemented in v1");
    expect_contains(e, "import_laser_segment");
  }
  EXPECT_FALSE(std::filesystem::exists(seg_dir));
}

TEST_F(SegmentFactoryLaserTest, import_rejects_non_laser_engine) {
  auto ids = labels(2);
  const auto seg_dir = seg_parent_ / "seg_00000001";

  try {
    (void)import_segment_from_artifacts(seg_dir,
                                        manifest(DiskIndexType::Flat, 128),
                                        src_dir_,
                                        ids.data(),
                                        ids.size());
    FAIL() << "expected Flat import rejection";
  } catch (const std::runtime_error &e) {
    expect_contains(e, "unsupported import path for engine 'disk_flat'");
  }
  EXPECT_FALSE(std::filesystem::exists(seg_dir));

  try {
    (void)import_segment_from_artifacts(seg_dir,
                                        manifest(DiskIndexType::Vamana, 128),
                                        src_dir_,
                                        ids.data(),
                                        ids.size());
    FAIL() << "expected Vamana import rejection";
  } catch (const std::runtime_error &e) {
    expect_contains(e, "unsupported import path for engine 'disk_vamana'");
  }
  EXPECT_FALSE(std::filesystem::exists(seg_dir));
}

TEST_F(SegmentFactoryLaserTest, unsupported_laser_load_and_import_throw_when_gate_closed) {
#if defined(ALAYA_ENABLE_LASER) && (ALAYA_ENABLE_LASER + 0) != 0 && defined(__linux__)
  GTEST_SKIP() << "covered by the supported Laser fixture-backed tests";
#else
  const auto seg_dir = seg_parent_ / "seg_00000001";
  write_minimal_laser_manifest(seg_dir);

  try {
    (void)load_segment_from_manifest(seg_dir);
    FAIL() << "expected unsupported Laser load rejection";
  } catch (const std::runtime_error &e) {
    expect_contains(e, "disk_laser");
    expect_contains(e, "not implemented in v1");
  }

  const auto imported_dir = seg_parent_ / "seg_00000002";
  auto ids = labels(2);
  try {
    (void)import_segment_from_artifacts(imported_dir,
                                        manifest(DiskIndexType::Laser, 128),
                                        src_dir_,
                                        ids.data(),
                                        ids.size());
    FAIL() << "expected unsupported Laser import rejection";
  } catch (const std::runtime_error &e) {
    expect_contains(e, "disk_laser");
    expect_contains(e, "not implemented in v1");
  }
  EXPECT_FALSE(std::filesystem::exists(imported_dir));
#endif
}

TEST_F(SegmentFactoryLaserTest, flat_unchanged) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 16;
  auto data = vectors(kN, kDim, 7);
  auto ids = labels(kN, 5000);
  const auto seg_dir = seg_parent_ / "seg_00000001";

  auto searcher = create_segment_from_pending(seg_dir,
                                              manifest(DiskIndexType::Flat, kDim),
                                              data.data(),
                                              ids.data(),
                                              ids.size());
  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->type(), DiskIndexType::Flat);
  EXPECT_EQ(searcher->size(), kN);
  EXPECT_TRUE(std::filesystem::exists(seg_dir / "manifest.txt"));
}

TEST_F(SegmentFactoryLaserTest, vamana_unchanged) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 128;
  auto data = vectors(kN, kDim, 11);
  auto ids = labels(kN, 7000);
  const auto seg_dir = seg_parent_ / "seg_00000001";

  auto created = create_segment_from_pending(seg_dir,
                                             manifest(DiskIndexType::Vamana, kDim),
                                             data.data(),
                                             ids.data(),
                                             ids.size());
  ASSERT_NE(created, nullptr);
  EXPECT_EQ(created->type(), DiskIndexType::Vamana);
  EXPECT_EQ(created->size(), kN);
  EXPECT_TRUE(std::filesystem::exists(seg_dir / "graph.index"));

  auto loaded = load_segment_from_manifest(seg_dir);
  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->type(), DiskIndexType::Vamana);
  EXPECT_EQ(loaded->size(), kN);
}

#if defined(ALAYA_ENABLE_LASER) && (ALAYA_ENABLE_LASER + 0) != 0 && defined(__linux__)
TEST_F(SegmentFactoryLaserTest, import_returns_searcher) {
  if (const auto reason = laser_fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }

  auto ids = labels(kLaserFixtureCount, 0);
  const auto seg_dir = seg_parent_ / "seg_00000001";

  auto searcher = import_fixture_segment(seg_dir, ids);

  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->type(), DiskIndexType::Laser);
  EXPECT_EQ(searcher->dim(), kLaserFixtureDim);
  EXPECT_EQ(searcher->size(), kLaserFixtureCount);

  const auto segment_manifest = SegmentManifest::load(seg_dir / "manifest.txt");
  EXPECT_EQ(segment_manifest.index_type, DiskIndexType::Laser);
  EXPECT_EQ(segment_manifest.count, kLaserFixtureCount);
  for (const auto &key : laser_fixture_required_keys()) {
    const auto &filename = segment_manifest.x_extras.at(key);
    EXPECT_TRUE(std::filesystem::is_regular_file(seg_dir / filename)) << key << "=" << filename;
  }
}

TEST_F(SegmentFactoryLaserTest, load_returns_searcher) {
  if (const auto reason = laser_fixture_skip_reason(); !reason.empty()) {
    GTEST_SKIP() << reason;
  }

  auto ids = labels(kLaserFixtureCount, 5000);
  const auto seg_dir = seg_parent_ / "seg_00000001";
  auto imported = import_fixture_segment(seg_dir, ids);
  auto loaded = load_segment_from_manifest(seg_dir);

  ASSERT_NE(imported, nullptr);
  ASSERT_NE(loaded, nullptr);
  EXPECT_EQ(loaded->type(), DiskIndexType::Laser);
  EXPECT_EQ(loaded->dim(), imported->dim());
  EXPECT_EQ(loaded->size(), imported->size());

  DiskSearchOptions opts;
  opts.top_k = kLaserTopK;
  opts.ef = 64;
  opts.beam_width = 4;
  const auto query = read_laser_fixture_query(31);
  const auto imported_hits = imported->search(query.data(), opts);
  const auto loaded_hits = loaded->search(query.data(), opts);

  ASSERT_EQ(imported_hits.size(), kLaserTopK);
  ASSERT_EQ(loaded_hits.size(), kLaserTopK);
  EXPECT_EQ(imported_hits.front().label, 5031u);
  EXPECT_EQ(loaded_hits.front().label, 5031u);
  for (const auto &hits : {imported_hits, loaded_hits}) {
    for (const auto &hit : hits) {
      EXPECT_GE(hit.label, 5000u);
      EXPECT_LT(hit.label, 5000u + kLaserFixtureCount);
      EXPECT_TRUE(is_nan_bits(hit.distance)) << hit.distance;
    }
  }
}
#endif

}  // namespace
}  // namespace alaya::disk
