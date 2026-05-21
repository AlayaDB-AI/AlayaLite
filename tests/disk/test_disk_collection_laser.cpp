// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "index/disk/disk_collection.hpp"
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

constexpr uint64_t kLaserFixtureCount = 2048;
constexpr uint32_t kLaserFixtureDim = 128;
constexpr uint32_t kLaserTopK = 10;
constexpr uint32_t kLaserEf = 64;
constexpr uint32_t kLaserBeamWidth = 4;

class DiskCollectionLaserTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    tmp_root_ = std::filesystem::temp_directory_path() /
                ("alaya_disk_collection_laser_" + std::to_string(::getpid()) + "_" +
                 info->test_suite_name() + "_" + info->name());
    std::filesystem::remove_all(tmp_root_);
    std::filesystem::create_directories(tmp_root_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_root_, ec);
  }

  static auto contains_all(std::string_view msg, std::initializer_list<std::string_view> needles)
      -> bool {
    for (auto needle : needles) {
      if (msg.find(needle) == std::string_view::npos) {
        return false;
      }
    }
    return true;
  }

  static void write_collection_manifest(const std::filesystem::path &coll,
                                        std::string_view index_type) {
    std::filesystem::create_directories(coll / "segments");
    std::ofstream ofs(coll / "collection_manifest.txt");
    ofs << "version=1\n"
        << "dim=128\n"
        << "metric=L2\n"
        << "index_type=" << index_type << "\n"
        << "next_segment_id=1\n";
  }

  static void write_nonempty_file(const std::filesystem::path &path) {
    std::ofstream ofs(path, std::ios::binary);
    const char byte = '\1';
    ofs.write(&byte, 1);
  }

  static auto fixture_dir() -> std::filesystem::path {
    return std::filesystem::path(ALAYA_LASER_FIXTURE_DIR);
  }

  static auto fixture_prefix() -> std::string { return std::string(ALAYA_LASER_FIXTURE_PREFIX); }

  static auto fixture_has_required_files(const std::filesystem::path &dir,
                                         const std::string &prefix) -> bool {
    if (dir.empty()) {
      return false;
    }
    const auto index = dir / (prefix + "_R64_MD128.index");
    const std::vector<std::filesystem::path> required{
        dir / (prefix + "_input.fbin"),
        index,
        std::filesystem::path(index.string() + "_rotator"),
        std::filesystem::path(index.string() + "_cache_ids"),
        std::filesystem::path(index.string() + "_cache_nodes"),
    };
    std::error_code ec;
    return std::all_of(required.begin(), required.end(), [&](const auto &path) {
      const bool ok = std::filesystem::is_regular_file(path, ec) && !ec &&
                      std::filesystem::file_size(path, ec) > 0 && !ec;
      ec.clear();
      return ok;
    });
  }

  static void require_fixture_available() {
    const auto dir = fixture_dir();
    const auto prefix = fixture_prefix();
    if (!fixture_has_required_files(dir, prefix)) {
      GTEST_SKIP() << "LASER fixture is missing or incomplete under " << dir;
    }
  }

  static auto identity_labels(uint64_t n = kLaserFixtureCount, uint64_t base = 0)
      -> std::vector<uint64_t> {
    std::vector<uint64_t> labels(n);
    std::iota(labels.begin(), labels.end(), base);
    return labels;
  }

  static auto transformed_labels(uint64_t n, uint64_t multiplier, uint64_t addend)
      -> std::vector<uint64_t> {
    std::vector<uint64_t> labels(n);
    for (uint64_t i = 0; i < n; ++i) {
      labels[i] = i * multiplier + addend;
    }
    return labels;
  }

  static auto read_fixture_vectors(const std::filesystem::path &dir, const std::string &prefix)
      -> std::vector<float> {
    const auto input_path = dir / (prefix + "_input.fbin");
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("failed to open LASER fixture vectors: " + input_path.string());
    }
    int32_t count = 0;
    int32_t dim = 0;
    input.read(reinterpret_cast<char *>(&count), sizeof(count));
    input.read(reinterpret_cast<char *>(&dim), sizeof(dim));
    if (count != static_cast<int32_t>(kLaserFixtureCount) ||
        dim != static_cast<int32_t>(kLaserFixtureDim)) {
      throw std::runtime_error("unexpected LASER fixture vector header in " + input_path.string());
    }
    std::vector<float> vectors(static_cast<size_t>(count) * dim);
    input.read(reinterpret_cast<char *>(vectors.data()),
               static_cast<std::streamsize>(vectors.size() * sizeof(float)));
    if (!input) {
      throw std::runtime_error("short LASER fixture vector read: " + input_path.string());
    }
    return vectors;
  }

  static auto query_row(const std::vector<float> &vectors, uint64_t row) -> const float * {
    return vectors.data() + static_cast<size_t>(row) * kLaserFixtureDim;
  }

  static auto search_options() -> DiskSearchOptions {
    DiskSearchOptions opts;
    opts.top_k = kLaserTopK;
    opts.ef = kLaserEf;
    opts.beam_width = kLaserBeamWidth;
    return opts;
  }

  static auto is_nan_bits(float value) -> bool {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(value));
    return (bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0;
  }

  static void expect_all_nan_distances(const std::vector<DiskSearchHit> &hits) {
    for (const auto &hit : hits) {
      EXPECT_TRUE(is_nan_bits(hit.distance)) << "label=" << hit.label;
    }
  }

  static void expect_labels_in_range(const std::vector<DiskSearchHit> &hits, uint64_t upper_bound) {
    for (const auto &hit : hits) {
      EXPECT_LT(hit.label, upper_bound) << "label=" << hit.label;
    }
  }

  static void expect_contains_label(const std::vector<DiskSearchHit> &hits, uint64_t label) {
    EXPECT_NE(std::find_if(hits.begin(),
                           hits.end(),
                           [&](const auto &hit) { return hit.label == label; }),
              hits.end());
  }

  static auto prepare_second_fixture_dir(const std::filesystem::path &dst_dir)
      -> std::filesystem::path {
    const auto src_dir = fixture_dir();
    const auto src_prefix = fixture_prefix();
    const std::string dst_prefix = "dsqg_seg_00000002";
    std::filesystem::create_directories(dst_dir);

    const std::vector<std::string> required_suffixes{
        "_R64_MD128.index",
        "_R64_MD128.index_rotator",
        "_R64_MD128.index_cache_ids",
        "_R64_MD128.index_cache_nodes",
    };
    const std::vector<std::string> optional_suffixes{
        "_medoids",
        "_medoids_indices",
        "_pca.bin",
    };

    for (const auto &suffix : required_suffixes) {
      std::filesystem::copy_file(src_dir / (src_prefix + suffix), dst_dir / (dst_prefix + suffix));
    }
    for (const auto &suffix : optional_suffixes) {
      const auto src = src_dir / (src_prefix + suffix);
      std::error_code ec;
      if (std::filesystem::exists(src, ec) && !ec) {
        std::filesystem::copy_file(src, dst_dir / (dst_prefix + suffix));
      }
    }
    return dst_dir;
  }

  static auto capture_open_logs(const std::filesystem::path &coll, uint64_t *opened_size = nullptr)
      -> std::string {
    std::ostringstream log_stream;
    auto previous_logger = spdlog::default_logger();
    auto logger = std::make_shared<spdlog::logger>("disk_collection_laser_test",
                                                   std::make_shared<spdlog::sinks::ostream_sink_mt>(
                                                       log_stream));
    logger->set_pattern("%v");
    spdlog::set_default_logger(logger);
    try {
      auto opened = DiskCollection::open(coll);
      if (opened_size != nullptr) {
        *opened_size = opened.size();
      }
      spdlog::set_default_logger(previous_logger);
    } catch (...) {
      spdlog::set_default_logger(previous_logger);
      throw;
    }
    return log_stream.str();
  }

  std::filesystem::path tmp_root_;
};

TEST_F(DiskCollectionLaserTest, import_on_non_laser_collection_throws) {
  const auto coll = tmp_root_ / "coll";
  DiskCollection col(coll, 128, MetricType::L2, DiskIndexType::Flat);
  std::vector<uint64_t> labels{42};

  try {
    col.import_laser_segment(tmp_root_ / "unused_src", labels.data(), labels.size());
    FAIL() << "expected import_laser_segment to reject a non-Laser collection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains_all(msg, {"import_laser_segment requires a disk_laser collection"}))
        << msg;
  }
  EXPECT_FALSE(std::filesystem::exists(coll / "segments" / "seg_00000001"));
}

#if !defined(ALAYA_ENABLE_LASER) || ALAYA_ENABLE_LASER == 0 || !defined(__linux__)

TEST_F(DiskCollectionLaserTest, laser_unsupported_when_disabled) {
  const auto coll = tmp_root_ / "coll";
  try {
    (void)DiskCollection(coll, 128, MetricType::L2, DiskIndexType::Laser);
    FAIL() << "expected disk_laser constructor rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains_all(msg, {"disk_laser", "not implemented in v1"})) << msg;
  }

  const auto existing = tmp_root_ / "existing";
  write_collection_manifest(existing, "disk_laser");
  try {
    (void)DiskCollection::open(existing);
    FAIL() << "expected disk_laser open rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains_all(msg, {"disk_laser", "not implemented in v1"})) << msg;
  }
}

#else

TEST_F(DiskCollectionLaserTest, constructor_accepts_laser_when_enabled) {
  const auto coll = tmp_root_ / "coll";
  DiskCollection col(coll, 128, MetricType::L2, DiskIndexType::Laser);
  auto manifest = CollectionManifest::load(coll / "collection_manifest.txt");
  EXPECT_EQ(manifest.index_type, DiskIndexType::Laser);
  EXPECT_EQ(col.size(), 0u);
}

TEST_F(DiskCollectionLaserTest, add_batch_rejects_laser) {
  const auto coll = tmp_root_ / "coll";
  DiskCollection col(coll, 128, MetricType::L2, DiskIndexType::Laser);
  std::vector<float> vectors(128, 0.0F);
  std::vector<uint64_t> labels{7};

  try {
    col.add_batch(vectors.data(), labels.data(), labels.size());
    FAIL() << "expected disk_laser add_batch rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains_all(msg, {"disk_laser", "not implemented in v1", "import_laser_segment"}))
        << msg;
  }

  col.flush();
  EXPECT_EQ(col.size(), 0u);
  EXPECT_FALSE(std::filesystem::exists(coll / "segments" / "seg_00000001"));
}

TEST_F(DiskCollectionLaserTest, add_batch_rejects_laser_even_when_n_is_zero) {
  const auto coll = tmp_root_ / "coll";
  DiskCollection col(coll, 128, MetricType::L2, DiskIndexType::Laser);

  try {
    col.add_batch(nullptr, nullptr, 0);
    FAIL() << "expected disk_laser add_batch rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains_all(msg, {"disk_laser", "not implemented in v1", "import_laser_segment"}))
        << msg;
  }

  col.flush();
  EXPECT_FALSE(std::filesystem::exists(coll / "segments" / "seg_00000001"));
}

TEST_F(DiskCollectionLaserTest, flush_on_empty_laser_is_noop) {
  const auto coll = tmp_root_ / "coll";
  DiskCollection col(coll, 128, MetricType::L2, DiskIndexType::Laser);
  const auto before = CollectionManifest::load(coll / "collection_manifest.txt");

  EXPECT_NO_THROW(col.flush());

  const auto after = CollectionManifest::load(coll / "collection_manifest.txt");
  EXPECT_EQ(after.segment_ids.size(), before.segment_ids.size());
  EXPECT_EQ(after.next_segment_id, before.next_segment_id);
  EXPECT_FALSE(std::filesystem::exists(coll / "segments" / "seg_00000001"));
}

TEST_F(DiskCollectionLaserTest, import_laser_segment_writes_segment) {
  require_fixture_available();
  const auto coll = tmp_root_ / "coll";
  const auto vectors = read_fixture_vectors(fixture_dir(), fixture_prefix());
  auto labels = identity_labels();

  {
    DiskCollection col(coll, kLaserFixtureDim, MetricType::L2, DiskIndexType::Laser);
    col.import_laser_segment(fixture_dir(), labels.data(), labels.size());
    EXPECT_EQ(col.size(), kLaserFixtureCount);
  }

  const auto manifest = CollectionManifest::load(coll / "collection_manifest.txt");
  ASSERT_EQ(manifest.segment_ids.size(), 1u);
  EXPECT_EQ(manifest.segment_ids[0], "seg_00000001");
  EXPECT_EQ(manifest.next_segment_id, 2u);

  const auto seg_dir = coll / "segments" / "seg_00000001";
  auto searcher = load_segment_from_manifest(seg_dir);
  ASSERT_NE(searcher, nullptr);
  EXPECT_EQ(searcher->type(), DiskIndexType::Laser);

  auto reopened = DiskCollection::open(coll);
  EXPECT_EQ(reopened.size(), kLaserFixtureCount);

  const auto *query = query_row(vectors, 0);
  const auto segment_hits = searcher->search(query, search_options());
  const auto hits = reopened.search(query, search_options());
  ASSERT_EQ(hits.size(), kLaserTopK);
  ASSERT_EQ(segment_hits.size(), kLaserTopK);
  expect_labels_in_range(hits, kLaserFixtureCount);
  expect_labels_in_range(segment_hits, kLaserFixtureCount);
  EXPECT_EQ(hits.front().label, 0u);
  EXPECT_EQ(segment_hits.front().label, 0u);
  EXPECT_TRUE(is_nan_bits(hits.front().distance));
}

TEST_F(DiskCollectionLaserTest, multi_laser_segment_search) {
  require_fixture_available();
  const auto coll = tmp_root_ / "coll";
  const auto second_fixture = prepare_second_fixture_dir(tmp_root_ / "fixture_seg_00000002");
  const auto vectors = read_fixture_vectors(fixture_dir(), fixture_prefix());
  auto labels1 = transformed_labels(kLaserFixtureCount, 2, 1);
  auto labels2 = transformed_labels(kLaserFixtureCount, 2, 0);

  {
    DiskCollection col(coll, kLaserFixtureDim, MetricType::L2, DiskIndexType::Laser);
    col.import_laser_segment(fixture_dir(), labels1.data(), labels1.size());

    const auto first_segment_hits = col.search(query_row(vectors, 0), search_options());
    ASSERT_EQ(first_segment_hits.size(), kLaserTopK);
    expect_all_nan_distances(first_segment_hits);

    col.import_laser_segment(second_fixture, labels2.data(), labels2.size());
    EXPECT_EQ(col.size(), kLaserFixtureCount * 2);

    auto first_segment = load_segment_from_manifest(coll / "segments" / "seg_00000001");
    auto second_segment = load_segment_from_manifest(coll / "segments" / "seg_00000002");
    ASSERT_NE(first_segment, nullptr);
    ASSERT_NE(second_segment, nullptr);
    const auto first_raw_hits = first_segment->search(query_row(vectors, 0), search_options());
    const auto second_raw_hits = second_segment->search(query_row(vectors, 0), search_options());
    ASSERT_EQ(first_raw_hits.size(), kLaserTopK);
    ASSERT_EQ(second_raw_hits.size(), kLaserTopK);
    EXPECT_EQ(first_segment_hits.front().label, 1u);
    EXPECT_EQ(first_raw_hits.front().label, 1u);
    EXPECT_EQ(second_raw_hits.front().label, 0u);

    const auto hits = col.search(query_row(vectors, 0), search_options());
    ASSERT_EQ(hits.size(), kLaserTopK);
    expect_all_nan_distances(hits);
    expect_labels_in_range(hits, kLaserFixtureCount * 2);
    expect_contains_label(hits, 0u);
    expect_contains_label(hits, 1u);
  }

  const auto manifest = CollectionManifest::load(coll / "collection_manifest.txt");
  ASSERT_EQ(manifest.segment_ids.size(), 2u);
  EXPECT_EQ(manifest.segment_ids[0], "seg_00000001");
  EXPECT_EQ(manifest.segment_ids[1], "seg_00000002");
  EXPECT_EQ(manifest.next_segment_id, 3u);

  auto reopened = DiskCollection::open(coll);
  EXPECT_EQ(reopened.size(), kLaserFixtureCount * 2);
  const auto reopened_hits = reopened.search(query_row(vectors, 0), search_options());
  ASSERT_EQ(reopened_hits.size(), kLaserTopK);
  expect_all_nan_distances(reopened_hits);
  expect_labels_in_range(reopened_hits, kLaserFixtureCount * 2);
  expect_contains_label(reopened_hits, 0u);
  expect_contains_label(reopened_hits, 1u);
}

TEST_F(DiskCollectionLaserTest, duplicate_label_across_laser_segments_throws) {
  require_fixture_available();
  const auto coll = tmp_root_ / "coll";
  const auto second_fixture = prepare_second_fixture_dir(tmp_root_ / "fixture_seg_00000002");
  auto labels1 = identity_labels();
  auto labels2 = identity_labels(kLaserFixtureCount, 100000);
  labels2[17] = 42;

  DiskCollection col(coll, kLaserFixtureDim, MetricType::L2, DiskIndexType::Laser);
  col.import_laser_segment(fixture_dir(), labels1.data(), labels1.size());
  EXPECT_EQ(col.size(), kLaserFixtureCount);

  try {
    col.import_laser_segment(second_fixture, labels2.data(), labels2.size());
    FAIL() << "expected duplicate cross-segment label rejection";
  } catch (const std::invalid_argument &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains_all(msg, {"duplicate label across segments", "42"})) << msg;
  }

  EXPECT_EQ(col.size(), kLaserFixtureCount);
  EXPECT_FALSE(std::filesystem::exists(coll / "segments" / "seg_00000002"));
  const auto manifest = CollectionManifest::load(coll / "collection_manifest.txt");
  ASSERT_EQ(manifest.segment_ids.size(), 1u);
  EXPECT_EQ(manifest.segment_ids[0], "seg_00000001");
  EXPECT_EQ(manifest.next_segment_id, 2u);
}

TEST_F(DiskCollectionLaserTest, open_classifies_laser_orphans) {
  require_fixture_available();
  const auto coll = tmp_root_ / "coll";
  auto labels = identity_labels();

  {
    DiskCollection col(coll, kLaserFixtureDim, MetricType::L2, DiskIndexType::Laser);
    col.import_laser_segment(fixture_dir(), labels.data(), labels.size());
    EXPECT_EQ(col.size(), kLaserFixtureCount);
  }

  auto collection_manifest = CollectionManifest::load(coll / "collection_manifest.txt");
  ASSERT_EQ(collection_manifest.segment_ids.size(), 1u);
  collection_manifest.segment_ids.clear();
  collection_manifest.save(coll / "collection_manifest.txt");

  const auto orphan = coll / "segments" / "seg_00000001";
  uint64_t opened_size = kLaserFixtureCount;
  auto logs = capture_open_logs(coll, &opened_size);
  EXPECT_EQ(opened_size, 0u);
  EXPECT_TRUE(contains_all(logs, {"seg_00000001", "kind=complete"})) << logs;

  const auto segment_manifest = SegmentManifest::load(orphan / "manifest.txt");
  ASSERT_TRUE(std::filesystem::remove(orphan / segment_manifest.x_extras.at("x_laser_index_file")));
  opened_size = kLaserFixtureCount;
  logs = capture_open_logs(coll, &opened_size);
  EXPECT_EQ(opened_size, 0u);
  EXPECT_TRUE(contains_all(logs, {"seg_00000001", "kind=truncated"})) << logs;

  ASSERT_TRUE(std::filesystem::remove(orphan / "manifest.txt"));
  opened_size = kLaserFixtureCount;
  logs = capture_open_logs(coll, &opened_size);
  EXPECT_EQ(opened_size, 0u);
  EXPECT_TRUE(contains_all(logs, {"seg_00000001", "kind=partial"})) << logs;
}

TEST_F(DiskCollectionLaserTest, orphan_classification_tolerates_empty_vectors_file) {
  const auto coll = tmp_root_ / "coll";
  write_collection_manifest(coll, "disk_laser");

  const auto orphan = coll / "segments" / "seg_00000001";
  std::filesystem::create_directories(orphan);

  SegmentManifest sm;
  sm.segment_id = "seg_00000001";
  sm.index_type = DiskIndexType::Laser;
  sm.metric = MetricType::L2;
  sm.dim = 128;
  sm.count = 1;
  sm.ids_file = "ids.u64.bin";
  sm.vectors_file = "";
  sm.x_extras["x_laser_index_file"] = "laser.index";
  sm.x_extras["x_laser_rotator_file"] = "laser.rotator";
  sm.x_extras["x_laser_cache_ids_file"] = "laser.cache_ids";
  sm.x_extras["x_laser_cache_nodes_file"] = "laser.cache_nodes";
  sm.save(orphan / "manifest.txt");

  {
    const uint64_t label = 123;
    std::ofstream ids(orphan / sm.ids_file, std::ios::binary);
    ids.write(reinterpret_cast<const char *>(&label), sizeof(label));
  }
  for (const auto &[key, value] : sm.x_extras) {
    (void)key;
    write_nonempty_file(orphan / value);
  }

  uint64_t opened_size = 1;
  const std::string logs = capture_open_logs(coll, &opened_size);

  EXPECT_EQ(opened_size, 0u);
  EXPECT_NE(logs.find("kind=complete"), std::string::npos) << logs;
  EXPECT_EQ(logs.find("vectors stat failed"), std::string::npos) << logs;
}

#endif

}  // namespace
}  // namespace alaya::disk
