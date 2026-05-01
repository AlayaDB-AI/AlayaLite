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

#include "index/disk/laser_segment_importer.hpp"
#include "index/disk/segment_factory.hpp"
#include "index/disk/segment_manifest.hpp"
#include "utils/metric_type.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iterator>
#include <numeric>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace alaya::disk {
namespace {

#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0

constexpr uint32_t kDim = 128;
constexpr uint64_t kCount = 4;

auto contains(std::string_view haystack, std::string_view needle) -> bool {
  return haystack.find(needle) != std::string_view::npos;
}

auto prefix_for(std::string_view seg_basename) -> std::string {
  return "dsqg_" + std::string(seg_basename);
}

auto index_filename(std::string_view seg_basename, const LaserSegmentImportParams &params)
    -> std::string {
  const auto main_dim = params.main_dim == 0 ? kDim : params.main_dim;
  return prefix_for(seg_basename) + "_R" + std::to_string(params.R) + "_MD" +
         std::to_string(main_dim) + ".index";
}

void write_bytes(const std::filesystem::path &path, std::string_view bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
  ASSERT_TRUE(out.is_open()) << path;
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  ASSERT_TRUE(out.good()) << path;
}

void write_index(const std::filesystem::path &path, uint64_t count, std::string_view payload) {
  std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
  ASSERT_TRUE(out.is_open()) << path;
  out.write(reinterpret_cast<const char *>(&count), sizeof(count));
  out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  ASSERT_TRUE(out.good()) << path;
}

auto read_bytes(const std::filesystem::path &path) -> std::vector<char> {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

auto labels(uint64_t n = kCount, uint64_t base = 1000) -> std::vector<uint64_t> {
  std::vector<uint64_t> out(n);
  std::iota(out.begin(), out.end(), base);
  return out;
}

auto read_ids(const std::filesystem::path &path) -> std::vector<uint64_t> {
  std::ifstream in(path, std::ios::binary);
  const auto bytes = std::filesystem::file_size(path);
  std::vector<uint64_t> out(bytes / sizeof(uint64_t));
  in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(bytes));
  return out;
}

void expect_no_tmp_debris(const std::filesystem::path &parent) {
  for (const auto &entry : std::filesystem::directory_iterator(parent)) {
    EXPECT_FALSE(entry.path().filename().string().starts_with(".tmp_"))
        << "leftover tmp dir: " << entry.path();
  }
}

class LaserSegmentImporterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    tmp_root_ =
        std::filesystem::temp_directory_path() /
        (std::string("alaya_laser_importer_") + std::to_string(::getpid()) + "_" + info->name());
    std::filesystem::remove_all(tmp_root_);
    src_dir_ = tmp_root_ / "src";
    seg_parent_ = tmp_root_ / "segments";
    std::filesystem::create_directories(src_dir_);
    std::filesystem::create_directories(seg_parent_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_root_, ec);
  }

  auto seg_dir(std::string_view id = "seg_00000001") const -> std::filesystem::path {
    return seg_parent_ / std::string(id);
  }

  auto populate_artifacts(std::string_view seg_basename,
                          uint64_t index_count = kCount,
                          bool optional = true,
                          LaserSegmentImportParams params = {}) const -> std::filesystem::path {
    const auto prefix = prefix_for(seg_basename);
    const auto index_name = index_filename(seg_basename, params);
    const auto index_path = src_dir_ / index_name;
    write_index(index_path, index_count, "index-payload");
    write_bytes(src_dir_ / (index_name + "_rotator"), "rotator-payload");
    write_bytes(src_dir_ / (index_name + "_cache_ids"), "cache-ids-payload");
    write_bytes(src_dir_ / (index_name + "_cache_nodes"), "cache-nodes-payload");
    if (optional) {
      write_bytes(src_dir_ / (prefix + "_medoids"), "medoids-payload");
      write_bytes(src_dir_ / (prefix + "_medoids_indices"), "medoids-indices-payload");
      write_bytes(src_dir_ / (prefix + "_pca.bin"), "pca-payload");
    }
    write_bytes(src_dir_ / (prefix + "_pca_base.fbin"), "ignored-intermediate");
    return index_path;
  }

  void skip_if_laser_gate_not_ready() const {
    if (!engine_supported_v1(DiskIndexType::Laser)) {
      GTEST_SKIP() << "Laser factory gate is not enabled yet; Phase 5 wiring owns the gate";
    }
  }

  std::filesystem::path tmp_root_;
  std::filesystem::path src_dir_;
  std::filesystem::path seg_parent_;
};

TEST_F(LaserSegmentImporterTest, laser_segment_importer_writes_expected_manifest_and_ids) {
  skip_if_laser_gate_not_ready();
  auto ids = labels();
  const auto target = seg_dir();
  populate_artifacts(target.filename().string());

  LaserSegmentImporter importer(kDim, MetricType::L2, {});
  const auto manifest = importer.import_from(src_dir_, ids.data(), ids.size(), target);

  EXPECT_EQ(manifest.version, kManifestVersion);
  EXPECT_EQ(manifest.segment_id, "seg_00000001");
  EXPECT_EQ(manifest.index_type, DiskIndexType::Laser);
  EXPECT_EQ(manifest.metric, MetricType::L2);
  EXPECT_EQ(manifest.dim, kDim);
  EXPECT_EQ(manifest.count, ids.size());
  EXPECT_EQ(manifest.ids_file, "ids.u64.bin");
  EXPECT_EQ(manifest.vectors_file, "");

  const auto loaded = SegmentManifest::load(target / "manifest.txt");
  EXPECT_EQ(loaded.vectors_file, "");
  EXPECT_EQ(read_ids(target / "ids.u64.bin"), ids);
  EXPECT_TRUE(std::filesystem::exists(target / loaded.x_extras.at("x_laser_index_file")));
  EXPECT_TRUE(std::filesystem::exists(target / loaded.x_extras.at("x_laser_rotator_file")));
  EXPECT_TRUE(std::filesystem::exists(target / loaded.x_extras.at("x_laser_cache_ids_file")));
  EXPECT_TRUE(std::filesystem::exists(target / loaded.x_extras.at("x_laser_cache_nodes_file")));
  EXPECT_TRUE(std::filesystem::exists(target / loaded.x_extras.at("x_laser_medoids_file")));
  EXPECT_TRUE(std::filesystem::exists(target / loaded.x_extras.at("x_laser_medoids_indices_file")));
  EXPECT_TRUE(std::filesystem::exists(target / loaded.x_extras.at("x_laser_pca_file")));
  EXPECT_FALSE(std::filesystem::exists(target / "dsqg_seg_00000001_pca_base.fbin"));
  expect_no_tmp_debris(seg_parent_);
}

TEST_F(LaserSegmentImporterTest, minimum_required_artifacts) {
  skip_if_laser_gate_not_ready();
  auto ids = labels();
  const auto target = seg_dir();
  populate_artifacts(target.filename().string(), kCount, false);

  LaserSegmentImporter importer(kDim, MetricType::L2, {});
  importer.import_from(src_dir_, ids.data(), ids.size(), target);

  const auto manifest = SegmentManifest::load(target / "manifest.txt");
  EXPECT_TRUE(manifest.x_extras.contains("x_laser_index_file"));
  EXPECT_TRUE(manifest.x_extras.contains("x_laser_rotator_file"));
  EXPECT_TRUE(manifest.x_extras.contains("x_laser_cache_ids_file"));
  EXPECT_TRUE(manifest.x_extras.contains("x_laser_cache_nodes_file"));
  EXPECT_FALSE(manifest.x_extras.contains("x_laser_medoids_file"));
  EXPECT_FALSE(manifest.x_extras.contains("x_laser_medoids_indices_file"));
  EXPECT_FALSE(manifest.x_extras.contains("x_laser_pca_file"));
}

TEST_F(LaserSegmentImporterTest, refuses_existing_seg_dir) {
  const auto target = seg_dir();
  std::filesystem::create_directories(target);
  write_bytes(target / "marker.txt", "keep");

  LaserSegmentImporter importer(kDim, MetricType::L2, {});
  EXPECT_THROW((void)importer.import_from(src_dir_, labels().data(), kCount, target),
               std::runtime_error);
  EXPECT_TRUE(std::filesystem::exists(target / "marker.txt"));
}

TEST_F(LaserSegmentImporterTest, rejects_missing_required_artifact) {
  skip_if_laser_gate_not_ready();
  auto ids = labels();
  const auto target = seg_dir();
  populate_artifacts(target.filename().string());
  const auto missing = src_dir_ / (index_filename(target.filename().string(), {}) + "_rotator");
  std::filesystem::remove(missing);

  LaserSegmentImporter importer(kDim, MetricType::L2, {});
  try {
    (void)importer.import_from(src_dir_, ids.data(), ids.size(), target);
    FAIL() << "expected missing artifact rejection";
  } catch (const std::runtime_error &e) {
    EXPECT_TRUE(contains(e.what(), missing.string())) << e.what();
  }
  EXPECT_FALSE(std::filesystem::exists(target));
  expect_no_tmp_debris(seg_parent_);
}

TEST_F(LaserSegmentImporterTest, rejects_count_mismatch) {
  skip_if_laser_gate_not_ready();
  auto ids = labels();
  const auto target = seg_dir();
  const auto index_path = populate_artifacts(target.filename().string(), kCount + 1);

  LaserSegmentImporter importer(kDim, MetricType::L2, {});
  try {
    (void)importer.import_from(src_dir_, ids.data(), ids.size(), target);
    FAIL() << "expected count mismatch rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, std::to_string(kCount))) << msg;
    EXPECT_TRUE(contains(msg, std::to_string(kCount + 1))) << msg;
    EXPECT_TRUE(contains(msg, index_path.string())) << msg;
  }
  EXPECT_FALSE(std::filesystem::exists(target));
}

TEST_F(LaserSegmentImporterTest, rejects_ip_metric) {
  const auto target = seg_dir();
  LaserSegmentImporter importer(kDim, MetricType::IP, {});
  try {
    (void)importer.import_from(src_dir_, labels().data(), kCount, target);
    FAIL() << "expected IP metric rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, "ip")) << msg;
    EXPECT_TRUE(contains(msg, "not implemented in v1")) << msg;
  }
  EXPECT_FALSE(std::filesystem::exists(target));
  expect_no_tmp_debris(seg_parent_);
}

TEST_F(LaserSegmentImporterTest, rejects_cos_metric) {
  const auto target = seg_dir();
  LaserSegmentImporter importer(kDim, MetricType::COS, {});
  try {
    (void)importer.import_from(src_dir_, labels().data(), kCount, target);
    FAIL() << "expected COS metric rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, "cos")) << msg;
    EXPECT_TRUE(contains(msg, "not implemented in v1")) << msg;
  }
  EXPECT_FALSE(std::filesystem::exists(target));
}

TEST(LaserSegmentImporterConstructorTest, rejects_dim_below_floor) {
  try {
    LaserSegmentImporter importer(64, MetricType::L2, {});
    (void)importer;
    FAIL() << "expected dim floor rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, "64")) << msg;
    EXPECT_TRUE(contains(msg, "128")) << msg;
  }
}

TEST(LaserSegmentImporterConstructorTest, rejects_dim_not_power_of_two) {
  try {
    LaserSegmentImporter importer(300, MetricType::L2, {});
    (void)importer;
    FAIL() << "expected power-of-two rejection";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_TRUE(contains(msg, "300")) << msg;
    EXPECT_TRUE(contains(msg, "power of two")) << msg;
  }
}

TEST_F(LaserSegmentImporterTest, manifest_records_x_laser_extras) {
  skip_if_laser_gate_not_ready();
  LaserSegmentImportParams params;
  params.R = 64;
  params.main_dim = kDim;
  params.default_ef = 64;
  params.default_beam_width = 4;
  params.search_dram_budget_gb = 0.5F;
  auto ids = labels();
  const auto target = seg_dir();
  populate_artifacts(target.filename().string(), kCount, true, params);

  LaserSegmentImporter importer(kDim, MetricType::L2, params);
  importer.import_from(src_dir_, ids.data(), ids.size(), target);

  const auto manifest = SegmentManifest::load(target / "manifest.txt");
  const auto prefix = prefix_for(target.filename().string());
  EXPECT_EQ(manifest.x_extras.at("x_laser_filename_prefix"), prefix);
  EXPECT_EQ(manifest.x_extras.at("x_laser_index_file"), prefix + "_R64_MD128.index");
  EXPECT_EQ(manifest.x_extras.at("x_laser_rotator_file"), prefix + "_R64_MD128.index_rotator");
  EXPECT_EQ(manifest.x_extras.at("x_laser_cache_ids_file"), prefix + "_R64_MD128.index_cache_ids");
  EXPECT_EQ(manifest.x_extras.at("x_laser_cache_nodes_file"),
            prefix + "_R64_MD128.index_cache_nodes");
  EXPECT_EQ(manifest.x_extras.at("x_laser_medoids_file"), prefix + "_medoids");
  EXPECT_EQ(manifest.x_extras.at("x_laser_medoids_indices_file"), prefix + "_medoids_indices");
  EXPECT_EQ(manifest.x_extras.at("x_laser_pca_file"), prefix + "_pca.bin");
  EXPECT_EQ(manifest.x_extras.at("x_R"), "64");
  EXPECT_EQ(manifest.x_extras.at("x_main_dim"), "128");
  EXPECT_EQ(manifest.x_extras.at("x_default_ef"), "64");
  EXPECT_EQ(manifest.x_extras.at("x_default_beam_width"), "4");
  EXPECT_EQ(manifest.x_extras.at("x_laser_native_format_version"), "1");
  EXPECT_EQ(manifest.x_extras.at("x_platform_requires"), "linux+libaio");
  EXPECT_EQ(manifest.x_extras.at("x_laser_search_dram_budget_gb"), "0.5");
  EXPECT_EQ(manifest.x_extras.at("x_laser_distance_field_supported"), "false");
  for (const auto &[key, value] : manifest.x_extras) {
    if (key.starts_with("x_laser_") && key.ends_with("_file")) {
      EXPECT_TRUE(std::filesystem::exists(target / value)) << key << "=" << value;
    }
  }
}

TEST_F(LaserSegmentImporterTest, native_artifact_byte_identical_to_source) {
  skip_if_laser_gate_not_ready();
  auto ids = labels();
  const auto target = seg_dir();
  populate_artifacts(target.filename().string());

  LaserSegmentImporter importer(kDim, MetricType::L2, {});
  importer.import_from(src_dir_, ids.data(), ids.size(), target);

  const auto manifest = SegmentManifest::load(target / "manifest.txt");
  for (const auto &key : {"x_laser_index_file",
                          "x_laser_rotator_file",
                          "x_laser_cache_ids_file",
                          "x_laser_cache_nodes_file",
                          "x_laser_medoids_file",
                          "x_laser_medoids_indices_file",
                          "x_laser_pca_file"}) {
    const auto &filename = manifest.x_extras.at(key);
    EXPECT_EQ(read_bytes(target / filename), read_bytes(src_dir_ / filename)) << key;
  }
}

#else

TEST(LaserSegmentImporterUnsupportedBuildTest, stub_throws_dual_substring_message) {
  try {
    LaserSegmentImporter importer(128, MetricType::L2, {});
    (void)importer;
    FAIL() << "expected unsupported Laser importer";
  } catch (const std::runtime_error &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("disk_laser"), std::string::npos) << msg;
    EXPECT_NE(msg.find("not implemented in v1"), std::string::npos) << msg;
  }
}

#endif

}  // namespace
}  // namespace alaya::disk
