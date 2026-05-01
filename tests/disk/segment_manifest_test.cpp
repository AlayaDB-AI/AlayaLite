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

#include "index/disk/segment_manifest.hpp"
#include <gtest/gtest.h>
#include <unistd.h>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include "index/disk/laser_segment_importer.hpp"
#include "index/disk/laser_segment_searcher.hpp"

namespace alaya::disk {

namespace {

constexpr const char *kCanonicalSegmentManifest =
    "version=1\n"
    "segment_id=seg_00000007\n"
    "index_type=disk_flat\n"
    "metric=L2\n"
    "dim=128\n"
    "count=1000\n"
    "ids_file=ids.u64.bin\n"
    "vectors_file=vectors.f32.bin\n";

constexpr const char *kCanonicalCollectionManifest =
    "version=1\n"
    "dim=128\n"
    "metric=L2\n"
    "index_type=disk_flat\n"
    "next_segment_id=4\n"
    "segment=seg_00000001\n"
    "segment=seg_00000002\n"
    "segment=seg_00000003\n";

class ManifestTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    tmp_dir_ = std::filesystem::temp_directory_path() /
               (std::string("alaya_disk_test_") + std::to_string(::getpid()) + "_" +
                info->test_suite_name() + "_" + info->name());
    std::filesystem::remove_all(tmp_dir_);
    std::filesystem::create_directories(tmp_dir_);
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_dir_, ec);
  }
  auto write_text(const std::string &filename, std::string_view contents) const
      -> std::filesystem::path {
    auto path = tmp_dir_ / filename;
    std::ofstream f(path, std::ios::binary | std::ios::out | std::ios::trunc);
    f.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    f.close();
    return path;
  }

  std::filesystem::path tmp_dir_;
};

}  // namespace

TEST_F(ManifestTest, SegmentManifestRoundtripAllMetrics) {
  for (auto metric : {MetricType::L2, MetricType::IP, MetricType::COS}) {
    SegmentManifest m;
    m.version = 1;
    m.segment_id = "seg_00000007";
    m.index_type = DiskIndexType::Flat;
    m.metric = metric;
    m.dim = 128;
    m.count = 1000;
    m.ids_file = "ids.u64.bin";
    m.vectors_file = "vectors.f32.bin";

    auto path = tmp_dir_ / ("manifest_metric_" + std::to_string(static_cast<int>(metric)) + ".txt");
    m.save(path);
    auto loaded = SegmentManifest::load(path);

    EXPECT_EQ(loaded.version, 1u);
    EXPECT_EQ(loaded.segment_id, "seg_00000007");
    EXPECT_EQ(loaded.index_type, DiskIndexType::Flat);
    EXPECT_EQ(loaded.metric, metric)
        << "metric round-trip failed for enum value " << static_cast<int>(metric);
    EXPECT_EQ(loaded.dim, 128u);
    EXPECT_EQ(loaded.count, 1000u);
    EXPECT_EQ(loaded.ids_file, "ids.u64.bin");
    EXPECT_EQ(loaded.vectors_file, "vectors.f32.bin");
  }
}

TEST_F(ManifestTest, SegmentManifestXExtrasFullRoundtrip) {
  // Write a manifest containing an x_* extension, load it, save it back,
  // load again, and confirm the x_* keys survive the second round-trip.
  std::string text{kCanonicalSegmentManifest};
  text += "x_content_sha256=deadbeef\n";
  text += "x_lineage_id=run_42\n";
  auto src = write_text("with_extras.txt", text);

  auto first = SegmentManifest::load(src);
  ASSERT_EQ(first.x_extras.size(), 2u);
  EXPECT_EQ(first.x_extras.at("x_content_sha256"), "deadbeef");
  EXPECT_EQ(first.x_extras.at("x_lineage_id"), "run_42");

  auto resaved = tmp_dir_ / "resaved.txt";
  first.save(resaved);

  auto second = SegmentManifest::load(resaved);
  ASSERT_EQ(second.x_extras.size(), 2u);
  EXPECT_EQ(second.x_extras.at("x_content_sha256"), "deadbeef");
  EXPECT_EQ(second.x_extras.at("x_lineage_id"), "run_42");
}

TEST_F(ManifestTest, SegmentManifestNegativeMissingField) {
  static constexpr std::array<std::string_view, 8> kFields{
      "version",
      "segment_id",
      "index_type",
      "metric",
      "dim",
      "count",
      "ids_file",
      "vectors_file",
  };
  static constexpr std::string_view kBase = kCanonicalSegmentManifest;

  for (auto field : kFields) {
    std::string mutated;
    mutated.reserve(kBase.size());
    size_t pos = 0;
    while (pos < kBase.size()) {
      size_t newline = kBase.find('\n', pos);
      auto line = kBase.substr(pos, newline - pos);
      if (line.starts_with(field) && line.size() > field.size() && line[field.size()] == '=') {
        // skip this line entirely
      } else {
        mutated.append(line);
        mutated += '\n';
      }
      pos = newline + 1;
    }
    auto path = write_text(std::string("missing_") + std::string(field) + ".txt", mutated);
    try {
      (void)SegmentManifest::load(path);
      ADD_FAILURE() << "expected throw for missing field '" << field << "'";
    } catch (const std::exception &e) {
      const std::string msg = e.what();
      EXPECT_NE(msg.find(field), std::string::npos)
          << "missing-field exception must name the field; field=" << field << " msg=" << msg;
      EXPECT_NE(msg.find("missing"), std::string::npos)
          << "missing-field exception must say 'missing'; field=" << field << " msg=" << msg;
    }
  }
}

TEST_F(ManifestTest, SegmentManifestEmptyValueDistinctFromMissing) {
  std::string mutated{kCanonicalSegmentManifest};
  // Replace `dim=128` with `dim=`
  auto pos = mutated.find("dim=128");
  ASSERT_NE(pos, std::string::npos);
  mutated.replace(pos, 7, "dim=");
  auto path = write_text("empty_dim.txt", mutated);
  try {
    (void)SegmentManifest::load(path);
    ADD_FAILURE() << "expected throw for empty dim value";
  } catch (const std::exception &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("dim"), std::string::npos);
    EXPECT_NE(msg.find("empty value"), std::string::npos)
        << "must distinguish empty value from missing key; msg=" << msg;
  }
}

TEST_F(ManifestTest, SegmentManifestNegativeInvalidValue) {
  // dim=0
  {
    std::string m{kCanonicalSegmentManifest};
    auto pos = m.find("dim=128");
    m.replace(pos, 7, "dim=0");
    auto path = write_text("dim_zero.txt", m);
    EXPECT_THROW((void)SegmentManifest::load(path), std::invalid_argument);
  }
  // count=0
  {
    std::string m{kCanonicalSegmentManifest};
    auto pos = m.find("count=1000");
    m.replace(pos, 10, "count=0");
    auto path = write_text("count_zero.txt", m);
    EXPECT_THROW((void)SegmentManifest::load(path), std::invalid_argument);
  }
  // metric=BOGUS — message must contain "BOGUS"
  {
    std::string m{kCanonicalSegmentManifest};
    auto pos = m.find("metric=L2");
    m.replace(pos, 9, "metric=BOGUS");
    auto path = write_text("metric_bogus.txt", m);
    try {
      (void)SegmentManifest::load(path);
      ADD_FAILURE() << "expected throw";
    } catch (const std::exception &e) {
      EXPECT_NE(std::string(e.what()).find("BOGUS"), std::string::npos);
    }
  }
  // metric=NONE
  {
    std::string m{kCanonicalSegmentManifest};
    auto pos = m.find("metric=L2");
    m.replace(pos, 9, "metric=NONE");
    auto path = write_text("metric_none.txt", m);
    EXPECT_THROW((void)SegmentManifest::load(path), std::invalid_argument);
  }
  // version=99 — message must contain both "99" and "1"
  {
    std::string m{kCanonicalSegmentManifest};
    auto pos = m.find("version=1");
    m.replace(pos, 9, "version=99");
    auto path = write_text("version_99.txt", m);
    try {
      (void)SegmentManifest::load(path);
      ADD_FAILURE() << "expected throw";
    } catch (const std::exception &e) {
      const std::string msg = e.what();
      EXPECT_NE(msg.find("99"), std::string::npos);
      EXPECT_NE(msg.find('1'), std::string::npos);
    }
  }
}

TEST_F(ManifestTest, SegmentManifestNegativeUnknownKey) {
  // bogus_field=foo (no x_ prefix) → reject
  {
    std::string m{kCanonicalSegmentManifest};
    m += "bogus_field=foo\n";
    auto path = write_text("bogus_key.txt", m);
    try {
      (void)SegmentManifest::load(path);
      ADD_FAILURE() << "expected throw for non-x_ unknown key";
    } catch (const std::exception &e) {
      EXPECT_NE(std::string(e.what()).find("bogus_field"), std::string::npos);
    }
  }
  // x_extra=foo → retained silently
  {
    std::string m{kCanonicalSegmentManifest};
    m += "x_extra_field=foo\n";
    auto path = write_text("x_extra.txt", m);
    auto loaded = SegmentManifest::load(path);
    ASSERT_TRUE(loaded.x_extras.contains("x_extra_field"));
    EXPECT_EQ(loaded.x_extras.at("x_extra_field"), "foo");
  }
}

TEST_F(ManifestTest, SegmentManifestPathTraversalInIdsFileRejected) {
  // The test filename slot uses an index because some bad_values contain '/'
  // or '..' which would re-interpret the std::filesystem path during write.
  int idx = 0;
  for (const auto *bad_value : {"../etc/passwd", "ids/foo.bin", "..", ".", "/tmp/x"}) {
    std::string m{kCanonicalSegmentManifest};
    auto pos = m.find("ids_file=ids.u64.bin");
    ASSERT_NE(pos, std::string::npos);
    m.replace(pos, 20, std::string("ids_file=") + bad_value);
    auto path = write_text("ids_traversal_" + std::to_string(idx++) + ".txt", m);
    EXPECT_THROW((void)SegmentManifest::load(path), std::invalid_argument)
        << "expected ids_file basename rejection for: " << bad_value;
  }

  // Embedded NUL: build the value with explicit length so the NUL byte
  // survives into the manifest text.
  {
    std::string m{kCanonicalSegmentManifest};
    auto pos = m.find("ids_file=ids.u64.bin");
    ASSERT_NE(pos, std::string::npos);
    std::string bad_with_nul("ids_file=a\0b", 12);  // 12 = literal byte count
    m.replace(pos, 20, bad_with_nul);
    auto path = write_text("ids_traversal_nul.txt", m);
    EXPECT_THROW((void)SegmentManifest::load(path), std::invalid_argument);
  }
}

TEST_F(ManifestTest, SegmentManifestPathTraversalInVectorsFileRejected) {
  std::string m{kCanonicalSegmentManifest};
  auto pos = m.find("vectors_file=vectors.f32.bin");
  ASSERT_NE(pos, std::string::npos);
  m.replace(pos, 28, "vectors_file=../../etc/shadow");
  auto path = write_text("vectors_traversal.txt", m);
  EXPECT_THROW((void)SegmentManifest::load(path), std::invalid_argument);
}

TEST_F(ManifestTest, SegmentManifestAllowsEmptyVectorsFileForLaserSegments) {
  SegmentManifest m;
  m.version = 1;
  m.segment_id = "seg_00000007";
  m.index_type = DiskIndexType::Laser;
  m.metric = MetricType::L2;
  m.dim = 128;
  m.count = 1000;
  m.ids_file = "ids.u64.bin";
  m.vectors_file = "";
  auto path = tmp_dir_ / "laser_empty_vectors_file.txt";
  m.save(path);

  std::ifstream saved(path, std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(saved)), std::istreambuf_iterator<char>());
  EXPECT_NE(text.find("\nvectors_file=\n"), std::string::npos);

  auto loaded = SegmentManifest::load(path);
  EXPECT_EQ(loaded.index_type, DiskIndexType::Laser);
  EXPECT_EQ(loaded.ids_file, "ids.u64.bin");
  EXPECT_TRUE(loaded.vectors_file.empty());
}

TEST_F(ManifestTest, SegmentManifestNegativeFormat) {
  // duplicate scalar key
  {
    std::string m{kCanonicalSegmentManifest};
    m += "dim=64\n";
    auto path = write_text("dup_dim.txt", m);
    try {
      (void)SegmentManifest::load(path);
      ADD_FAILURE() << "expected throw for duplicate scalar key";
    } catch (const std::exception &e) {
      const std::string msg = e.what();
      EXPECT_NE(msg.find("dim"), std::string::npos);
      EXPECT_NE(msg.find("duplicate"), std::string::npos);
    }
  }
  // UTF-8 BOM
  {
    std::string m;
    m += static_cast<char>(0xEF);
    m += static_cast<char>(0xBB);
    m += static_cast<char>(0xBF);
    m += kCanonicalSegmentManifest;
    auto path = write_text("bom.txt", m);
    try {
      (void)SegmentManifest::load(path);
      ADD_FAILURE() << "expected throw for UTF-8 BOM";
    } catch (const std::exception &e) {
      const std::string msg = e.what();
      bool mentions_encoding =
          msg.find("BOM") != std::string::npos || msg.find("encoding") != std::string::npos;
      EXPECT_TRUE(mentions_encoding) << "msg=" << msg;
    }
  }
  // segment_id=../etc/passwd
  {
    std::string m{kCanonicalSegmentManifest};
    auto pos = m.find("segment_id=seg_00000007");
    m.replace(pos, 23, "segment_id=../etc/passwd");
    auto path = write_text("traversal.txt", m);
    EXPECT_THROW((void)SegmentManifest::load(path), std::invalid_argument);
  }
  // segment_id=seg_001 (wrong digit count)
  {
    std::string m{kCanonicalSegmentManifest};
    auto pos = m.find("segment_id=seg_00000007");
    m.replace(pos, 23, "segment_id=seg_001");
    auto path = write_text("seg_short.txt", m);
    EXPECT_THROW((void)SegmentManifest::load(path), std::invalid_argument);
  }
}

TEST_F(ManifestTest, SegmentManifestFormatQuirks) {
  // CRLF line endings
  {
    std::string m;
    m.reserve(256);
    std::string_view base = kCanonicalSegmentManifest;
    size_t pos = 0;
    while (pos < base.size()) {
      size_t nl = base.find('\n', pos);
      m.append(base.substr(pos, nl - pos));
      m += "\r\n";
      pos = nl + 1;
    }
    auto path = write_text("crlf.txt", m);
    auto loaded = SegmentManifest::load(path);
    EXPECT_EQ(loaded.dim, 128u);
    EXPECT_EQ(loaded.metric, MetricType::L2);
  }
  // Leading and trailing whitespace + comments + blank lines
  {
    std::string m =
        "# this is a comment\n"
        "\n"
        "   version  =  1   \n"
        "  segment_id =seg_00000007\n"
        "index_type=disk_flat   \n"
        "  metric  =  L2  \n"
        "dim=128\n"
        "\n"
        "count=1000\n"
        "ids_file=ids.u64.bin\n"
        "vectors_file=vectors.f32.bin\n"
        "# trailing comment\n";
    auto path = write_text("quirks.txt", m);
    auto loaded = SegmentManifest::load(path);
    EXPECT_EQ(loaded.dim, 128u);
    EXPECT_EQ(loaded.segment_id, "seg_00000007");
    EXPECT_EQ(loaded.metric, MetricType::L2);
  }
}

TEST_F(ManifestTest, CollectionManifestRoundtripRepeatedSegmentForm) {
  CollectionManifest c;
  c.version = 1;
  c.dim = 128;
  c.metric = MetricType::COS;
  c.index_type = DiskIndexType::Flat;
  c.next_segment_id = 4;
  c.segment_ids = {"seg_00000001", "seg_00000002", "seg_00000003"};

  auto path = tmp_dir_ / "coll.txt";
  c.save(path);
  auto loaded = CollectionManifest::load(path);
  EXPECT_EQ(loaded.dim, 128u);
  EXPECT_EQ(loaded.metric, MetricType::COS);
  EXPECT_EQ(loaded.index_type, DiskIndexType::Flat);
  EXPECT_EQ(loaded.next_segment_id, 4u);
  ASSERT_EQ(loaded.segment_ids.size(), 3u);
  EXPECT_EQ(loaded.segment_ids[0], "seg_00000001");
  EXPECT_EQ(loaded.segment_ids[1], "seg_00000002");
  EXPECT_EQ(loaded.segment_ids[2], "seg_00000003");
}

TEST_F(ManifestTest, CollectionManifestEmptySegmentList) {
  // freshly-created collection: no segments, next_segment_id == 1
  std::string m =
      "version=1\n"
      "dim=128\n"
      "metric=L2\n"
      "index_type=disk_flat\n"
      "next_segment_id=1\n";
  auto path = write_text("empty.txt", m);
  auto loaded = CollectionManifest::load(path);
  EXPECT_EQ(loaded.next_segment_id, 1u);
  EXPECT_TRUE(loaded.segment_ids.empty());
}

TEST_F(ManifestTest, CollectionManifestSegmentsCsvForm) {
  std::string m =
      "version=1\n"
      "dim=128\n"
      "metric=L2\n"
      "index_type=disk_flat\n"
      "next_segment_id=3\n"
      "segments=seg_00000001,seg_00000002\n";
  auto path = write_text("csv.txt", m);
  auto loaded = CollectionManifest::load(path);
  ASSERT_EQ(loaded.segment_ids.size(), 2u);
  EXPECT_EQ(loaded.segment_ids[0], "seg_00000001");
  EXPECT_EQ(loaded.segment_ids[1], "seg_00000002");
}

TEST_F(ManifestTest, CollectionManifestSegmentsTrailingComma) {
  std::string bad =
      "version=1\n"
      "dim=128\n"
      "metric=L2\n"
      "index_type=disk_flat\n"
      "next_segment_id=3\n"
      "segments=seg_00000001,seg_00000002,\n";
  auto path = write_text("trailing.txt", bad);
  try {
    (void)CollectionManifest::load(path);
    ADD_FAILURE() << "expected throw for trailing comma";
  } catch (const std::exception &e) {
    const std::string msg = e.what();
    bool mentions =
        msg.find("trailing") != std::string::npos || msg.find("empty entry") != std::string::npos;
    EXPECT_TRUE(mentions) << "msg=" << msg;
  }
}

TEST_F(ManifestTest, CollectionManifestSegmentsEmptyEntry) {
  std::string bad =
      "version=1\n"
      "dim=128\n"
      "metric=L2\n"
      "index_type=disk_flat\n"
      "next_segment_id=3\n"
      "segments=seg_00000001,,seg_00000002\n";
  auto path = write_text("empty_entry.txt", bad);
  EXPECT_THROW((void)CollectionManifest::load(path), std::invalid_argument);
}

TEST_F(ManifestTest, CollectionManifestMixedFormsRejected) {
  std::string bad =
      "version=1\n"
      "dim=128\n"
      "metric=L2\n"
      "index_type=disk_flat\n"
      "next_segment_id=3\n"
      "segment=seg_00000001\n"
      "segments=seg_00000002\n";
  auto path = write_text("mixed.txt", bad);
  EXPECT_THROW((void)CollectionManifest::load(path), std::invalid_argument);
}

}  // namespace alaya::disk
