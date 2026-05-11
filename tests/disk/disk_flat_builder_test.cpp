// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <signal.h>     // NOLINT(modernize-deprecated-headers)
#include <sys/resource.h>
#include <unistd.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "index/disk/disk_flat_builder.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "utils/metric_type.hpp"

namespace alaya::disk {

namespace {

class DiskFlatBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto pid_str = std::to_string(static_cast<long long>(::getpid()));
    auto base = std::filesystem::temp_directory_path() /
                ("alaya_flat_builder_" + pid_str + "_" +
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

  static auto fnv1a64(const void *data, size_t bytes) -> uint64_t {
    const auto *p = static_cast<const uint8_t *>(data);
    uint64_t h = 0xCBF29CE484222325ULL;
    for (size_t i = 0; i < bytes; ++i) {
      h ^= p[i];
      h *= 0x100000001B3ULL;
    }
    return h;
  }

  std::filesystem::path tmp_root_;
  std::filesystem::path seg_parent_;
};

TEST_F(DiskFlatBuilderTest, RoundtripL2) {
  constexpr uint32_t kDim = 16;
  constexpr uint64_t kN = 32;
  auto vectors = make_random_vectors(kN, kDim);
  auto labels = sequential_labels(kN);

  DiskFlatBuilder b(kDim, MetricType::L2);
  b.add_batch(vectors.data(), labels.data(), kN);
  auto seg_dir = seg_parent_ / "seg_00000001";
  auto manifest = b.finish(seg_dir);

  EXPECT_EQ(manifest.dim, kDim);
  EXPECT_EQ(manifest.count, kN);
  EXPECT_EQ(manifest.metric, MetricType::L2);
  EXPECT_EQ(manifest.index_type, DiskIndexType::Flat);
  EXPECT_EQ(manifest.segment_id, "seg_00000001");

  ASSERT_TRUE(std::filesystem::exists(seg_dir / "manifest.txt"));
  ASSERT_TRUE(std::filesystem::exists(seg_dir / "ids.u64.bin"));
  ASSERT_TRUE(std::filesystem::exists(seg_dir / "vectors.f32.bin"));

  auto loaded = SegmentManifest::load(seg_dir / "manifest.txt");
  EXPECT_EQ(loaded.segment_id, "seg_00000001");
  EXPECT_EQ(loaded.dim, kDim);
  EXPECT_EQ(loaded.count, kN);
  EXPECT_EQ(loaded.metric, MetricType::L2);

  EXPECT_EQ(std::filesystem::file_size(seg_dir / "ids.u64.bin"), kN * sizeof(uint64_t));
  EXPECT_EQ(std::filesystem::file_size(seg_dir / "vectors.f32.bin"),
            static_cast<uintmax_t>(kN) * kDim * sizeof(float));

  // No leftover tmp dir.
  for (const auto &entry : std::filesystem::directory_iterator(seg_parent_)) {
    auto name = entry.path().filename().string();
    EXPECT_FALSE(name.starts_with(".tmp_")) << "leftover tmp dir: " << name;
  }
}

TEST_F(DiskFlatBuilderTest, ConstructionRejectsMetricNone) {
  EXPECT_THROW(DiskFlatBuilder(16, MetricType::NONE), std::invalid_argument);
}

TEST_F(DiskFlatBuilderTest, ConstructionRejectsDimZero) {
  EXPECT_THROW(DiskFlatBuilder(0, MetricType::L2), std::invalid_argument);
}

TEST_F(DiskFlatBuilderTest, CallerBufferNotMutatedBuildL2) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 12;
  auto vectors = make_random_vectors(kN, kDim);
  auto labels = sequential_labels(kN);
  const auto vec_hash_before = fnv1a64(vectors.data(), vectors.size() * sizeof(float));
  const auto lab_hash_before = fnv1a64(labels.data(), labels.size() * sizeof(uint64_t));

  DiskFlatBuilder b(kDim, MetricType::L2);
  b.add_batch(vectors.data(), labels.data(), kN);
  b.finish(seg_parent_ / "seg_00000001");

  EXPECT_EQ(fnv1a64(vectors.data(), vectors.size() * sizeof(float)), vec_hash_before);
  EXPECT_EQ(fnv1a64(labels.data(), labels.size() * sizeof(uint64_t)), lab_hash_before);
}

TEST_F(DiskFlatBuilderTest, CallerBufferNotMutatedBuildIp) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 12;
  auto vectors = make_random_vectors(kN, kDim, 7);
  auto labels = sequential_labels(kN);
  const auto vec_hash_before = fnv1a64(vectors.data(), vectors.size() * sizeof(float));
  const auto lab_hash_before = fnv1a64(labels.data(), labels.size() * sizeof(uint64_t));

  DiskFlatBuilder b(kDim, MetricType::IP);
  b.add_batch(vectors.data(), labels.data(), kN);
  b.finish(seg_parent_ / "seg_00000002");

  EXPECT_EQ(fnv1a64(vectors.data(), vectors.size() * sizeof(float)), vec_hash_before);
  EXPECT_EQ(fnv1a64(labels.data(), labels.size() * sizeof(uint64_t)), lab_hash_before);
}

TEST_F(DiskFlatBuilderTest, CallerBufferNotMutatedBuildCos) {
  constexpr uint32_t kDim = 8;
  constexpr uint64_t kN = 12;
  auto vectors = make_random_vectors(kN, kDim, 11);
  auto labels = sequential_labels(kN);
  const auto vec_hash_before = fnv1a64(vectors.data(), vectors.size() * sizeof(float));
  const auto lab_hash_before = fnv1a64(labels.data(), labels.size() * sizeof(uint64_t));

  DiskFlatBuilder b(kDim, MetricType::COS);
  b.add_batch(vectors.data(), labels.data(), kN);
  auto manifest = b.finish(seg_parent_ / "seg_00000003");

  EXPECT_EQ(fnv1a64(vectors.data(), vectors.size() * sizeof(float)), vec_hash_before);
  EXPECT_EQ(fnv1a64(labels.data(), labels.size() * sizeof(uint64_t)), lab_hash_before);

  // Stored vectors are normalized (norm 1.0 within tolerance).
  std::ifstream ifs(seg_parent_ / "seg_00000003" / "vectors.f32.bin", std::ios::binary);
  std::vector<float> stored(static_cast<size_t>(kN) * kDim);
  ifs.read(reinterpret_cast<char *>(stored.data()),
           static_cast<std::streamsize>(stored.size() * sizeof(float)));
  for (uint64_t r = 0; r < kN; ++r) {
    double sum_sq = 0.0;
    for (uint32_t c = 0; c < kDim; ++c) {
      const double v = stored[r * kDim + c];
      sum_sq += v * v;
    }
    EXPECT_NEAR(std::sqrt(sum_sq), 1.0, 1e-5) << "row " << r << " not normalized";
  }
  (void)manifest;
}

TEST_F(DiskFlatBuilderTest, ZeroVectorThrowsBuildCos) {
  constexpr uint32_t kDim = 4;
  constexpr uint64_t kN = 10;
  auto vectors = make_random_vectors(kN, kDim);
  auto labels = sequential_labels(kN);
  // Zero out row 7.
  for (uint32_t c = 0; c < kDim; ++c) {
    vectors[7 * kDim + c] = 0.0F;
  }

  DiskFlatBuilder b(kDim, MetricType::COS);
  b.add_batch(vectors.data(), labels.data(), kN);
  try {
    (void)b.finish(seg_parent_ / "seg_00000001");
    FAIL() << "expected exception on zero-magnitude COS vector";
  } catch (const std::exception &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("7"), std::string::npos)
        << "message must contain offending row index '7', got: " << msg;
  }
}

TEST_F(DiskFlatBuilderTest, ZeroVectorL2Succeeds) {
  constexpr uint32_t kDim = 4;
  constexpr uint64_t kN = 5;
  std::vector<float> vectors(kN * kDim, 0.0F);
  auto labels = sequential_labels(kN);

  DiskFlatBuilder b(kDim, MetricType::L2);
  b.add_batch(vectors.data(), labels.data(), kN);
  EXPECT_NO_THROW(b.finish(seg_parent_ / "seg_00000001"));
}

TEST_F(DiskFlatBuilderTest, SubnormalMagnitudeCosSucceeds) {
  constexpr uint32_t kDim = 4;
  constexpr uint64_t kN = 1;
  std::vector<float> vectors(kN * kDim, 1e-30F);
  auto labels = sequential_labels(kN);

  DiskFlatBuilder b(kDim, MetricType::COS);
  b.add_batch(vectors.data(), labels.data(), kN);
  ASSERT_NO_THROW(b.finish(seg_parent_ / "seg_00000001"));

  // Verify the row was normalized to 1.0 ± 1e-5.
  std::ifstream ifs(seg_parent_ / "seg_00000001" / "vectors.f32.bin", std::ios::binary);
  std::vector<float> stored(kN * kDim);
  ifs.read(reinterpret_cast<char *>(stored.data()),
           static_cast<std::streamsize>(stored.size() * sizeof(float)));
  double sum_sq = 0.0;
  for (uint32_t c = 0; c < kDim; ++c) {
    sum_sq += static_cast<double>(stored[c]) * stored[c];
  }
  EXPECT_NEAR(std::sqrt(sum_sq), 1.0, 1e-5);
}

class NonFiniteParam : public DiskFlatBuilderTest,
                      public ::testing::WithParamInterface<MetricType> {};

TEST_P(NonFiniteParam, NaNComponentThrows) {
  constexpr uint32_t kDim = 4;
  constexpr uint64_t kN = 5;
  auto vectors = make_random_vectors(kN, kDim);
  auto labels = sequential_labels(kN);
  vectors[3 * kDim + 2] = std::numeric_limits<float>::quiet_NaN();  // row 3, pos 2

  DiskFlatBuilder b(kDim, GetParam());
  try {
    b.add_batch(vectors.data(), labels.data(), kN);
    FAIL() << "expected throw on NaN component";
  } catch (const std::exception &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("3"), std::string::npos) << "message must mention row 3: " << msg;
    EXPECT_NE(msg.find("2"), std::string::npos) << "message must mention position 2: " << msg;
    EXPECT_NE(msg.find("NaN"), std::string::npos) << "message must mention NaN: " << msg;
  }
}

TEST_P(NonFiniteParam, PosInfComponentThrows) {
  constexpr uint32_t kDim = 4;
  constexpr uint64_t kN = 5;
  auto vectors = make_random_vectors(kN, kDim);
  auto labels = sequential_labels(kN);
  vectors[3 * kDim + 2] = std::numeric_limits<float>::infinity();

  DiskFlatBuilder b(kDim, GetParam());
  try {
    b.add_batch(vectors.data(), labels.data(), kN);
    FAIL() << "expected throw on +Inf component";
  } catch (const std::exception &e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("Inf"), std::string::npos) << "message must mention Inf: " << msg;
  }
}

TEST_P(NonFiniteParam, NegInfComponentThrows) {
  constexpr uint32_t kDim = 4;
  constexpr uint64_t kN = 5;
  auto vectors = make_random_vectors(kN, kDim);
  auto labels = sequential_labels(kN);
  vectors[3 * kDim + 2] = -std::numeric_limits<float>::infinity();

  DiskFlatBuilder b(kDim, GetParam());
  EXPECT_THROW(b.add_batch(vectors.data(), labels.data(), kN), std::exception);
}

INSTANTIATE_TEST_SUITE_P(AllMetrics, NonFiniteParam,
                         ::testing::Values(MetricType::L2, MetricType::IP, MetricType::COS));

TEST_F(DiskFlatBuilderTest, NZeroAddBatchIsNoop) {
  DiskFlatBuilder b(8, MetricType::L2);
  EXPECT_NO_THROW(b.add_batch(nullptr, nullptr, 0));
  EXPECT_NO_THROW(b.add_batch(reinterpret_cast<const float *>(0xDEADBEEF),
                              reinterpret_cast<const uint64_t *>(0xCAFE), 0));

  // Adding a real batch and finishing produces a count==1 segment, not anything from the no-ops.
  std::vector<float> v(8, 1.0F);
  std::vector<uint64_t> l{42};
  b.add_batch(v.data(), l.data(), 1);
  auto m = b.finish(seg_parent_ / "seg_00000001");
  EXPECT_EQ(m.count, 1u);
}

TEST_F(DiskFlatBuilderTest, ZeroBatchesThenFinishThrows) {
  DiskFlatBuilder b(8, MetricType::L2);
  EXPECT_THROW(b.finish(seg_parent_ / "seg_00000001"), std::runtime_error);

  // No tmp dir was even created (we throw before mkdir).
  for (const auto &entry : std::filesystem::directory_iterator(seg_parent_)) {
    auto name = entry.path().filename().string();
    EXPECT_FALSE(name.starts_with(".tmp_")) << "unexpected tmp: " << name;
  }
}

TEST_F(DiskFlatBuilderTest, ReuseAfterFinishThrows) {
  DiskFlatBuilder b(4, MetricType::L2);
  std::vector<float> v(4, 1.0F);
  std::vector<uint64_t> l{1};
  b.add_batch(v.data(), l.data(), 1);
  ASSERT_NO_THROW(b.finish(seg_parent_ / "seg_00000001"));

  EXPECT_THROW(b.add_batch(v.data(), l.data(), 1), std::runtime_error);
  EXPECT_THROW(b.finish(seg_parent_ / "seg_00000002"), std::runtime_error);
}

TEST_F(DiskFlatBuilderTest, ExistingSegmentDirRefused) {
  auto seg_dir = seg_parent_ / "seg_00000001";
  std::filesystem::create_directories(seg_dir);

  DiskFlatBuilder b(4, MetricType::L2);
  std::vector<float> v(4, 1.0F);
  std::vector<uint64_t> l{1};
  b.add_batch(v.data(), l.data(), 1);
  EXPECT_THROW(b.finish(seg_dir), std::runtime_error);

  // Existing dir is left untouched.
  EXPECT_TRUE(std::filesystem::exists(seg_dir));
  EXPECT_TRUE(std::filesystem::is_directory(seg_dir));
}

// 5.26 — fault injection via RLIMIT_FSIZE. We size the test so that ids.u64.bin
// (8 bytes per row) fits under the limit but vectors.f32.bin (dim * 4 bytes per
// row) does not. Result: write of vectors.f32.bin fails after ids.u64.bin
// succeeded, exactly the spec's described scenario.
TEST_F(DiskFlatBuilderTest, MidWriteFailureCleansTmp) {
  constexpr uint32_t kDim = 256;     // vectors row = 1024 bytes
  constexpr uint64_t kN = 64;        // ids file = 512 bytes; vectors file = 65536 bytes
  auto vectors = make_random_vectors(kN, kDim);
  auto labels = sequential_labels(kN);

  // Ignore SIGXFSZ so EFBIG is returned to write() instead of killing the process.
  struct sigaction old_act {};
  struct sigaction ign_act {};
  ign_act.sa_handler = SIG_IGN;
  ::sigemptyset(&ign_act.sa_mask);
  ASSERT_EQ(::sigaction(SIGXFSZ, &ign_act, &old_act), 0);

  // RLIMIT_FSIZE: large enough for ids.u64.bin (512 B) plus slack, smaller
  // than vectors.f32.bin (64 KiB).
  struct rlimit old_lim {};
  struct rlimit new_lim {};
  ASSERT_EQ(::getrlimit(RLIMIT_FSIZE, &old_lim), 0);
  new_lim.rlim_cur = 4096;
  new_lim.rlim_max = old_lim.rlim_max;
  ASSERT_EQ(::setrlimit(RLIMIT_FSIZE, &new_lim), 0);

  DiskFlatBuilder b(kDim, MetricType::L2);
  bool threw = false;
  try {
    b.add_batch(vectors.data(), labels.data(), kN);
    (void)b.finish(seg_parent_ / "seg_00000001");
  } catch (const std::exception &) {
    threw = true;
  }

  // Restore rlimit and signal handler before any further test work.
  ::setrlimit(RLIMIT_FSIZE, &old_lim);
  ::sigaction(SIGXFSZ, &old_act, nullptr);

  EXPECT_TRUE(threw) << "expected write failure to throw under RLIMIT_FSIZE";

  // The final segment must not exist.
  EXPECT_FALSE(std::filesystem::exists(seg_parent_ / "seg_00000001"));

  // No .tmp_* dir should remain in the parent.
  for (const auto &entry : std::filesystem::directory_iterator(seg_parent_)) {
    auto name = entry.path().filename().string();
    EXPECT_FALSE(name.starts_with(".tmp_")) << "leftover tmp dir not cleaned: " << name;
  }
}

}  // namespace
}  // namespace alaya::disk
