// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/graph/diskann/diskann_index.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <random>
#include <vector>

namespace {

using alaya::diskann::DiskANNBuildParams;
using alaya::diskann::DiskANNIndex;
using alaya::diskann::DiskANNLoadParams;
using alaya::diskann::DiskANNUpdateIO;

class DiskANNPortableUpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<uint64_t> counter{0};
    dir_ = std::filesystem::temp_directory_path() /
           ("diskann_portable_update_" + std::to_string(counter.fetch_add(1)));
    vectors_.resize(kCount * kDim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (float &value : vectors_) {
      value = distribution(rng);
    }
    labels_.resize(kCount);
    std::iota(labels_.begin(), labels_.end(), uint64_t{1000});

    DiskANNBuildParams build_params;
    build_params.R = 16;
    build_params.L = 32;
    DiskANNIndex::build(dir_.string(), vectors_.data(), labels_.data(), kCount, kDim, build_params);
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(dir_, error);
  }

  static constexpr uint64_t kCount = 96;
  static constexpr uint64_t kDim = 16;
  std::filesystem::path dir_;
  std::vector<float> vectors_;
  std::vector<uint64_t> labels_;
};

TEST_F(DiskANNPortableUpdateTest, ExternalIdUpdatesPersistAcrossWritableAndReadOnlyReloads) {
  DiskANNLoadParams update_params;
  update_params.updatable = true;
  update_params.update_io = DiskANNUpdateIO::kBlocking;
  update_params.num_threads = 2;
  update_params.update_insert_threads = 2;
  update_params.update_reconnect_threads = 2;

  {
    DiskANNIndex index;
    index.load(dir_.string(), update_params);
    const std::vector<float> inserted(vectors_.begin(), vectors_.begin() + kDim);
    index.insert(inserted.data(), 9000);
    EXPECT_TRUE(index.contains_label(9000));
    index.remove_by_label(1001);
    EXPECT_FALSE(index.contains_label(1001));
    index.flush();
  }

  DiskANNIndex read_only;
  read_only.load(dir_.string());
  EXPECT_TRUE(read_only.contains_label(9000));
  EXPECT_FALSE(read_only.contains_label(1001));
}

}  // namespace
