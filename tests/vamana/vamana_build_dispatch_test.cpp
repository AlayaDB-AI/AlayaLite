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

#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "index/graph/vamana/build_dispatch.hpp"

namespace {

alaya::vamana::BuildVamanaParams valid_params_until_file_read() {
  alaya::vamana::BuildVamanaParams params =
      alaya::vamana::kDefaultVamanaBuildParams;
  params.data_path = "/tmp/alayalite-vamana-missing.fbin";
  params.output_path = "/tmp/alayalite-vamana-out.index";
  params.R = 64;
  params.L = 100;
  return params;
}

void expect_invalid_sampling_rate(float sampling_rate) {
  auto params = valid_params_until_file_read();
  params.sampling_rate = sampling_rate;
  try {
    alaya::vamana::build_vamana(params);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("sampling_rate"), std::string::npos);
  }
}

void expect_invalid_alpha(float alpha) {
  auto params = valid_params_until_file_read();
  params.alpha = alpha;
  try {
    alaya::vamana::build_vamana(params);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("alpha"), std::string::npos);
  }
}

void write_idmap(const std::filesystem::path& path, const std::vector<uint32_t>& ids) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  const uint32_t count = static_cast<uint32_t>(ids.size());
  const uint32_t stride = 1;
  out.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(&stride), sizeof(uint32_t));
  out.write(reinterpret_cast<const char*>(ids.data()),
            static_cast<std::streamsize>(ids.size()) * sizeof(uint32_t));
}

TEST(VamanaBuildDispatchTest, RejectsExplicitZeroSamplingRate) {
  expect_invalid_sampling_rate(0.0F);
}

TEST(VamanaBuildDispatchTest, RejectsSamplingRateAboveOne) {
  expect_invalid_sampling_rate(1.1F);
}

TEST(VamanaBuildDispatchTest, RejectsAlphaBelowOne) {
  expect_invalid_alpha(0.99F);
}

TEST(VamanaBuildDispatchTest, RejectsNonFiniteAlpha) {
  expect_invalid_alpha(std::numeric_limits<float>::quiet_NaN());
  expect_invalid_alpha(std::numeric_limits<float>::infinity());
}

TEST(VamanaBuildDispatchTest, MergeShardsWritesBareOutputFilename) {
  const auto cwd = std::filesystem::current_path();
  const auto root = std::filesystem::temp_directory_path() / "alaya_vamana_merge_bare_output";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root);

  const auto shard = root / "shard.index";
  const auto idmap = root / "shard.idmap";
  alaya::vamana::save_graph({{1}, {0}}, shard, /*max_degree=*/1, /*start=*/0);
  write_idmap(idmap, {0, 1});

  std::filesystem::current_path(root);
  EXPECT_NO_THROW({
    (void)alaya::vamana::merge_shards({shard}, {idmap}, "merged.index", /*R=*/1, /*medoid=*/0,
                                      /*seed=*/42);
  });
  std::filesystem::current_path(cwd);

  EXPECT_TRUE(std::filesystem::is_regular_file(root / "merged.index"));
  EXPECT_TRUE(std::filesystem::is_regular_file(root / "merged.index_medoids.bin"));
  std::filesystem::remove_all(root, ec);
}

}  // namespace
