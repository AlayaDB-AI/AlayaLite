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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_reader.hpp"
#include "index/graph/vamana/vamana_writer.hpp"

namespace {

// Per-process counter so each test gets a unique temp path even when the
// suite runs in parallel.
std::atomic<uint64_t> g_path_counter{0};

std::filesystem::path unique_temp_path(const std::string &stem) {
  const uint64_t n = g_path_counter.fetch_add(1, std::memory_order_relaxed);
  std::string name = stem + "_pid" + std::to_string(::getpid()) + "_" + std::to_string(n) +
                     ".index";
  return std::filesystem::temp_directory_path() / name;
}

std::vector<float> make_data(uint32_t n, uint32_t dim, uint64_t seed) {
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> out(static_cast<size_t>(n) * dim);
  for (auto &v : out) {
    v = dist(rng);
  }
  return out;
}

struct BuildResult {
  std::filesystem::path path;
  std::vector<float> data;
  uint32_t medoid;
  std::vector<std::vector<uint32_t>> graph;
  uint32_t max_degree;
};

BuildResult build_and_save(uint32_t n,
                           uint32_t dim,
                           uint32_t R,
                           uint32_t L,
                           float alpha,
                           uint64_t seed,
                           const std::string &stem) {
  BuildResult br;
  br.data = make_data(n, dim, seed);
  alaya::vamana::VamanaBuildParams params{};
  params.R = R;
  params.L = L;
  params.alpha = alpha;
  params.num_threads = 1;
  params.seed = seed;
  alaya::vamana::VamanaBuilder builder(br.data.data(), n, dim, params);
  builder.build();
  br.medoid = builder.medoid();
  br.graph = builder.graph();
  br.max_degree = R;
  br.path = unique_temp_path(stem);
  alaya::vamana::save_graph(br.graph, br.path, R, br.medoid, /*frozen_pts=*/0);
  return br;
}

// Walk the file's per-node records to find the byte offset of node
// `node_id`'s `neighbor_index`-th neighbor field. Used by hand-edit tests
// that corrupt a single neighbor id without touching the writer's
// header.
uint64_t neighbor_byte_offset(const std::filesystem::path &path,
                              uint32_t node_id,
                              uint32_t neighbor_index) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("neighbor_byte_offset: cannot open " + path.string());
  }
  in.seekg(24, std::ios::beg);
  uint64_t offset = 24;
  for (uint32_t i = 0; i < node_id; ++i) {
    uint32_t k = 0;
    in.read(reinterpret_cast<char *>(&k), sizeof(uint32_t));
    offset += sizeof(uint32_t) + static_cast<uint64_t>(k) * sizeof(uint32_t);
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  }
  return offset + sizeof(uint32_t) + static_cast<uint64_t>(neighbor_index) * sizeof(uint32_t);
}

void overwrite_uint32_at(const std::filesystem::path &path, uint64_t offset, uint32_t value) {
  std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
  f.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
  f.write(reinterpret_cast<const char *>(&value), sizeof(uint32_t));
  f.close();
}

void overwrite_uint64_at(const std::filesystem::path &path, uint64_t offset, uint64_t value) {
  std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
  f.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
  f.write(reinterpret_cast<const char *>(&value), sizeof(uint64_t));
  f.close();
}

// Run `fn`, capture the runtime_error message. Fail the test if `fn`
// did not throw a runtime_error.
template <typename Fn>
std::string capture_runtime_error_message(Fn &&fn) {
  try {
    fn();
  } catch (const std::runtime_error &e) {
    return std::string(e.what());
  } catch (...) {
    ADD_FAILURE() << "Expected std::runtime_error, got a different exception type";
    return {};
  }
  ADD_FAILURE() << "Expected std::runtime_error, but no exception was thrown";
  return {};
}

}  // namespace

class VamanaReaderTest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (const auto &p : owned_paths_) {
      std::error_code ec;
      std::filesystem::remove(p, ec);
    }
  }

  void track(const std::filesystem::path &p) { owned_paths_.push_back(p); }

  std::vector<std::filesystem::path> owned_paths_;
};

// 9.1 — round-trip equivalence with the existing builder.
TEST_F(VamanaReaderTest, LoadsWriterOutput) {
  const uint32_t N = 128, dim = 8, R = 16, L = 64;
  BuildResult br =
      build_and_save(N, dim, R, L, /*alpha=*/1.2f, /*seed=*/42, "vamana_reader_round_trip");
  track(br.path);

  alaya::vamana::VamanaReader reader{br.path};
  EXPECT_EQ(reader.max_degree(), R);
  EXPECT_EQ(reader.start(), br.medoid);
  EXPECT_EQ(reader.frozen_pts(), 0u);
  EXPECT_EQ(reader.num_nodes(), static_cast<size_t>(N));

  ASSERT_EQ(reader.graph().size(), br.graph.size());
  for (size_t i = 0; i < br.graph.size(); ++i) {
    std::unordered_set<uint32_t> expected(br.graph[i].begin(), br.graph[i].end());
    std::unordered_set<uint32_t> actual(reader.graph()[i].begin(), reader.graph()[i].end());
    EXPECT_EQ(expected, actual) << "node " << i;
  }
}

TEST_F(VamanaReaderTest, SaveGraphWritesBareFilename) {
  const auto cwd = std::filesystem::current_path();
  const auto root = std::filesystem::temp_directory_path() / "alaya_vamana_writer_bare_output";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  std::filesystem::create_directories(root);

  std::filesystem::current_path(root);
  EXPECT_NO_THROW(
      alaya::vamana::save_graph({{1}, {0}}, "bare.index", /*max_degree=*/1, /*start=*/0));
  std::filesystem::current_path(cwd);

  const auto path = root / "bare.index";
  ASSERT_TRUE(std::filesystem::is_regular_file(path));
  alaya::vamana::VamanaReader reader{path};
  EXPECT_EQ(reader.num_nodes(), 2u);
  EXPECT_EQ(reader.max_degree(), 1u);
  EXPECT_EQ(reader.start(), 0u);
  EXPECT_EQ(reader.graph()[0], std::vector<uint32_t>{1});
  EXPECT_EQ(reader.graph()[1], std::vector<uint32_t>{0});

  std::filesystem::remove_all(root, ec);
}

// 9.2 — truncated by 1 byte.
TEST_F(VamanaReaderTest, RejectsTruncatedFile) {
  BuildResult br = build_and_save(64, 4, 8, 32, 1.2f, 7, "vamana_reader_truncated");
  track(br.path);

  const auto orig_size = std::filesystem::file_size(br.path);
  std::filesystem::resize_file(br.path, orig_size - 1);

  std::string msg = capture_runtime_error_message(
      [&] { alaya::vamana::VamanaReader reader{br.path}; });
  EXPECT_NE(msg.find(std::to_string(orig_size - 1)), std::string::npos)
      << "msg should reference actual size: " << msg;
}

// 9.3 — trailing bytes appended after a valid file.
TEST_F(VamanaReaderTest, RejectsSizeMismatchTrailingBytes) {
  BuildResult br = build_and_save(64, 4, 8, 32, 1.2f, 11, "vamana_reader_trailing");
  track(br.path);

  const auto orig_size = std::filesystem::file_size(br.path);
  {
    std::ofstream f(br.path, std::ios::binary | std::ios::app);
    const std::vector<char> garbage(8, '\xAB');
    f.write(garbage.data(), 8);
  }
  ASSERT_EQ(std::filesystem::file_size(br.path), orig_size + 8);

  std::string msg = capture_runtime_error_message(
      [&] { alaya::vamana::VamanaReader reader{br.path}; });
  EXPECT_NE(msg.find("trailing"), std::string::npos)
      << "msg should mention trailing bytes: " << msg;
}

// 9.4 — neighbor id corrupted to 0xFFFFFFFF.
TEST_F(VamanaReaderTest, RejectsInvalidNeighborId) {
  BuildResult br = build_and_save(64, 4, 8, 32, 1.2f, 13, "vamana_reader_bad_neighbor");
  track(br.path);

  ASSERT_GE(br.graph[0].size(), 1u);
  const uint64_t off = neighbor_byte_offset(br.path, /*node=*/0, /*neighbor_index=*/0);
  overwrite_uint32_at(br.path, off, 0xFFFFFFFFu);

  std::string msg = capture_runtime_error_message(
      [&] { alaya::vamana::VamanaReader reader{br.path}; });
  EXPECT_NE(msg.find("node 0"), std::string::npos) << msg;
  EXPECT_NE(msg.find("4294967295"), std::string::npos) << msg;
}

// 9.5 — k of node 0 corrupted to max_degree + 1.
TEST_F(VamanaReaderTest, RejectsDegreeOverMaxDegree) {
  const uint32_t R = 8;
  BuildResult br = build_and_save(64, 4, R, 32, 1.2f, 17, "vamana_reader_over_degree");
  track(br.path);

  // k field of node 0 lives at byte offset 24.
  const uint32_t bad_k = R + 1;
  overwrite_uint32_at(br.path, 24, bad_k);

  std::string msg = capture_runtime_error_message(
      [&] { alaya::vamana::VamanaReader reader{br.path}; });
  EXPECT_NE(msg.find("node 0"), std::string::npos) << msg;
  EXPECT_NE(msg.find(std::to_string(bad_k)), std::string::npos) << msg;
  EXPECT_NE(msg.find(std::to_string(R)), std::string::npos) << msg;
}

// 9.6 — neighbor of node 5 hand-edited to 5 (self-loop).
TEST_F(VamanaReaderTest, RejectsSelfLoop) {
  const uint32_t N = 64;
  BuildResult br = build_and_save(N, 4, 8, 32, 1.2f, 19, "vamana_reader_self_loop");
  track(br.path);

  ASSERT_GE(br.graph[5].size(), 1u);
  const uint64_t off = neighbor_byte_offset(br.path, /*node=*/5, /*neighbor_index=*/0);
  overwrite_uint32_at(br.path, off, 5u);

  std::string msg = capture_runtime_error_message(
      [&] { alaya::vamana::VamanaReader reader{br.path}; });
  EXPECT_NE(msg.find("node 5"), std::string::npos) << msg;
  EXPECT_NE(msg.find("self-loop"), std::string::npos) << msg;
}

// 9.7 — header's max_observed_degree set to 0.
TEST_F(VamanaReaderTest, RejectsZeroMaxDegree) {
  BuildResult br = build_and_save(64, 4, 8, 32, 1.2f, 23, "vamana_reader_zero_degree");
  track(br.path);

  // max_observed_degree lives at byte offset 8 (uint32_t).
  overwrite_uint32_at(br.path, 8, 0u);

  std::string msg = capture_runtime_error_message(
      [&] { alaya::vamana::VamanaReader reader{br.path}; });
  EXPECT_NE(msg.find("max_observed_degree"), std::string::npos) << msg;
}

// 9.8 — header's frozen_pts set to 1.
TEST_F(VamanaReaderTest, RejectsNonZeroFrozenPts) {
  BuildResult br = build_and_save(64, 4, 8, 32, 1.2f, 29, "vamana_reader_frozen");
  track(br.path);

  // frozen_pts lives at byte offset 16 (uint64_t).
  overwrite_uint64_at(br.path, 16, 1u);

  std::string msg = capture_runtime_error_message(
      [&] { alaya::vamana::VamanaReader reader{br.path}; });
  EXPECT_NE(msg.find("frozen_pts"), std::string::npos) << msg;
}

// 9.9 — header's start set out of range.
TEST_F(VamanaReaderTest, RejectsStartOutOfRange) {
  const uint32_t N = 64;
  BuildResult br = build_and_save(N, 4, 8, 32, 1.2f, 31, "vamana_reader_bad_start");
  track(br.path);

  // start lives at byte offset 12 (uint32_t).
  const uint32_t bad_start = N + 5;
  overwrite_uint32_at(br.path, 12, bad_start);

  std::string msg = capture_runtime_error_message(
      [&] { alaya::vamana::VamanaReader reader{br.path}; });
  EXPECT_NE(msg.find(std::to_string(bad_start)), std::string::npos) << msg;
  EXPECT_NE(msg.find(std::to_string(N)), std::string::npos) << msg;
}

// 9.10 — non-copyable contract is enforced by the type system.
TEST_F(VamanaReaderTest, IsNoncopyable) {
  static_assert(!std::is_copy_constructible_v<alaya::vamana::VamanaReader>,
                "VamanaReader must not be copy-constructible");
  static_assert(!std::is_copy_assignable_v<alaya::vamana::VamanaReader>,
                "VamanaReader must not be copy-assignable");
  SUCCEED();
}

// 9.11 — file deletion after construction does not affect the reader.
TEST_F(VamanaReaderTest, SurvivesFileDeletion) {
  const uint32_t N = 64;
  BuildResult br = build_and_save(N, 4, 8, 32, 1.2f, 37, "vamana_reader_delete");
  // Don't track for teardown — we delete inside the test.

  alaya::vamana::VamanaReader reader{br.path};
  std::vector<uint32_t> neighbors_before = reader.graph()[0];

  std::error_code ec;
  std::filesystem::remove(br.path, ec);
  ASSERT_FALSE(std::filesystem::exists(br.path));

  EXPECT_EQ(reader.graph()[0], neighbors_before);
  EXPECT_EQ(reader.num_nodes(), static_cast<size_t>(N));
}

// Spec scenario: "Missing file is rejected".
TEST_F(VamanaReaderTest, RejectsMissingFile) {
  const auto path = unique_temp_path("vamana_reader_missing");
  // Don't track for teardown — file shouldn't exist.
  ASSERT_FALSE(std::filesystem::exists(path));

  std::string msg = capture_runtime_error_message(
      [&] { alaya::vamana::VamanaReader reader{path}; });
  EXPECT_NE(msg.find(path.string()), std::string::npos)
      << "msg should reference the offending path: " << msg;
}

// Spec follow-on (tasks.md 5.1): empty file is rejected at the header
// truncation pre-check with a message identifying the actual size.
TEST_F(VamanaReaderTest, RejectsEmptyFile) {
  const auto path = unique_temp_path("vamana_reader_empty");
  track(path);
  {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
  }
  ASSERT_EQ(std::filesystem::file_size(path), 0u);

  std::string msg = capture_runtime_error_message(
      [&] { alaya::vamana::VamanaReader reader{path}; });
  EXPECT_NE(msg.find("header truncated"), std::string::npos) << msg;
  EXPECT_NE(msg.find("0"), std::string::npos) << msg;
}

// Spec scenario: "Zero-degree node is rejected". Hand-edit node 0's
// k field to 0 directly; this is caught by step 4.2 before any neighbor
// bytes are consumed.
TEST_F(VamanaReaderTest, RejectsZeroDegreeNode) {
  BuildResult br = build_and_save(64, 4, 8, 32, 1.2f, 41, "vamana_reader_zero_node");
  track(br.path);

  // k field of node 0 lives at byte offset 24.
  overwrite_uint32_at(br.path, 24, 0u);

  std::string msg = capture_runtime_error_message(
      [&] { alaya::vamana::VamanaReader reader{br.path}; });
  EXPECT_NE(msg.find("node 0"), std::string::npos) << msg;
  EXPECT_NE(msg.find("zero out-degree"), std::string::npos) << msg;
}

// Spec scenario: "Mid-record truncation is rejected". Construct a valid
// 2-node graph hand-written via save_graph (so we control max_degree
// independently of actual k values), then inflate node 1's k from 1 to
// 2. The expected_file_size is unchanged (avoids step 3.3); the inflated
// k stays under max_degree=4 (avoids step 4.2 over-degree); the byte
// budget after reading the inflated k has only 4 remaining bytes versus
// the 8 the record now claims, triggering the mid-record bound check
// in step 4.3 with a message identifying the offset.
TEST_F(VamanaReaderTest, RejectsMidRecordTruncation) {
  const auto path = unique_temp_path("vamana_reader_mid_record");
  track(path);

  std::vector<std::vector<uint32_t>> graph = {{1u}, {0u}};
  alaya::vamana::save_graph(graph, path, /*max_degree=*/4u, /*start=*/0u,
                            /*frozen_pts=*/0u);

  // Layout: 24 (header) + 4 (node0 k) + 4 (node0 neighbors[0]) = 32.
  // Node 1's k field begins at offset 32.
  overwrite_uint32_at(path, 32u, 2u);

  std::string msg = capture_runtime_error_message(
      [&] { alaya::vamana::VamanaReader reader{path}; });
  EXPECT_NE(msg.find("node 1"), std::string::npos) << msg;
  EXPECT_NE(msg.find("mid-record truncation"), std::string::npos) << msg;
}
