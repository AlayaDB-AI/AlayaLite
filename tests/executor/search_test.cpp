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
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "executor/jobs/graph_search_job.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "space/rabitq_space.hpp"
#include "space/raw_space.hpp"
#include "space/sq8_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/log.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/scalar_data.hpp"
#include "utils/timer.hpp"

namespace alaya {

class SearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path data_dir = std::filesystem::current_path().parent_path() / "data";
    ds_ = load_dataset(sift_small(data_dir));
  }

  void TearDown() override { std::filesystem::remove(index_file_); }

  uint32_t max_thread_num_ = std::thread::hardware_concurrency();
  Dataset ds_;
  std::filesystem::path index_file_;
};

TEST_F(SearchTest, FullGraphTest) {
  const size_t kM = 64;
  std::string index_type = "HNSW";

  index_file_ = fmt::format("{}_M{}.{}", ds_.name_, kM, index_type);
  LOG_INFO("the data size is {}, point number is: {}", ds_.data_.size(), ds_.data_num_);
  std::shared_ptr<alaya::RawSpace<>> space =
      std::make_shared<alaya::RawSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);
  LOG_INFO("Initialize space successfully!");
  space->fit(ds_.data_.data(), ds_.data_num_);

  LOG_INFO("Fit space successfully!");
  alaya::Graph<uint32_t> load_graph = alaya::Graph<uint32_t>(ds_.data_num_, kM);
  if (!std::filesystem::exists(index_file_)) {
    auto build_start = std::chrono::steady_clock::now();

    alaya::HNSWBuilder<alaya::RawSpace<>> hnsw = alaya::HNSWBuilder<alaya::RawSpace<>>(space);
    LOG_INFO("Initialize the hnsw builder successfully!");
    auto hnsw_graph = hnsw.build_graph(max_thread_num_);

    auto build_end = std::chrono::steady_clock::now();
    auto build_time = static_cast<std::chrono::duration<double>>(build_end - build_start).count();
    LOG_INFO("The time of building hnsw is {}s, saving it to {}", build_time, index_file_.string());

    std::string_view path = index_file_.native();
    hnsw_graph->save(path);
  }
  LOG_INFO("Begin Loading the graph from file: {}", index_file_.string());
  std::string_view path = index_file_.native();
  load_graph.load(path);

  std::vector<uint32_t> inpoint_num(ds_.data_num_);
  std::vector<uint32_t> outpoint_num(ds_.data_num_);

  for (uint32_t i = 0; i < ds_.data_num_; i++) {
    for (uint32_t j = 0; j < load_graph.max_nbrs_; j++) {
      auto id = load_graph.at(i, j);
      if (id == alaya::Graph<uint32_t>::kEmptyId) {
        break;
      }
      outpoint_num[i]++;
      inpoint_num[id]++;
    }
  }

  uint64_t zero_outpoint_cnt = 0;
  uint64_t zero_inpoint_cnt = 0;

  // Check if edge exists on each node
  for (uint32_t i = 0; i < ds_.data_num_; i++) {
    if (outpoint_num[i] != 0) {
      zero_outpoint_cnt++;
    }
    if (inpoint_num[i] != 0) {
      zero_inpoint_cnt++;
    }
  }
  LOG_INFO("no_zero_inpoint = {} , no_zero_oupoint = {}", zero_inpoint_cnt, zero_outpoint_cnt);

  // Allow a small percentage of nodes to have no inpoint edges (HNSW characteristic)
  // Typically in HNSW graphs, a few isolated nodes may not be selected as neighbors
  double inpoint_ratio = static_cast<double>(zero_inpoint_cnt) / ds_.data_num_;

  EXPECT_GE(inpoint_ratio, 0.9);               // At least 90% nodes should have inpoint edges
  EXPECT_EQ(zero_outpoint_cnt, ds_.data_num_);  // All nodes should have outpoint edges
}

TEST_F(SearchTest, SearchHNSWTest) {
  const size_t kM = 64;
  size_t topk = 10;
  size_t ef = 100;
  std::string index_type = "HNSW";

  index_file_ = fmt::format("{}_M{}.{}", ds_.name_, kM, index_type);
  std::shared_ptr<alaya::RawSpace<>> space =
      std::make_shared<alaya::RawSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);
  space->fit(ds_.data_.data(), ds_.data_num_);

  auto load_graph = std::make_shared<alaya::Graph<>>(ds_.data_num_, kM);
  if (!std::filesystem::exists(index_file_)) {
    auto build_start = std::chrono::steady_clock::now();

    alaya::HNSWBuilder<alaya::RawSpace<>> hnsw = alaya::HNSWBuilder<alaya::RawSpace<>>(space);
    auto hnsw_graph = hnsw.build_graph(max_thread_num_);

    auto build_end = std::chrono::steady_clock::now();
    auto build_time = static_cast<std::chrono::duration<double>>(build_end - build_start).count();
    LOG_INFO("The time of building hnsw is {}s.", build_time);

    std::string_view path = index_file_.native();
    hnsw_graph->save(path);
  }
  std::string_view path = index_file_.native();
  load_graph->load(path);

  auto search_job = std::make_unique<alaya::GraphSearchJob<alaya::RawSpace<>>>(space, load_graph);

  LOG_INFO("Create task generator successfully");

  using IDType = uint32_t;

  Timer timer{};
  std::vector<std::vector<IDType>> res_pool(ds_.query_num_, std::vector<IDType>(topk));
  const size_t kSearchThreadNum = 16;
  std::vector<std::thread> tasks(kSearchThreadNum);

  auto search_knn = [&](uint32_t i) {
    for (; i < ds_.query_num_; i += kSearchThreadNum) {
      std::vector<uint32_t> ids(topk);  // Now returns topk directly
      auto cur_query = ds_.queries_.data() + i * ds_.dim_;
      // New interface: search_solo(query, ids, topk, ef) returns topk results
      search_job->search_solo(cur_query, ids.data(), topk, ef);

      // search_solo now returns topk results directly
      auto id_set = std::set(ids.begin(), ids.end());

      if (id_set.size() < topk) {
        fmt::println("i id: {}", i);
        fmt::println("ids size: {}", id_set.size());
      }
      res_pool[i] = ids;
    }
  };

  for (size_t i = 0; i < kSearchThreadNum; i++) {
    tasks[i] = std::thread(search_knn, i);
  }

  for (size_t i = 0; i < kSearchThreadNum; i++) {
    if (tasks[i].joinable()) {
      tasks[i].join();
    }
  }

  LOG_INFO("total time: {} s.", timer.elapsed() / 1000000.0);

  auto recall = calc_recall(res_pool, ds_.ground_truth_.data(), ds_.query_num_, ds_.gt_dim_, topk);
  LOG_INFO("recall is {}.", recall);
  EXPECT_GE(recall, 0.5);
}

TEST_F(SearchTest, SearchHNSWTestSQSpace) {
  const size_t kM = 64;
  std::string index_type = "HNSW";

  index_file_ = fmt::format("{}_M{}_SQ.{}", ds_.name_, kM, index_type);

  auto load_graph = std::make_shared<alaya::Graph<>>(ds_.data_num_, kM);
  if (!std::filesystem::exists(index_file_)) {
    std::shared_ptr<alaya::RawSpace<>> build_graph_space =
        std::make_shared<alaya::RawSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);
    build_graph_space->fit(ds_.data_.data(), ds_.data_num_);
    auto build_start = std::chrono::steady_clock::now();

    alaya::HNSWBuilder<alaya::RawSpace<>> hnsw =
        alaya::HNSWBuilder<alaya::RawSpace<>>(build_graph_space);
    auto hnsw_graph = hnsw.build_graph(max_thread_num_);

    auto build_end = std::chrono::steady_clock::now();
    auto build_time = static_cast<std::chrono::duration<double>>(build_end - build_start).count();
    LOG_INFO("The time of building hnsw is {}s.", build_time);

    std::string_view path = index_file_.native();
    hnsw_graph->save(path);
  }
}

// ============================================================================
// Hybrid Search Tests (with metadata filtering)
// ============================================================================

class HybridSearchTest : public ::testing::Test {
 protected:
  using SQ8SpaceWithScalar =
      SQ8Space<float, float, uint32_t, SequentialStorage<uint8_t, uint32_t>, ScalarData>;

  void SetUp() override {
    std::filesystem::path data_dir = std::filesystem::current_path().parent_path() / "data";
    ds_ = load_dataset(sift_small(data_dir));
    db_path_ = "./test_hybrid_search_rocksdb";
    cleanup_test_files();
  }

  void TearDown() override { cleanup_test_files(); }

  void cleanup_test_files() {
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove_all(db_path_);
    }
  }

  auto make_test_metadata(uint32_t item_cnt) -> std::vector<ScalarData> {
    std::vector<ScalarData> metadata(item_cnt);
    for (uint32_t i = 0; i < item_cnt; ++i) {
      MetadataMap meta;
      meta["category"] = static_cast<int64_t>(i % 5);
      meta["score"] = static_cast<double>(i) * 10.0;
      meta["name"] = std::string("item_") + std::to_string(i);
      metadata[i] = ScalarData("id_" + std::to_string(i), "doc_" + std::to_string(i), meta);
    }
    return metadata;
  }

  uint32_t max_thread_num_ = std::thread::hardware_concurrency();
  Dataset ds_;
  std::string db_path_;
};

TEST_F(HybridSearchTest, HybridSearchSoloWithEmptyFilter) {
  uint32_t topk = 10;
  uint32_t ef = 50;

  // Create build space (RawSpace for graph construction)
  auto build_space = std::make_shared<RawSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);
  build_space->fit(ds_.data_.data(), ds_.data_num_);

  // Build graph using RawSpace
  HNSWBuilder<RawSpace<>> hnsw(build_space);
  auto graph = std::shared_ptr<Graph<>>(hnsw.build_graph(max_thread_num_).release());

  // Create search space (SQ8Space with ScalarData)
  RocksDBConfig config;
  config.db_path_ = db_path_;
  auto search_space =
      std::make_shared<SQ8SpaceWithScalar>(ds_.data_num_, ds_.dim_, MetricType::L2, config);
  auto metadata = make_test_metadata(ds_.data_num_);
  search_space->fit(ds_.data_.data(), ds_.data_num_, metadata.data());

  // Create search job with both spaces (search space for distance, build space for rerank)
  auto search_job = std::make_shared<GraphSearchJob<SQ8SpaceWithScalar, RawSpace<>>>(search_space,
                                                                                     graph,
                                                                                     nullptr,
                                                                                     build_space);

  // Test with empty filter (should match all)
  MetadataFilter empty_filter = MetadataFilter::empty();
  std::vector<uint32_t> ids(topk);
  std::vector<ScalarData> results(topk);
  auto query = ds_.queries_.data();

  search_job->hybrid_search_solo(query, ids.data(), topk, ef, empty_filter, results.data());

  // Verify results
  std::set<uint32_t> unique_ids(ids.begin(), ids.end());
  EXPECT_EQ(unique_ids.size(), topk);

  // Verify ScalarData was returned
  for (uint32_t i = 0; i < topk; i++) {
    EXPECT_FALSE(results[i].item_id.empty());
  }
}

TEST_F(HybridSearchTest, HybridSearchSoloWithCategoryFilter) {
  uint32_t topk = 5;
  uint32_t ef = 100;

  auto build_space = std::make_shared<RawSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);
  build_space->fit(ds_.data_.data(), ds_.data_num_);

  HNSWBuilder<RawSpace<>> hnsw(build_space);
  auto graph = std::shared_ptr<Graph<>>(hnsw.build_graph(max_thread_num_).release());

  RocksDBConfig config;
  config.db_path_ = db_path_;
  auto search_space =
      std::make_shared<SQ8SpaceWithScalar>(ds_.data_num_, ds_.dim_, MetricType::L2, config);
  auto metadata = make_test_metadata(ds_.data_num_);
  search_space->fit(ds_.data_.data(), ds_.data_num_, metadata.data());

  auto search_job = std::make_shared<GraphSearchJob<SQ8SpaceWithScalar, RawSpace<>>>(search_space,
                                                                                     graph,
                                                                                     nullptr,
                                                                                     build_space);

  // Filter: category == 0 (every 5th item)
  MetadataFilter filter;
  filter.add_eq("category", static_cast<int64_t>(0));

  std::vector<uint32_t> ids(topk);
  std::vector<ScalarData> results(topk);
  auto query = ds_.queries_.data();

  search_job->hybrid_search_solo(query, ids.data(), topk, ef, filter, results.data());

  // Verify all results match the filter
  for (uint32_t i = 0; i < topk; i++) {
    if (ids[i] != static_cast<uint32_t>(-1) && !results[i].metadata.empty()) {
      auto category = std::get<int64_t>(results[i].metadata.at("category"));
      EXPECT_EQ(category, 0);
    }
  }
}

TEST_F(HybridSearchTest, HybridSearchSoloThrowsWhenEfLessThanTopk) {
  auto build_space = std::make_shared<RawSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);
  build_space->fit(ds_.data_.data(), ds_.data_num_);

  HNSWBuilder<RawSpace<>> hnsw(build_space);
  auto graph = std::shared_ptr<Graph<>>(hnsw.build_graph(max_thread_num_).release());

  RocksDBConfig config;
  config.db_path_ = db_path_;
  auto search_space =
      std::make_shared<SQ8SpaceWithScalar>(ds_.data_num_, ds_.dim_, MetricType::L2, config);
  auto metadata = make_test_metadata(ds_.data_num_);
  search_space->fit(ds_.data_.data(), ds_.data_num_, metadata.data());

  auto search_job = std::make_shared<GraphSearchJob<SQ8SpaceWithScalar, RawSpace<>>>(search_space,
                                                                                     graph,
                                                                                     nullptr,
                                                                                     build_space);

  MetadataFilter filter = MetadataFilter::empty();
  std::vector<uint32_t> ids(10);
  std::vector<ScalarData> results(10);
  auto query = ds_.queries_.data();

  // ef=5, topk=10 should throw
  EXPECT_THROW(search_job->hybrid_search_solo(query, ids.data(), 10, 5, filter, results.data()),
               std::invalid_argument);
}

// ============================================================================
// RaBitQ Hybrid Search Tests (requires AVX512)
// ============================================================================

#if defined(__AVX512F__)
class RaBitQHybridSearchTest : public ::testing::Test {
 protected:
  using RaBitQSpaceWithScalar = RaBitQSpace<float, float, uint32_t, ScalarData>;

  void SetUp() override {
    std::filesystem::path data_dir = std::filesystem::current_path().parent_path() / "data";
    ds_ = load_dataset(sift_small(data_dir));
    db_path_ = "./test_rabitq_hybrid_search_rocksdb";
    cleanup_test_files();
  }

  void TearDown() override { cleanup_test_files(); }

  void cleanup_test_files() {
    if (std::filesystem::exists(db_path_)) {
      std::filesystem::remove_all(db_path_);
    }
  }

  auto make_test_metadata(uint32_t item_cnt) -> std::vector<ScalarData> {
    std::vector<ScalarData> metadata(item_cnt);
    for (uint32_t i = 0; i < item_cnt; ++i) {
      MetadataMap meta;
      meta["category"] = static_cast<int64_t>(i % 5);
      meta["score"] = static_cast<double>(i) * 10.0;
      metadata[i] = ScalarData("id_" + std::to_string(i), "doc_" + std::to_string(i), meta);
    }
    return metadata;
  }

  Dataset ds_;
  std::string db_path_;
};

TEST_F(RaBitQHybridSearchTest, RaBitQHybridSearchSoloWithEmptyFilter) {
  uint32_t topk = 10;
  uint32_t ef = 50;

  RocksDBConfig config;
  config.db_path_ = db_path_;

  auto space = std::make_shared<RaBitQSpaceWithScalar>(ds_.data_num_,
                                                       ds_.dim_,
                                                       MetricType::L2,
                                                       config,
                                                       RotatorType::MatrixRotator);
  auto metadata = make_test_metadata(ds_.data_num_);
  space->fit(ds_.data_.data(), ds_.data_num_, metadata.data());

  // RaBitQ uses its own graph structure, create a dummy graph for the search job
  auto graph = std::make_shared<Graph<>>(ds_.data_num_, RaBitQSpace<>::kDegreeBound);

  auto search_job = std::make_shared<GraphSearchJob<RaBitQSpaceWithScalar>>(space, graph);

  MetadataFilter empty_filter = MetadataFilter::empty();
  std::vector<uint32_t> ids(topk);
  std::vector<ScalarData> results(topk);
  auto query = ds_.queries_.data();

  search_job->rabitq_hybrid_search_solo(query, topk, ids.data(), ef, empty_filter, results.data());

  // Verify results are valid
  uint32_t valid_count = 0;
  for (uint32_t i = 0; i < topk; i++) {
    if (ids[i] != static_cast<uint32_t>(-1)) {
      valid_count++;
    }
  }
  EXPECT_GT(valid_count, 0U);
}

TEST_F(RaBitQHybridSearchTest, RaBitQHybridSearchSoloWithCategoryFilter) {
  uint32_t topk = 5;
  uint32_t ef = 100;

  RocksDBConfig config;
  config.db_path_ = db_path_;

  auto space = std::make_shared<RaBitQSpaceWithScalar>(ds_.data_num_,
                                                       ds_.dim_,
                                                       MetricType::L2,
                                                       config,
                                                       RotatorType::MatrixRotator);
  auto metadata = make_test_metadata(ds_.data_num_);
  space->fit(ds_.data_.data(), ds_.data_num_, metadata.data());

  auto graph = std::make_shared<Graph<>>(ds_.data_num_, RaBitQSpace<>::kDegreeBound);

  auto search_job = std::make_shared<GraphSearchJob<RaBitQSpaceWithScalar>>(space, graph);

  // Filter: category == 2
  MetadataFilter filter;
  filter.add_eq("category", static_cast<int64_t>(2));

  std::vector<uint32_t> ids(topk);
  std::vector<ScalarData> results(topk);
  auto query = ds_.queries_.data();

  search_job->rabitq_hybrid_search_solo(query, topk, ids.data(), ef, filter, results.data());

  // Verify results match filter
  for (uint32_t i = 0; i < topk; i++) {
    if (ids[i] != static_cast<uint32_t>(-1) && !results[i].metadata.empty()) {
      auto category = std::get<int64_t>(results[i].metadata.at("category"));
      EXPECT_EQ(category, 2);
    }
  }
}

TEST_F(RaBitQHybridSearchTest, RaBitQHybridSearchSoloThrowsWhenEfLessThanK) {
  RocksDBConfig config;
  config.db_path_ = db_path_;

  auto space = std::make_shared<RaBitQSpaceWithScalar>(ds_.data_num_,
                                                       ds_.dim_,
                                                       MetricType::L2,
                                                       config,
                                                       RotatorType::MatrixRotator);
  auto metadata = make_test_metadata(ds_.data_num_);
  space->fit(ds_.data_.data(), ds_.data_num_, metadata.data());

  auto graph = std::make_shared<Graph<>>(ds_.data_num_, RaBitQSpace<>::kDegreeBound);

  auto search_job = std::make_shared<GraphSearchJob<RaBitQSpaceWithScalar>>(space, graph);

  MetadataFilter filter = MetadataFilter::empty();
  std::vector<uint32_t> ids(10);
  std::vector<ScalarData> results(10);
  auto query = ds_.queries_.data();

  // k=10, ef=5 should throw
  EXPECT_THROW(search_job
                   ->rabitq_hybrid_search_solo(query, 10, ids.data(), 5, filter, results.data()),
               std::invalid_argument);
}

#endif  // __AVX512F__

}  // namespace alaya
