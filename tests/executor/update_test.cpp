// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <unordered_set>
#include <vector>
#include "executor/jobs/graph_search_job.hpp"
#include "executor/jobs/graph_update_job.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "space/raw_space.hpp"
#include "space/sq8_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/log.hpp"
#include "utils/scalar_data.hpp"

namespace alaya {

class UpdateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path data_dir = std::filesystem::current_path().parent_path() / "data";
    ds_ = load_dataset(sift_small(data_dir));
  }

  void TearDown() override {}

  Dataset ds_;
  std::unordered_set<uint32_t> point_set_;  ///< The set of points that has been inserted.
};

TEST_F(UpdateTest, HalfInsertTest) {
  uint32_t topk = 10;
  uint32_t half_size = ds_.data_.size() / ds_.dim_ / 2;

  LOG_DEBUG("the data size is {}", ds_.data_.size());
  auto space = std::make_shared<alaya::RawSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);

  // Use the first half of the data to build the graph.
  space->fit(ds_.data_.data(), half_size);

  auto build_start = std::chrono::steady_clock::now();

  alaya::HNSWBuilder<alaya::RawSpace<>> hnsw = alaya::HNSWBuilder<alaya::RawSpace<>>(space);
  std::shared_ptr<alaya::Graph<>> hnsw_graph = hnsw.build_graph();

  auto build_end = std::chrono::steady_clock::now();
  auto build_time = static_cast<std::chrono::duration<double>>(build_end - build_start).count();
  LOG_INFO("The time of building hnsw is {}s.", build_time);

  std::vector<float> half_data(half_size * ds_.dim_);
  half_data.insert(half_data.begin(), ds_.data_.begin(), ds_.data_.begin() + half_size * ds_.dim_);

  auto half_gt = find_exact_gt<>(ds_.queries_, half_data, ds_.dim_, topk);

  auto search_job = std::make_shared<alaya::GraphSearchJob<alaya::RawSpace<>>>(space, hnsw_graph);
  std::vector<uint32_t> ids(ds_.query_num_ * topk);
  size_t ef_1 = 30;
  for (uint32_t i = 0; i < ds_.query_num_; i++) {
    auto cur_query = ds_.queries_.data() + i * ds_.dim_;
    // New interface: search_solo(query, ids, topk, ef) returns topk results directly
    search_job->search_solo(cur_query, ids.data() + i * topk, topk, ef_1);
  }

  auto recall = calc_recall(ids.data(), half_gt.data(), ds_.query_num_, topk, topk);
  ASSERT_GT(recall, 0.9);

  auto update_job = std::make_shared<alaya::GraphUpdateJob<RawSpace<>>>(search_job);

  for (uint32_t i = half_size; i < ds_.data_num_; i++) {
    auto cur_data = ds_.data_.data() + i * ds_.dim_;
    update_job->insert_and_update(cur_data, 50);
  }

  size_t ef_2 = 50;
  for (uint32_t i = 0; i < ds_.query_num_; i++) {
    auto cur_query = ds_.queries_.data() + i * ds_.dim_;
    // New interface: search_solo(query, ids, topk, ef) returns topk results directly
    search_job->search_solo(cur_query, ids.data() + i * topk, topk, ef_2);
  }

  auto full_gt = find_exact_gt(ds_.queries_, ds_.data_, ds_.dim_, topk);
  auto full_recall = calc_recall(ids.data(), full_gt.data(), ds_.query_num_, topk, topk);
  ASSERT_GT(full_recall, 0.9);

  for (uint32_t i = half_size; i < ds_.data_num_; i++) {
    update_job->remove(i);
  }
  size_t ef_3 = 50;
  std::vector<uint32_t> ef_results_3(ef_3);
  for (uint32_t i = 0; i < ds_.query_num_; i++) {
    auto cur_query = ds_.queries_.data() + i * ds_.dim_;
    search_job->search_solo_updated(cur_query, ef_results_3.data(), ef_3, topk);
    std::copy(ef_results_3.begin(), ef_results_3.begin() + topk, ids.data() + i * topk);
  }
  auto recall_after_delete = calc_recall(ids.data(), full_gt.data(), ds_.query_num_, topk, topk);
  LOG_INFO("The recall after delete is {}", recall_after_delete);

  auto gt_after_delete =
      find_exact_gt<>(ds_.queries_, ds_.data_, ds_.dim_, topk,
                      &update_job->job_context_->removed_vertices_);

  auto recall_after_delete_gt = calc_recall(ids.data(), gt_after_delete.data(), ds_.query_num_, topk, topk);
  LOG_INFO("The recall after delete gt is {}", recall_after_delete_gt);
}

TEST(GraphUpdateJobSplitSpaceTest, IncrementalInsertKeepsBuildAndSearchSpacesConsistent) {
  namespace fs = std::filesystem;

  using IDType = uint32_t;
  using BuildSpaceType = RawSpace<float, float, IDType, SequentialStorage<float, IDType>, EmptyScalarData>;
  using SearchSpaceType = SQ8Space<float,
                                   float,
                                   IDType,
                                   SequentialStorage<uint8_t, IDType>,
                                   ScalarData>;

  auto temp_dir = fs::temp_directory_path() / "graph_update_job_split_space_test";
  fs::remove_all(temp_dir);
  fs::create_directories(temp_dir);

  RocksDBConfig config;
  config.db_path_ = (temp_dir / "rocksdb").string();

  auto build_space = std::make_shared<BuildSpaceType>(8, 2, MetricType::L2);
  auto search_space = std::make_shared<SearchSpaceType>(8, 2, MetricType::L2, config);

  float base_vectors[] = {
      0.0F, 0.0F,
      1.0F, 1.0F,
  };
  ScalarData base_scalar[] = {
      {"seed_0", "doc0", {}},
      {"seed_1", "doc1", {}},
  };

  build_space->fit(base_vectors, 2);
  search_space->fit(base_vectors, 2, base_scalar);

  auto graph_builder = std::make_shared<HNSWBuilder<BuildSpaceType>>(build_space);
  auto graph = std::shared_ptr<Graph<>>(graph_builder->build_graph(1).release());
  auto job_context = std::make_shared<JobContext<IDType>>();
  auto search_job = std::make_shared<GraphSearchJob<SearchSpaceType, BuildSpaceType>>(search_space,
                                                                                       graph,
                                                                                       job_context,
                                                                                       build_space);
  auto update_job = std::make_shared<GraphUpdateJob<SearchSpaceType, BuildSpaceType>>(search_job);

  float new_vector[] = {0.05F, 0.05F};
  ScalarData new_scalar{"seed_2", "doc2", {{"group", std::string("new")}}};

  auto inserted_id = update_job->insert_and_update(new_vector, 32, &new_scalar);
  EXPECT_EQ(inserted_id, 2U);
  EXPECT_EQ(build_space->get_data_num(), 3U);
  EXPECT_EQ(search_space->get_data_num(), 3U);
  EXPECT_EQ(search_space->get_scalar_data(inserted_id).item_id, "seed_2");

  auto *stored_vector = build_space->get_data_by_id(inserted_id);
  EXPECT_FLOAT_EQ(stored_vector[0], new_vector[0]);
  EXPECT_FLOAT_EQ(stored_vector[1], new_vector[1]);

  std::vector<IDType> ids(1, std::numeric_limits<IDType>::max());
  search_job->search_solo(new_vector, ids.data(), 1, 16);
  EXPECT_EQ(ids[0], inserted_id);

  fs::remove_all(temp_dir);
}

}  // namespace alaya
