// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <memory>
#include <string_view>
#include "index/graph/nsg/nsg_builder.hpp"

#include "executor/jobs/graph_search_job.hpp"
#include "index/graph/graph.hpp"
#include "space/raw_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/thread_config.hpp"
#include "utils/timer.hpp"

namespace alaya {
namespace {

auto make_one_dim_space(const std::vector<float> &values) -> std::shared_ptr<RawSpace<>> {
  auto space =
      std::make_shared<RawSpace<>>(static_cast<uint32_t>(values.size()), 1, MetricType::L2);
  space->fit(values.data(), static_cast<uint32_t>(values.size()));
  return space;
}

}  // namespace

class NSGTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path data_dir = std::filesystem::current_path().parent_path() / "data";
    ds_ = load_dataset(sift_micro(data_dir));

    space_ = std::make_shared<RawSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);
    space_->fit(ds_.data_.data(), ds_.data_num_);
    nsg_ = std::make_unique<NSGBuilder<RawSpace<>>>(space_);
  }

  void TearDown() override {
    if (std::filesystem::exists(filename_)) {
      remove(filename_.data());
    }
  }

  Dataset ds_;
  std::unique_ptr<NSGBuilder<RawSpace<>>> nsg_ = nullptr;
  std::shared_ptr<RawSpace<>> space_ = nullptr;
  std::string_view filename_ = "nsg_test.graph";
};

TEST_F(NSGTest, BuildGraphTest) {
  auto graph = nsg_->build_graph();
  // graph->print_graph();
}

TEST(NSGInternalTest, SyncPruneOccludesLongerNeighbor) {
  auto space = make_one_dim_space({0.0F, 1.0F, 2.0F});
  NSGBuilder<RawSpace<>> builder(space, 2, 4);

  auto knng = std::make_unique<Graph<>>(3, 1);
  Graph<> graph(3, 2);
  std::vector<Node<uint32_t>> pool = {{1U, 1.0F}, {2U, 4.0F}};
  std::vector<bool> vis(3, false);

  builder.sync_prune(0, pool, vis, knng, graph);

  EXPECT_EQ(graph.at(0, 0), 1U);
  EXPECT_EQ(graph.at(0, 1), Graph<>::kEmptyId);
}

TEST(NSGInternalTest, AddReverseLinksPrunesWhenDestinationIsFull) {
  auto space = make_one_dim_space({0.0F, 1.0F, 2.0F, 10.0F});
  NSGBuilder<RawSpace<>> builder(space, 2, 4);

  Graph<> graph(4, 2);
  graph.at(0, 0) = 1;
  graph.at(1, 0) = 2;
  graph.at(1, 1) = 3;

  std::vector<std::mutex> locks(4);
  builder.add_reverse_links(0, locks, graph);

  EXPECT_EQ(graph.at(1, 0), 2U);
  EXPECT_EQ(graph.at(1, 1), 3U);
}

TEST(NSGInternalTest, AttachUnlinkedFallsBackToRandomAvailableNode) {
  auto space = make_one_dim_space({0.0F, 1.0F, 2.0F, 3.0F});
  NSGBuilder<RawSpace<>> builder(space, 1, 1);

  builder.final_graph_ = std::make_unique<Graph<>>(4, 1);
  builder.final_graph_->eps_.push_back(0);
  builder.final_graph_->at(0, 0) = 1;
  builder.final_graph_->at(1, 0) = 0;
  builder.ep_ = 0;

  std::vector<uint32_t> degrees = {1, 1, 0, 0};
  std::vector<bool> vis = {true, true, false, false};
  std::vector<bool> vis2(4, false);

  auto node = builder.attach_unlinked(vis, vis2, degrees);

  EXPECT_TRUE(node == 2U || node == 3U);
  EXPECT_EQ(builder.final_graph_->at(node, 0), 0U);
  EXPECT_EQ(degrees[node], 1U);
}

TEST(NSGInternalTest, InsertIntoPoolRejectsDuplicateAndFarNeighbor) {
  auto space = make_one_dim_space({0.0F, 1.0F});
  NSGBuilder<RawSpace<>> builder(space, 2, 4);

  Neighbor<uint32_t> pool[4] = {
      Neighbor<uint32_t>(0, 1.0F),
      Neighbor<uint32_t>(1, 2.0F),
      Neighbor<uint32_t>(2, 3.0F),
      Neighbor<uint32_t>(3, 4.0F),
  };

  EXPECT_EQ(builder.insert_into_pool(pool, 3, Neighbor<uint32_t>(1, 1.5F)), 3);
  EXPECT_EQ(builder.insert_into_pool(pool, 3, Neighbor<uint32_t>(4, 4.0F)), 3);

  auto pos = builder.insert_into_pool(pool, 3, Neighbor<uint32_t>(5, 1.5F));
  EXPECT_EQ(pos, 1);
  EXPECT_EQ(pool[1].id_, 5U);
}

class NSGSearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path data_dir = std::filesystem::current_path().parent_path() / "data";
    ds_ = load_dataset(sift_micro(data_dir));
  }

  void TearDown() override {}

  Dataset ds_;
};

TEST_F(NSGSearchTest, SimpleSearchTest) {
  const size_t kM = 32;
  size_t topk = 10;
  size_t ef = 10;
  std::string index_type = "NSG";

  std::filesystem::path index_file = fmt::format("{}_M{}.{}", ds_.name_, kM, index_type);
  std::shared_ptr<alaya::RawSpace<>> space =
      std::make_shared<alaya::RawSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);
  space->fit(ds_.data_.data(), ds_.data_num_);

  auto load_graph = std::make_shared<alaya::Graph<>>(ds_.data_num_, kM);

  if (!std::filesystem::exists(index_file)) {
    auto t1 = Timer();

    alaya::NSGBuilder<alaya::RawSpace<>> nsg = alaya::NSGBuilder<alaya::RawSpace<>>(space, kM);
    auto graph = nsg.build_graph();

    LOG_INFO("The time of building hnsw is {}s.", t1.elapsed() / 1000000.0);

    std::string_view path = index_file.native();
    graph->save(path);
  }
  std::string_view path = index_file.native();
  load_graph->load(path);

  // load_graph->print_graph();

  auto task_generator =
      std::make_unique<alaya::GraphSearchJob<alaya::RawSpace<>>>(space, load_graph);

  LOG_INFO("creator task generator success");

  using IDType = uint32_t;

  Timer timer{};
  std::vector<std::vector<IDType>> res_pool(ds_.query_num_, std::vector<IDType>(topk));
  const size_t kSearchThreadNum =
      std::min<size_t>(cap_thread_count(16), static_cast<size_t>(ds_.query_num_));
  std::vector<std::thread> tasks(kSearchThreadNum);

  auto search_knn = [&](uint32_t i) {
    for (; i < ds_.query_num_; i += kSearchThreadNum) {
      std::vector<uint32_t> ids(topk);  // Now returns topk directly
      auto cur_query = ds_.queries_.data() + i * ds_.dim_;
      // New interface: search_solo(query, ids, topk, ef) returns topk results
      task_generator->search_solo(cur_query, ids.data(), topk, ef);

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

  // Computing recall
  auto recall = calc_recall(res_pool, ds_.ground_truth_.data(), ds_.query_num_, ds_.gt_dim_, topk);
  LOG_INFO("recall is {}.", recall);
  EXPECT_GE(recall, 0.5);
}

}  // namespace alaya
