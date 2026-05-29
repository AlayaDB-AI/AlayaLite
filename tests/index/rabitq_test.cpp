// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <fmt/core.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

#include "executor/jobs/graph_search_job.hpp"
#include "index/graph/qg/qg_builder.hpp"
#include "space/rabitq_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/log.hpp"

namespace alaya {
class RaBitQSiftSmallTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_ = sift_tiny(resolve_data_dir());
    ds_ = load_dataset(config_);
  }

  void TearDown() override {}

  DatasetConfig config_;
  Dataset ds_;
};

using IDType = uint32_t;
TEST_F(RaBitQSiftSmallTest, SiftSmallQGTest) {  // for code coverage
  LOG_INFO("Building QG...");
  std::filesystem::path index_file = config_.dir_ / fmt::format("{}_rabitq.qg", config_.name_);
  std::string_view path = index_file.native();

  if (!std::filesystem::exists(index_file)) {
    std::shared_ptr<alaya::RaBitQSpace<>> space =
        std::make_shared<alaya::RaBitQSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);
    space->fit(ds_.data_.data(), ds_.data_num_);
    LOG_INFO("Successfully fit data into space");

    auto qg = alaya::QGBuilder<RaBitQSpace<>>(space);
    qg.build_graph();

    space->save(path);
  }
  LOG_INFO("Successfully build qg!");
  auto load_space = std::make_shared<alaya::RaBitQSpace<>>();
  load_space->load(path);
  auto search_job = std::make_unique<alaya::GraphSearchJob<RaBitQSpace<>>>(load_space, nullptr);

  constexpr size_t topk = 10;
  constexpr size_t ef = 120;
  size_t total_correct = 0;
  std::vector<IDType> results(topk);

  LOG_INFO("Start querying...");
  for (uint32_t n = 0; n < ds_.query_num_; ++n) {
    search_job->rabitq_search_solo(ds_.queries_.data() + (n * ds_.dim_), topk, results.data(), ef);

    for (size_t k = 0; k < topk; ++k) {
      for (size_t j = 0; j < topk; ++j) {
        if (results[k] == ds_.ground_truth_[(n * ds_.gt_dim_) + j]) {
          total_correct++;
          break;
        }
      }
    }
  }

  auto recall = static_cast<float>(total_correct) / static_cast<float>(ds_.query_num_ * topk);
  EXPECT_GT(recall, 0.75F);
}

TEST_F(RaBitQSiftSmallTest, InvalidParameterTest) {
  std::filesystem::path index_file = config_.dir_ / fmt::format("{}_rabitq.qg", config_.name_);
  std::string_view path = index_file.native();

  if (!std::filesystem::exists(index_file)) {
    std::shared_ptr<alaya::RaBitQSpace<>> space =
        std::make_shared<alaya::RaBitQSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);
    space->fit(ds_.data_.data(), ds_.data_num_);

    auto qg = alaya::QGBuilder<RaBitQSpace<>>(space);
    qg.build_graph();
    space->save(path);
  }

  auto load_space = std::make_shared<alaya::RaBitQSpace<>>();
  load_space->load(path);
  auto search_job = std::make_unique<alaya::GraphSearchJob<RaBitQSpace<>>>(load_space, nullptr);

  size_t topk = 10;
  size_t ef = 5;  // ef < k, should throw exception
  std::vector<IDType> results(topk);
  auto query = ds_.queries_.data();

  EXPECT_THROW(search_job->rabitq_search_solo(query, topk, results.data(), ef),
               std::invalid_argument);
}
}  // namespace alaya
