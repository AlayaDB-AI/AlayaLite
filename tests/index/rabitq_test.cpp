/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <fmt/core.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "executor/jobs/graph_search_job.hpp"
#include "index/graph/qg/qg_builder.hpp"
#include "space/rabitq_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/log.hpp"
#include "utils/rabitq_utils/search_utils/stopw.hpp"

namespace alaya {
class RaBitQSiftSmallTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path data_dir = std::filesystem::current_path().parent_path() / "data";
    config_ = sift_small(data_dir);
    ds_ = load_dataset(config_);
  }

  void TearDown() override {}

  DatasetConfig config_;
  Dataset ds_;
};

using IDType = uint32_t;
TEST_F(RaBitQSiftSmallTest, SiftSmallQGTest) {  // for code coverage
  // ***************INDEX******************
  LOG_INFO("Building QG...");
  std::filesystem::path index_file =
      fmt::format("{}_rabitq.qg", config_.dir_.string() + "/siftsmall");
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
  // ***************QUERY*******************
  auto load_space = std::make_shared<alaya::RaBitQSpace<>>();
  load_space->load(path);
  auto search_job = std::make_unique<alaya::GraphSearchJob<RaBitQSpace<>>>(load_space, nullptr);

  // std::shared_ptr<alaya::RaBitQSpace<>> space =
  //     std::make_shared<alaya::RaBitQSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);
  // space->fit(ds_.data_.data(), ds_.data_num_);
  // LOG_INFO("Successfully fit data into space");

  // auto qg = alaya::QGBuilder<RaBitQSpace<>>(space);
  // qg.build_graph();
  // auto search_job = std::make_unique<alaya::GraphSearchJob<RaBitQSpace<>>>(space, nullptr);

  std::vector<size_t> efs = {10,  20,  40,  50,  55,  60,  80,  100, 150, 170,
                             190, 200, 250, 300, 400, 500, 600, 800, 1500};
  size_t test_round = 1;
  size_t topk = 10;
  alaya::StopW timer;
  std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(efs.size()));
  std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(efs.size()));

  LOG_INFO("Start querying...");
  for (size_t r = 0; r < test_round; ++r) {
    for (size_t i = 0; i < efs.size(); ++i) {  // NOLINT
      size_t ef = efs[i];
      size_t total_correct = 0;
      float total_time = 0;
      std::vector<IDType> results(topk);
      LOG_INFO("current ef in this round:{}", ef);
      for (uint32_t n = 0; n < ds_.query_num_; ++n) {
        timer.reset();
#if defined(__AVX512F__)
        search_job->rabitq_search_solo(ds_.queries_.data() + (n * ds_.dim_), topk, results.data(),
                                       ef);
#else
        EXPECT_THROW(search_job->rabitq_search_solo(ds_.queries_.data() + (n * ds_.dim_), topk,
                                                    results.data(), ef),
                     std::runtime_error);
        return;
#endif

        total_time += timer.get_elapsed_micro();
        // recall
        for (size_t k = 0; k < topk; ++k) {
          for (size_t j = 0; j < topk; ++j) {
            if (results[k] == ds_.ground_truth_[(n * ds_.gt_dim_) + j]) {
              total_correct++;
              break;
            }
          }
        }
      }
      float qps = static_cast<float>(ds_.query_num_) / (total_time / 1e6F);
      float recall = static_cast<float>(total_correct) / static_cast<float>(ds_.query_num_ * topk);

      all_qps[r][i] = qps;
      all_recall[r][i] = recall;
    }
  }

  auto avg_qps = alaya::horizontal_avg(all_qps);
  auto avg_recall = alaya::horizontal_avg(all_recall);

  std::cout << "ef\tQPS\tRecall\n";
  for (size_t i = 0; i < avg_qps.size(); ++i) {
    std::cout << efs[i] << '\t' << avg_qps[i] << '\t' << avg_recall[i] << '\n';
  }
}
}  // namespace alaya
