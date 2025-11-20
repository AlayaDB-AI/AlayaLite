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
#include <vector>

#include "executor/jobs/graph_search_job.hpp"
#include "gtest/gtest.h"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "index/graph/nsg/nsg_builder.hpp"
#include "index/graph/qg/qg_builder.hpp"
#include "space/rabitq_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/io_utils.hpp"
#include "utils/log.hpp"
#include "utils/rabitq_utils/search_utils/stopw.hpp"

namespace alaya {
class Deep1MTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::filesystem::path data_dir = std::filesystem::current_path().parent_path() / "data";
    dataset_ = std::make_unique<DEEP1MTestData>(data_dir.string());
    dataset_->ensure_dataset();

    data_ = dataset_->get_data();
    queries_ = dataset_->get_queries();
    answers_ = dataset_->get_answers();

    points_num_ = dataset_->get_data_num();
    query_num_ = dataset_->get_query_num();
    ans_num_ = dataset_->get_ans_num();

    dim_ = dataset_->get_dim();
    query_dim_ = dim_;
    gt_col_ = dataset_->get_ans_dim();

    index_save_path_ = dataset_->get_dataset_dir();
  }

  void TearDown() override {}

  std::unique_ptr<TestDatasetBase> dataset_;

  std::vector<float> data_;
  uint32_t points_num_;
  uint32_t dim_;

  std::vector<float> queries_;
  uint32_t query_num_;
  uint32_t query_dim_;

  std::vector<uint32_t> answers_;
  uint32_t ans_num_;
  uint32_t gt_col_;

  std::filesystem::path index_save_path_;
};

using IDType = uint32_t;

TEST_F(Deep1MTest, DISABLED_Deep1mNSGTest) {
  // ***************INDEX******************
  LOG_INFO("Building nsg...");
  // std::filesystem::path index_file = fmt::format("{}_rabitq.nsg", dir_name_.string());
  // std::string_view path = index_file.native();

  /// todo: save and load
  std::shared_ptr<alaya::RaBitQSpace<>> space =
      std::make_shared<alaya::RaBitQSpace<>>(points_num_, dim_, MetricType::L2);
  space->fit(data_.data(), points_num_);
  LOG_INFO("Successfully fit data into space");

  auto nsg = alaya::NSGBuilder<alaya::RaBitQSpace<>>(space);
  auto graph = nsg.build_graph(96);

  // ***************QUERY******************
  auto search_job = std::make_unique<alaya::GraphSearchJob<RaBitQSpace<>>>(space, nullptr);
  std::vector<size_t> efs = {10,  20,  40,  50,  60,  80,  100, 150, 170, 190,
                             200, 250, 300, 400, 500, 600, 700, 800, 1500};
  size_t test_round = 3;
  size_t topk = 10;
  alaya::StopW timer;
  std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(efs.size()));
  std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(efs.size()));

  LOG_INFO("Start querying...");
  LOG_INFO("search entry point: {}", space->get_ep());
  for (size_t r = 0; r < test_round; ++r) {
    for (size_t i = 0; i < efs.size(); ++i) {  // NOLINT
      size_t ef = efs[i];
      size_t total_correct = 0;
      float total_time = 0;
      std::vector<IDType> results(topk);
      LOG_INFO("current ef in this round:{}", ef);
      for (uint32_t n = 0; n < query_num_; ++n) {
        timer.reset();
#if defined(__AVX512F__)
        search_job->rabitq_search_solo(queries_.data() + (n * query_dim_), topk, results.data(),
                                       ef);
#else
        EXPECT_EXIT(search_job->rabitq_search_solo(queries_.data() + (n * query_dim_), topk,
                                                   results.data(), ef),
                    ::testing::ExitedWithCode(1), "no avx simd supported!");
        return;
#endif
        total_time += timer.get_elapsed_micro();
        // recall
        for (size_t k = 0; k < topk; ++k) {
          for (size_t j = 0; j < topk; ++j) {
            if (results[k] == answers_[(n * gt_col_) + j]) {
              total_correct++;
              break;
            }
          }
        }
      }
      float qps = static_cast<float>(query_num_) / (total_time / 1e6F);
      float recall = static_cast<float>(total_correct) / static_cast<float>(query_num_ * topk);

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

TEST_F(Deep1MTest, DISABLED_Deep1mHNSWTest) {
  // ***************INDEX******************
  LOG_INFO("Building hnsw...");
  // std::filesystem::path index_file = fmt::format("{}_rabitq.hnsw", dir_name_.string());
  // std::string_view path = index_file.native();

  /// todo: save space
  std::shared_ptr<alaya::RaBitQSpace<>> space =
      std::make_shared<alaya::RaBitQSpace<>>(points_num_, dim_, MetricType::L2);
  space->fit(data_.data(), points_num_);
  LOG_INFO("Successfully fit data into space");
  auto hnsw = alaya::HNSWBuilder<alaya::RaBitQSpace<>>(space);
  auto graph = hnsw.build_graph(96);

  // ***************QUERY******************
  /// todo: load space
  auto search_job = std::make_unique<alaya::GraphSearchJob<RaBitQSpace<>>>(space, nullptr);
  std::vector<size_t> efs = {10,  20,  40,  50,  60,  80,  100, 150, 170, 190,
                             200, 250, 300, 400, 500, 600, 700, 800, 1500};
  size_t test_round = 3;
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
      for (uint32_t n = 0; n < query_num_; ++n) {
        timer.reset();
#if defined(__AVX512F__)
        search_job->rabitq_search_solo(queries_.data() + (n * query_dim_), topk, results.data(),
                                       ef);
#else
        EXPECT_EXIT(search_job->rabitq_search_solo(queries_.data() + (n * query_dim_), topk,
                                                   results.data(), ef),
                    ::testing::ExitedWithCode(1), "no avx simd supported!");
        return;
#endif
        total_time += timer.get_elapsed_micro();
        // recall
        for (size_t k = 0; k < topk; ++k) {
          for (size_t j = 0; j < topk; ++j) {
            if (results[k] == answers_[(n * gt_col_) + j]) {
              total_correct++;
              break;
            }
          }
        }
      }
      float qps = static_cast<float>(query_num_) / (total_time / 1e6F);
      float recall = static_cast<float>(total_correct) / static_cast<float>(query_num_ * topk);

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

TEST_F(Deep1MTest, DISABLED_DataSetCorrectnessTest) {
  size_t len = 10;
  // data
  std::cerr << "data:\n";
  for (size_t i = 0; i < len; ++i) {
    std::cerr << data_[i] << " ";
  }
  std::cerr << "\n";
  // query
  std::cerr << "query:\n";
  for (size_t i = 0; i < len; ++i) {
    std::cerr << queries_[i] << " ";
  }
  std::cerr << "\n";
  // answers
  std::cerr << "answers:\n";
  for (size_t i = 0; i < len; ++i) {
    std::cerr << answers_[i] << " ";
  }
  std::cerr << "\n";
}

TEST_F(Deep1MTest, Deep1mQGTest) {
  // ***************INDEX******************
  LOG_INFO("Building QG...");
  std::filesystem::path index_file =
      fmt::format("{}_rabitq.qg", index_save_path_.string() + "/deep1M");
  std::string_view path = index_file.native();

  if (!std::filesystem::exists(index_file)) {
    std::shared_ptr<alaya::RaBitQSpace<>> space =
        std::make_shared<alaya::RaBitQSpace<>>(points_num_, dim_, MetricType::L2);
    space->fit(data_.data(), points_num_);
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
  //     std::make_shared<alaya::RaBitQSpace<>>(points_num_, dim_, MetricType::L2);
  // space->fit(data_.data(), points_num_);
  // LOG_INFO("Successfully fit data into space");

  // auto qg = alaya::QGBuilder<RaBitQSpace<>>(space);
  // qg.build_graph();
  // auto search_job = std::make_unique<alaya::GraphSearchJob<RaBitQSpace<>>>(space, nullptr);

  std::vector<size_t> efs = {10,  20,  40,  50,  55,  60,  80,  100, 150, 170,
                             190, 200, 250, 300, 400, 500, 600, 800, 1500};
  size_t test_round = 3;
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
      for (uint32_t n = 0; n < query_num_; ++n) {
        timer.reset();
#if defined(__AVX512F__)
        search_job->rabitq_search_solo(queries_.data() + (n * query_dim_), topk, results.data(),
                                       ef);
#else
        EXPECT_EXIT(search_job->rabitq_search_solo(queries_.data() + (n * query_dim_), topk,
                                                   results.data(), ef),
                    ::testing::ExitedWithCode(1), "no avx simd supported!");
        return;
#endif
        total_time += timer.get_elapsed_micro();
        // recall
        for (size_t k = 0; k < topk; ++k) {
          for (size_t j = 0; j < topk; ++j) {
            if (results[k] == answers_[(n * gt_col_) + j]) {
              total_correct++;
              break;
            }
          }
        }
      }
      float qps = static_cast<float>(query_num_) / (total_time / 1e6F);
      float recall = static_cast<float>(total_correct) / static_cast<float>(query_num_ * topk);

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
