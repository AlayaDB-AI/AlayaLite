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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "executor/jobs/graph_search_job.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "index/graph/nsg/nsg_builder.hpp"
#include "index/graph/qg/qg_builder.hpp"
#include "space/rabitq_space.hpp"
#include "utils/evaluate.hpp"
#include "utils/io_utils.hpp"
#include "utils/log.hpp"
#include "utils/rabitq_utils/search_utils/stopw.hpp"

// NOLINTBEGIN
int main(int argc, char **argv) {
  std::filesystem::path dir_name_ = std::filesystem::current_path() / "data" / "deep1M";
  std::filesystem::path data_file_ = dir_name_ / "deep1M_base.fvecs";
  std::filesystem::path query_file_ = dir_name_ / "deep1M_query.fvecs";
  std::filesystem::path gt_file_ = dir_name_ / "deep1M_groundtruth.ivecs";

  std::vector<float> data_;
  uint32_t points_num_;
  uint32_t dim_;

  std::vector<float> queries_;
  uint32_t query_num_;
  uint32_t query_dim_;

  std::vector<uint32_t> answers_;
  uint32_t ans_num_;
  uint32_t gt_col_;

  alaya::load_fvecs(data_file_, data_, points_num_, dim_);

  alaya::load_fvecs(query_file_, queries_, query_num_, query_dim_);
  assert(dim_ == query_dim_);

  alaya::load_ivecs(gt_file_, answers_, ans_num_, gt_col_);
  assert(ans_num_ == query_num_);

  // *************** TEST PERFORMANCE ********************
  using IDType = uint32_t;
  // ***************INDEX******************
  LOG_INFO("Building QG...");
  std::filesystem::path index_file = fmt::format("{}_rabitq.qg", dir_name_.string());
  std::string_view path = index_file.native();

  if (!std::filesystem::exists(index_file)) {
    std::shared_ptr<alaya::RaBitQSpace<>> space =
        std::make_shared<alaya::RaBitQSpace<>>(points_num_, dim_, alaya::MetricType::L2);
    space->fit(data_.data(), points_num_);
    LOG_INFO("Successfully fit data into space");

    auto qg = alaya::QGBuilder<alaya::RaBitQSpace<>>(space);
    qg.build_graph();

    space->save(path);
  }
  LOG_INFO("Successfully build qg!");
  // ***************QUERY*******************
  auto load_space = std::make_shared<alaya::RaBitQSpace<>>();
  load_space->load(path);
  auto search_job =
      std::make_unique<alaya::GraphSearchJob<alaya::RaBitQSpace<>>>(load_space, nullptr);

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
        search_job->rabitq_search_solo(queries_.data() + (n * query_dim_), topk, results.data(),
                                       ef);
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
// NOLINTEND
