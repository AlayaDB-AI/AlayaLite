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
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "index/graph/nsg/nsg_builder.hpp"
#include "index/graph/qg/qg_builder.hpp"
#include "space/rabitq_space.hpp"
#include "utils/io_utils.hpp"
#include "utils/log.hpp"
#include "utils/rbq_utils/search_utils/stopw.hpp"

namespace alaya {
class Deep1MTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!std::filesystem::exists(dir_name_)) {
      // mkdir data
      std::filesystem::create_directories(dir_name_.parent_path());
      int ret = std::system(
          "wget -P ./data http://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/deep1M.tar.gz");
      if (ret != 0) {
        throw std::runtime_error("Download deep1M.tar.gz failed");
      }
      ret = std::system("tar -zxvf ./data/deep1M.tar.gz -C ./data");
      if (ret != 0) {
        throw std::runtime_error("Unzip deep1M.tar.gz failed");
      }
    }

    alaya::load_fvecs(data_file_, data_, points_num_, dim_);

    alaya::load_fvecs(query_file_, queries_, query_num_, query_dim_);
    assert(dim_ == query_dim_);

    alaya::load_ivecs(gt_file_, answers_, ans_num_, gt_col_);
    assert(ans_num_ == query_num_);
  }

  void TearDown() override {}
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
};

using IDType = uint32_t;

TEST_F(Deep1MTest, DISABLED_QuantizationTest) {
  fmt::println("data_num: {}, dim: {}", points_num_, dim_);

  size_t degree_bound = fastscan::kBatchSize;

  // ----------------simple connection----------------
  std::unique_ptr<alaya::RBQSpace<>> space =
      std::make_unique<alaya::RBQSpace<>>(points_num_, dim_, MetricType::L2);
  space->fit(data_.data(), points_num_);

  std::mt19937 gen(654321);
  std::uniform_int_distribution<int> dis(0, static_cast<int>(points_num_ - 1));
  auto centroid = static_cast<IDType>(dis(gen));
  fmt::println("centroid: {}", centroid);

  std::cerr << "centroid's first few dimensions(from fvecs): ";
  auto c_fptr = data_.data() + (centroid * dim_);
  for (int i = 0; i < 10; ++i) {
    std::cerr << *(c_fptr + i) << " ";
  }
  std::cerr << "\n";

  std::cerr << "centroid's first few dimensions(from space): ";
  auto c_ptr = space->get_data_by_id(centroid);
  for (int i = 0; i < 10; ++i) {
    std::cerr << *(c_ptr + i) << " ";
  }
  std::cerr << "\n";

  std::vector<IDType> neighbors(degree_bound);
  std::cerr << "Its' neighbors:\n";
  for (size_t i = 0; i < degree_bound; ++i) {
    neighbors[i] = static_cast<IDType>(dis(gen));
    std::cerr << neighbors[i] << " ";
  }
  std::cerr << "\n";
  // ----------------quantization----------------
  int print_dim_num = 6;
  size_t padded_dim = space->get_padded_dim();

  std::vector<float> rotated_c(padded_dim);
  space->rotate_vec(space->get_data_by_id(centroid), rotated_c.data());
  std::cerr << "First few dimension of rotated_centroid: \n";
  fmt::print("{} : ", centroid);
  for (int i = 0; i < print_dim_num; ++i) {
    std::cerr << rotated_c[i] << " ";
  }
  std::cerr << "\n";

  std::cerr << "Rotated neighbors' first few dimensions: \n";

  std::vector<float> rotated_nei(degree_bound * padded_dim);
  for (size_t i = 0; i < degree_bound; ++i) {
    auto nei_id = neighbors[i];
    space->rotate_vec(space->get_data_by_id(nei_id), &rotated_nei[i * padded_dim]);
    std::cerr << nei_id << ": ";
    for (int j = 0; j < print_dim_num; ++j) {
      std::cerr << rotated_nei[(i * padded_dim) + j] << " ";
    }
    std::cerr << "\n";
  }

  space->update_batch_data(centroid, neighbors.data());

  // byte size
  size_t bin_code_size = padded_dim * degree_bound / 8;
  int check_qc_byte_cnt = 4;
  auto qc_ptr = space->get_nei_qc_ptr(centroid);

  std::cerr << "qc example(first " << check_qc_byte_cnt << " bytes): \n";
  for (size_t i = 0; i < check_qc_byte_cnt; ++i) {
    auto ele = *(qc_ptr + i);
    for (int j = 7; j >= 0; --j) {
      std::cerr << ((ele >> j) & 1) << " ";
    }
  }
  std::cerr << "\n";

  std::cerr << "qc example(last " << check_qc_byte_cnt << " bytes): \n";
  for (size_t i = bin_code_size - check_qc_byte_cnt; i < bin_code_size; ++i) {
    auto ele = *(qc_ptr + i);
    for (int j = 7; j >= 0; --j) {
      std::cerr << ((ele >> j) & 1) << " ";
    }
  }
  std::cerr << "\n";

  int check_precomputed_data_cnt = 8;
  std::cerr << "precomputed f_add data(last " << check_precomputed_data_cnt << " elements): \n";
  auto f_add_ptr = space->get_f_add_ptr(centroid);
  for (size_t i = degree_bound - check_precomputed_data_cnt; i < degree_bound; ++i) {
    std::cerr << *(f_add_ptr + i) << " ";
  }
  std::cerr << "\n";

  std::cerr << "precomputed f_rescale data(last " << check_precomputed_data_cnt << " elements):\n ";
  auto f_rescale_ptr = space->get_f_rescale_ptr(centroid);
  for (size_t i = degree_bound - check_precomputed_data_cnt; i < degree_bound; ++i) {
    std::cerr << *(f_rescale_ptr + i) << " ";
  }
  std::cerr << "\n";

  // ----------------est_dist calculation----------------
  fmt::println("query num: {}", query_num_);
  auto query_id = 888;
  auto query_ptr = queries_.data() + (query_id * query_dim_);
  std::cerr << "first few dimensions of query_" << query_id << " :";
  for (int i = 0; i < 8; ++i) {
    std::cerr << query_ptr[i] << " ";
  }
  std::cerr << "\n";

  auto q_computer = space->get_query_computer(query_ptr);
  q_computer.load_centroid(centroid, neighbors.data());
  std::cerr << "Print est dist for every neighbor:\n";
  for (int k = 0; k < degree_bound; ++k) {
    auto nei_id = neighbors[k];
    fmt::println("{} 's est_dist: {}", nei_id, q_computer(k));
  }
}

// TEST_F(Deep1MTest, DISABLED_SingleQueryTest) {
//   // ***************INDEX******************
//   LOG_INFO("Building QG...");
//   std::filesystem::path index_file = fmt::format("{}_single_query.qg", dir_name_.string());
//   std::string_view path = index_file.native();

//   /// todo: save and load
//   std::shared_ptr<alaya::RBQSpace<>> space =
//       std::make_shared<alaya::RBQSpace<>>(points_num_, dim_, MetricType::L2);
//   space->fit(data_.data(), points_num_);
//   LOG_INFO("Successfully fit data into space");
//   if (!std::filesystem::exists(index_file)) {
//     std::cerr << "Didn't find index file, start to build from scratch\n";
//     auto qg = alaya::QGBuilder<RBQSpace<>>(space);
//     std::cerr << "Done initialize qg.\n";
//     auto graph = qg.build_graph();
//     std::cerr << "Done building graph.\n";
//     graph->save(path);
//   }
//   // ***************QUERY******************
//   /// todo: 此处space没被refine(即space也要save and load)
//   auto load_graph = std::make_shared<alaya::Graph<>>();
//   load_graph->load(path);

//   // ---------- check quantization and neighbors ----------
//   int node_id = 888;
//   auto degree_bound = RBQSpace<>::kDegreeBound;

//   // print edges
//   auto edges = load_graph->edges(node_id);
//   fmt::print("node_{} 's edges: \n", node_id);
//   for (int i = 0; i < load_graph->max_nbrs_; ++i) {
//     std::cout << *(edges + i) << " ";
//   }
//   std::cout << "\n";

//   // print quantization code
//   int num_byte = 4;  // only first 4 bytes;
//   auto qc_ptr = space->get_nei_qc_ptr(node_id);
//   fmt::print("print first {} bytes of quantization code: \n", num_byte);
//   for (int i = 0; i < num_byte; ++i) {
//     uint8_t qc = *(qc_ptr + i);
//     for (int j = 7; j >= 0; --j) {
//       std::cout << ((qc >> j) & 1) << " ";
//     }
//   }
//   std::cout << "\n";

//   // print f_add and f_scale
//   auto f_add_ptr = space->get_f_add_ptr(node_id);
//   fmt::print("f_add of node_{} 's neighbors:\n", node_id);
//   for (int i = 0; i < degree_bound; ++i) {
//     std::cout << *(f_add_ptr + i) << " ";
//   }
//   std::cout << "\n";

//   auto f_scale_ptr = space->get_f_rescale_ptr(node_id);
//   fmt::print("f_scale of node_{} 's neighbors:\n", node_id);
//   for (int i = 0; i < degree_bound; ++i) {
//     std::cout << *(f_scale_ptr + i) << " ";
//   }
//   std::cout << "\n";
//   // ------------------------------------------------------
//   int query_index = 666;
//   size_t ef = 100;
//   int topk = 10;
//   auto search_job = std::make_unique<alaya::GraphSearchJob<RBQSpace<>>>(space, load_graph);
//   std::vector<IDType> results(topk);
//   search_job->rabitq_search_solo(queries_.data() + (query_index * query_dim_), topk, results.data(),
//                                  ef);
//   fmt::print("query_{}'s search topk:\n", query_index);
//   for (int i = 0; i < topk; ++i) {
//     std::cout << results[i] << " ";
//   }
//   std::cout << "\n";

//   fmt::print("query_{}'s topk gt:\n", query_index);
//   for (int i = 0; i < topk; ++i) {
//     std::cout << answers_[(query_index * gt_col_) + i] << " ";
//   }
//   std::cout << "\n";
// }

TEST_F(Deep1MTest, DISABLED_Deep1mNSGTest) {
  // ***************INDEX******************
  LOG_INFO("Building nsg...");
  std::filesystem::path index_file = fmt::format("{}_rabitq.nsg", dir_name_.string());
  std::string_view path = index_file.native();

  /// todo: save and load
  std::shared_ptr<alaya::RBQSpace<>> space =
      std::make_shared<alaya::RBQSpace<>>(points_num_, dim_, MetricType::L2);
  space->fit(data_.data(), points_num_);
  LOG_INFO("Successfully fit data into space");
  if (!std::filesystem::exists(index_file)) {
    auto nsg = alaya::NSGBuilder<alaya::RBQSpace<>>(space);
    auto graph = nsg.build_graph(96);
    graph->save(path);
  }
  // ***************QUERY******************
  auto load_graph = std::make_shared<alaya::Graph<>>();
  load_graph->load(path);

  auto search_job = std::make_unique<alaya::GraphSearchJob<RBQSpace<>>>(space, load_graph);
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
        // results is overwritten
        // search_job->rabitq_search_solo(queries_.data() + (n * query_dim_), topk, results.data(),
        //                                ef);
        search_job->rabitq_search_optimized(queries_.data() + (n * query_dim_), topk,
                                            results.data(), ef);
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
  std::filesystem::path index_file = fmt::format("{}_rabitq.hnsw", dir_name_.string());
  std::string_view path = index_file.native();

  /// todo: save and load
  std::shared_ptr<alaya::RBQSpace<>> space =
      std::make_shared<alaya::RBQSpace<>>(points_num_, dim_, MetricType::L2);
  space->fit(data_.data(), points_num_);
  LOG_INFO("Successfully fit data into space");
  if (!std::filesystem::exists(index_file)) {
    auto hnsw = alaya::HNSWBuilder<alaya::RBQSpace<>>(space);
    auto graph = hnsw.build_graph(96);
    graph->save(path);
  }
  // ***************QUERY******************
  /// todo: 此处space没被refine(即space也要save and load)
  auto load_graph = std::make_shared<alaya::Graph<>>();
  load_graph->load(path);

  auto search_job = std::make_unique<alaya::GraphSearchJob<RBQSpace<>>>(space, load_graph);
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
        // results is overwritten
        // search_job->rabitq_search_solo(queries_.data() + (n * query_dim_), topk, results.data(),
        //                                ef);
        search_job->rabitq_search_optimized(queries_.data() + (n * query_dim_), topk,
                                            results.data(), ef);
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

// TEST_F(Deep1MTest, Deep1mQGTest) {
//   // ***************INDEX******************
//   LOG_INFO("Building QG...");
//   std::filesystem::path index_file = fmt::format("{}_rabitq.qg", dir_name_.string());
//   std::string_view path = index_file.native();

//   /// todo: save and load space
//   std::shared_ptr<alaya::RBQSpace<>> space =
//       std::make_shared<alaya::RBQSpace<>>(points_num_, dim_, MetricType::L2);
//   space->fit(data_.data(), points_num_);
//   LOG_INFO("Successfully fit data into space");
//   if (!std::filesystem::exists(index_file)) {
//     auto qg = alaya::QGBuilder<RBQSpace<>>(space);
//     auto graph = qg.build_graph();
//     graph->save(path);
//   }

//   // ***************QUERY******************
//   auto load_graph = std::make_shared<alaya::Graph<>>();
//   load_graph->load(path);

//   auto search_job = std::make_unique<alaya::GraphSearchJob<RBQSpace<>>>(space, load_graph);
//   std::vector<size_t> efs = {10,  20,  40,  50,  60,  80,  100, 150, 170, 190,
//                              200, 250, 300, 400, 500, 600, 700, 800, 1500};
//   size_t test_round = 3;
//   size_t topk = 10;
//   alaya::StopW timer;
//   std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(efs.size()));
//   std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(efs.size()));

//   LOG_INFO("Start querying...");
//   for (size_t r = 0; r < test_round; ++r) {
//     for (size_t i = 0; i < efs.size(); ++i) {  // NOLINT
//       size_t ef = efs[i];
//       size_t total_correct = 0;
//       float total_time = 0;
//       std::vector<IDType> results(topk);
//       LOG_INFO("current ef in this round:{}", ef);
//       for (uint32_t n = 0; n < query_num_; ++n) {
//         timer.reset();
//         // results is overwritten
//         // search_job->rabitq_search_solo(queries_.data() + (n * query_dim_), topk, results.data(),
//         //                                ef);
//         search_job->rabitq_search_optimized(queries_.data() + (n * query_dim_), topk,
//                                             results.data(), ef);
//         total_time += timer.get_elapsed_micro();
//         // recall
//         for (size_t k = 0; k < topk; ++k) {
//           for (size_t j = 0; j < topk; ++j) {
//             if (results[k] == answers_[(n * gt_col_) + j]) {
//               total_correct++;
//               break;
//             }
//           }
//         }
//       }
//       float qps = static_cast<float>(query_num_) / (total_time / 1e6F);
//       float recall = static_cast<float>(total_correct) / static_cast<float>(query_num_ * topk);

//       all_qps[r][i] = qps;
//       all_recall[r][i] = recall;
//     }
//   }

//   auto avg_qps = alaya::horizontal_avg(all_qps);
//   auto avg_recall = alaya::horizontal_avg(all_recall);

//   std::cout << "ef\tQPS\tRecall\n";
//   for (size_t i = 0; i < avg_qps.size(); ++i) {
//     std::cout << efs[i] << '\t' << avg_qps[i] << '\t' << avg_recall[i] << '\n';
//   }
// }
}  // namespace alaya