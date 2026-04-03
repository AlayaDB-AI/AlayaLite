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

#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "executor/jobs/graph_search_job.hpp"
#include "index/graph/qg/qg_builder.hpp"
#include "space/rabitq_space.hpp"
#include "utils/evaluate.hpp"
#include "utils/io_utils.hpp"
#include "utils/timer.hpp"

namespace {

constexpr std::string_view kInternalQueryMode = "__query_only";
constexpr size_t kDegreeBound = 32;
constexpr size_t kEfBuild = 400;
constexpr size_t kTopK = 10;
constexpr size_t kTestRound = 3;

struct DatasetSpec {
  std::string_view name_;
  std::filesystem::path data_file_;
  std::filesystem::path query_file_;
  std::filesystem::path gt_file_;
  std::filesystem::path index_file_;
  alaya::MetricType metric_;
};

struct QueryOnlyDataset {
  std::vector<float> queries_;
  std::vector<uint32_t> ground_truth_;
  uint32_t query_num_ = 0;
  uint32_t gt_num_ = 0;
  uint32_t dim_ = 0;
  uint32_t gt_dim_ = 0;
};

auto try_find_repo_root(const std::filesystem::path &start) -> std::optional<std::filesystem::path> {
  std::filesystem::path cur = start;
  while (!cur.empty()) {
    if (std::filesystem::exists(cur / "CMakeLists.txt") && std::filesystem::exists(cur / "data")) {
      return cur;
    }
    if (cur == cur.root_path()) {
      break;
    }
    cur = cur.parent_path();
  }
  return std::nullopt;
}

auto get_self_path(const char *argv0) -> std::filesystem::path {
  const std::filesystem::path proc_self("/proc/self/exe");
  if (std::filesystem::exists(proc_self)) {
    return std::filesystem::read_symlink(proc_self);
  }
  return std::filesystem::absolute(argv0);
}

auto find_repo_root(const char *argv0) -> std::filesystem::path {
  if (auto root = try_find_repo_root(std::filesystem::current_path()); root.has_value()) {
    return *root;
  }

  const auto self_path = get_self_path(argv0);
  if (auto root = try_find_repo_root(self_path.parent_path()); root.has_value()) {
    return *root;
  }

  throw std::runtime_error("Cannot locate repo root from current path or executable path");
}

auto parse_mode(std::string_view mode, const std::filesystem::path &repo_root) -> DatasetSpec {
  if (mode == "l2") {
    const auto data_dir = repo_root / "data" / "deep1M";
    return DatasetSpec{
        .name_ = "deep1M",
        .data_file_ = data_dir / "deep1M_base.fvecs",
        .query_file_ = data_dir / "deep1M_query.fvecs",
        .gt_file_ = data_dir / "deep1M_groundtruth.ivecs",
        .index_file_ = data_dir / "deep1M_rabitq.qg",
        .metric_ = alaya::MetricType::L2,
    };
  }

  if (mode == "ip") {
    const auto data_dir = repo_root / "data" / "t2i-1m";
    return DatasetSpec{
        .name_ = "t2i-1m",
        .data_file_ = data_dir / "base.fvecs",
        .query_file_ = data_dir / "query.fvecs",
        .gt_file_ = data_dir / "groundtruth.ivecs",
        .index_file_ = data_dir / "t2i1m_rabitq.qg",
        .metric_ = alaya::MetricType::IP,
    };
  }

  throw std::invalid_argument("mode must be one of: l2, ip");
}

auto load_query_only_dataset(const DatasetSpec &spec) -> QueryOnlyDataset {
  QueryOnlyDataset ds;
  alaya::load_fvecs(spec.query_file_, ds.queries_, ds.query_num_, ds.dim_);
  alaya::load_ivecs(spec.gt_file_, ds.ground_truth_, ds.gt_num_, ds.gt_dim_);
  if (ds.query_num_ != ds.gt_num_) {
    throw std::runtime_error("Query count does not match ground truth count");
  }
  return ds;
}

void build_index(const DatasetSpec &spec) {
  std::vector<float> data;
  uint32_t data_num = 0;
  uint32_t dim = 0;
  alaya::load_fvecs(spec.data_file_, data, data_num, dim);

  alaya::Timer timer;
  auto space = std::make_shared<alaya::RaBitQSpace<>>(data_num, dim, spec.metric_);
  space->fit(data.data(), data_num);

  auto builder = alaya::QGBuilder<alaya::RaBitQSpace<>>(space);
  builder.set_ef_build(kEfBuild);
  builder.build_graph();

  const std::string index_path = spec.index_file_.string();
  space->save(index_path);
  std::cout << "Dataset " << spec.name_ << ", indexing time " << timer.elapsed_s() << " secs\n";
}

auto run_query(const DatasetSpec &spec) -> int {
  auto ds = load_query_only_dataset(spec);

  auto space = std::make_shared<alaya::RaBitQSpace<>>();
  const std::string index_path = spec.index_file_.string();
  space->load(index_path);

  if (space->get_dim() != ds.dim_) {
    throw std::runtime_error("Query dimension does not match loaded index dimension");
  }

  auto search_job = std::make_unique<alaya::GraphSearchJob<alaya::RaBitQSpace<>>>(space, nullptr);
  const std::vector<size_t> efs = {
      10, 20, 40, 50, 60, 80, 100, 150, 170, 190, 200, 250, 300, 400, 500, 600, 700, 800, 1500};

  alaya::Timer timer;
  std::vector<std::vector<float>> all_qps(kTestRound, std::vector<float>(efs.size()));
  std::vector<std::vector<float>> all_recall(kTestRound, std::vector<float>(efs.size()));

  for (size_t r = 0; r < kTestRound; ++r) {
    for (size_t i = 0; i < efs.size(); ++i) {
      const size_t ef = efs[i];
      size_t total_correct = 0;
      double total_time = 0;
      std::vector<uint32_t> results(kTopK);

      for (uint32_t n = 0; n < ds.query_num_; ++n) {
        timer.reset();
        search_job->rabitq_search_solo(ds.queries_.data() + (n * ds.dim_), kTopK, results.data(), ef);
        total_time += timer.elapsed_us();

        for (size_t k = 0; k < kTopK; ++k) {
          for (size_t j = 0; j < kTopK; ++j) {
            if (results[k] == ds.ground_truth_[(n * ds.gt_dim_) + j]) {
              total_correct++;
              break;
            }
          }
        }
      }

      all_qps[r][i] = static_cast<float>(ds.query_num_) / static_cast<float>(total_time / 1e6);
      all_recall[r][i] =
          static_cast<float>(total_correct) / static_cast<float>(ds.query_num_ * kTopK);
    }
  }

  const auto avg_qps = alaya::horizontal_avg(all_qps);
  const auto avg_recall = alaya::horizontal_avg(all_recall);

  std::cout << "Dataset\t" << spec.name_ << '\n';
  std::cout << "EF\tQPS\tRecall\n";
  for (size_t i = 0; i < avg_qps.size(); ++i) {
    std::cout << efs[i] << '\t' << avg_qps[i] << '\t' << avg_recall[i] << '\n';
  }
  return 0;
}

auto exec_query_mode(const char *argv0, std::string_view mode) -> int {
  const auto self_path = get_self_path(argv0);
  const std::string self = self_path.string();
  const std::string mode_str(mode);
  if (::execl(self.c_str(), self.c_str(), kInternalQueryMode.data(), mode_str.c_str(), nullptr) ==
      -1) {
    throw std::runtime_error(std::string("execl failed: ") + std::strerror(errno));
  }
  return 1;
}

}  // namespace

auto main(int argc, char **argv) -> int {
#if !defined(__AVX512F__)
  std::cerr << "rabitq_benchmark requires AVX512 build support.\n";
  return 1;
#else
  const auto repo_root = find_repo_root(argv[0]);

  if (argc == 3 && std::string_view(argv[1]) == kInternalQueryMode) {
    return run_query(parse_mode(argv[2], repo_root));
  }

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <l2|ip>\n";
    return 1;
  }

  static_assert(kDegreeBound == alaya::RaBitQSpace<>::kDegreeBound);
  const auto spec = parse_mode(argv[1], repo_root);

  if (!std::filesystem::exists(spec.index_file_)) {
    std::cout << "Index not found, rebuilding " << spec.index_file_ << '\n';
    build_index(spec);
  }

  return exec_query_mode(argv[0], argv[1]);
#endif
}
