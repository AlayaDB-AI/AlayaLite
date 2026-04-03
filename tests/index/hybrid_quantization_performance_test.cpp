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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "executor/jobs/graph_hybrid_search_job.hpp"
#include "index/graph/graph.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "index/graph/qg/qg_builder.hpp"
#include "space/rabitq_space.hpp"
#include "space/raw_space.hpp"
#include "space/sq4_space.hpp"
#include "space/sq8_space.hpp"
#include "storage/sequential_storage.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/log.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/scalar_data.hpp"
#include "utils/timer.hpp"

namespace alaya {

namespace {

using IDType = uint32_t;
using DataType = float;
using DistanceType = float;

using RawBuildSpace =
    RawSpace<DataType, DistanceType, IDType, SequentialStorage<DataType, IDType>, EmptyScalarData>;
using RawSearchSpace =
    RawSpace<DataType, DistanceType, IDType, SequentialStorage<DataType, IDType>, ScalarData>;
using SQ8SearchSpace =
    SQ8Space<DataType, DistanceType, IDType, SequentialStorage<uint8_t, IDType>, ScalarData>;
using SQ4SearchSpace =
    SQ4Space<DataType, DistanceType, IDType, SequentialStorage<uint8_t, IDType>, ScalarData>;
using RaBitQSearchSpace = RaBitQSpace<DataType, DistanceType, IDType, ScalarData>;

constexpr uint32_t kTopK = 100;
constexpr uint32_t kEfSearch = 160;
constexpr uint32_t kMaxNbrs = 32;
constexpr uint32_t kDefaultQueryLimit = 50;

struct BenchmarkSettings {
  uint32_t query_limit_ = kDefaultQueryLimit;
  uint32_t data_limit_ = 0;
  uint32_t target_count_ = 0;
  std::vector<uint32_t> filter_counts_;
};

struct FilterScenario {
  std::string label_;
  uint32_t upper_bound_ = 0;
  double selectivity_ = 0.0;
  MetadataFilter full_filter_;
  MetadataFilter residual_filter_;
};

struct BaselineResult {
  double build_seconds_ = 0.0;
  double query_seconds_ = 0.0;
  double avg_ms_per_query_ = 0.0;
  double qps_ = 0.0;
  std::vector<IDType> ids_;
};

struct BenchmarkResult {
  std::string name_;
  double build_seconds_ = 0.0;
  double query_seconds_ = 0.0;
  double avg_ms_per_query_ = 0.0;
  double qps_ = 0.0;
  double speedup_vs_bf_ = 0.0;
  float recall_ = 0.0F;
  bool available_ = true;
  std::string note_;
};

template <typename SearchSpaceType, typename BuildSpaceType>
struct PartitionBundle {
  std::string name_;
  std::shared_ptr<SearchSpaceType> search_space_{nullptr};
  std::shared_ptr<BuildSpaceType> build_space_{nullptr};
  std::shared_ptr<Graph<DataType, IDType>> graph_{nullptr};
  double build_seconds_ = 0.0;
  bool available_ = true;
  std::string note_;
};

auto parse_env_u32(const char *name, uint32_t default_value) -> uint32_t {
  const char *raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return default_value;
  }
  try {
    return static_cast<uint32_t>(std::stoul(raw));
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("invalid value for ") + name + ": " + e.what());
  }
}

auto parse_env_counts(const char *name) -> std::vector<uint32_t> {
  const char *raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return {};
  }

  std::vector<uint32_t> values;
  std::stringstream ss(raw);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    try {
      values.push_back(static_cast<uint32_t>(std::stoul(token)));
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("invalid value in ") + name + ": " + e.what());
    }
  }
  return values;
}

auto load_benchmark_settings() -> BenchmarkSettings {
  BenchmarkSettings settings;
  settings.query_limit_ = parse_env_u32("ALAYALITE_HYBRID_PERF_QUERY_LIMIT", kDefaultQueryLimit);
  settings.data_limit_ = parse_env_u32("ALAYALITE_HYBRID_PERF_DATA_LIMIT", 0);
  settings.target_count_ = parse_env_u32("ALAYALITE_HYBRID_PERF_TARGET_COUNT", 0);
  settings.filter_counts_ = parse_env_counts("ALAYALITE_HYBRID_PERF_FILTER_COUNTS");
  return settings;
}

auto resolve_data_dir() -> std::filesystem::path {
  if (const char *env = std::getenv("ALAYALITE_DATA_DIR"); env != nullptr && *env != '\0') {
    return env;
  }

  auto current = std::filesystem::current_path();
  while (true) {
    auto candidate = current / "data";
    if (std::filesystem::exists(candidate / "deep1M")) {
      return candidate;
    }
    auto parent = current.parent_path();
    if (parent == current) {
      break;
    }
    current = parent;
  }

  return std::filesystem::current_path() / "data";
}

void normalize_counts(std::vector<uint32_t> &counts, uint32_t min_value, uint32_t max_value) {
  for (auto &count : counts) {
    count = std::max(min_value, std::min(count, max_value));
  }
  std::ranges::sort(counts);
  counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
}

auto build_scalar_data(uint32_t data_num, uint32_t target_count) -> std::vector<ScalarData> {
  std::vector<ScalarData> scalar_data;
  scalar_data.reserve(data_num);
  for (uint32_t i = 0; i < data_num; ++i) {
    MetadataMap metadata;
    metadata["label"] = std::string(i < target_count ? "target_label" : "other");
    metadata["id"] = static_cast<int64_t>(i);
    scalar_data.emplace_back(std::to_string(i), "", std::move(metadata));
  }
  return scalar_data;
}

auto build_filter_scenarios(const std::vector<uint32_t> &filter_counts,
                            uint32_t target_count,
                            uint32_t data_num) -> std::vector<FilterScenario> {
  std::vector<FilterScenario> scenarios;
  scenarios.reserve(filter_counts.size());

  for (auto upper_bound : filter_counts) {
    FilterScenario scenario;
    scenario.upper_bound_ = upper_bound;
    scenario.selectivity_ = static_cast<double>(upper_bound) / static_cast<double>(data_num);
    scenario.label_ = fmt::format("label=target_label AND id<{} ({:.3f}%)",
                                  upper_bound,
                                  scenario.selectivity_ * 100.0);

    scenario.full_filter_.add_eq("label", std::string("target_label"));
    scenario.full_filter_.add_lt("id", static_cast<int64_t>(upper_bound));

    if (upper_bound < target_count) {
      scenario.residual_filter_.add_lt("id", static_cast<int64_t>(upper_bound));
    } else {
      scenario.residual_filter_ = MetadataFilter::empty();
    }

    scenarios.push_back(std::move(scenario));
  }
  return scenarios;
}

auto slice_vectors(const std::vector<DataType> &vectors, uint32_t vector_num, uint32_t dim)
    -> std::vector<DataType> {
  auto end = static_cast<size_t>(vector_num) * dim;
  return std::vector<DataType>(vectors.begin(), vectors.begin() + static_cast<std::ptrdiff_t>(end));
}

auto slice_scalar_data(const std::vector<ScalarData> &scalar_data, uint32_t count)
    -> std::vector<ScalarData> {
  return std::vector<ScalarData>(scalar_data.begin(), scalar_data.begin() + count);
}

auto make_dummy_graph() -> std::shared_ptr<Graph<DataType, IDType>> {
  auto graph = std::make_shared<Graph<DataType, IDType>>(1, 1);
  graph->eps_.push_back(0);
  return graph;
}

template <typename SearchSpaceType>
auto make_search_space(const std::vector<DataType> &vectors,
                       uint32_t vector_num,
                       uint32_t dim,
                       const std::vector<ScalarData> &scalar_data,
                       const std::filesystem::path &db_path,
                       const std::vector<std::string> &indexed_fields,
                       double *build_seconds = nullptr) -> std::shared_ptr<SearchSpaceType> {
  RocksDBConfig config;
  config.db_path_ = db_path.string();
  config.indexed_fields_ = indexed_fields;

  Timer timer;
  auto search_space = std::make_shared<SearchSpaceType>(vector_num, dim, MetricType::L2, config);
  search_space->fit(vectors.data(), vector_num, scalar_data.data());
  if (build_seconds != nullptr) {
    *build_seconds = timer.elapsed_us() / 1e6;
  }
  return search_space;
}

auto make_raw_build_space(const std::vector<DataType> &vectors, uint32_t vector_num, uint32_t dim)
    -> std::shared_ptr<RawBuildSpace> {
  auto build_space = std::make_shared<RawBuildSpace>(vector_num, dim, MetricType::L2);
  build_space->fit(vectors.data(), vector_num);
  return build_space;
}

auto build_hnsw_graph(const std::shared_ptr<RawBuildSpace> &build_space,
                      uint32_t num_threads) -> std::shared_ptr<Graph<DataType, IDType>> {
  HNSWBuilder<RawBuildSpace> hnsw(build_space, kMaxNbrs, 200);
  return std::shared_ptr<Graph<DataType, IDType>>(hnsw.build_graph(num_threads).release());
}

auto build_raw_partition_bundle(const std::vector<DataType> &vectors,
                                uint32_t vector_num,
                                uint32_t dim,
                                const std::vector<ScalarData> &scalar_data,
                                const std::filesystem::path &db_path,
                                uint32_t num_threads)
    -> PartitionBundle<RawSearchSpace, RawSearchSpace> {
  PartitionBundle<RawSearchSpace, RawSearchSpace> bundle;
  bundle.name_ = "raw_mv";

  Timer timer;
  bundle.search_space_ = make_search_space<RawSearchSpace>(
      vectors, vector_num, dim, scalar_data, db_path, {"label", "id"}, nullptr);
  HNSWBuilder<RawSearchSpace> hnsw(bundle.search_space_, kMaxNbrs, 200);
  bundle.graph_ =
      std::shared_ptr<Graph<DataType, IDType>>(hnsw.build_graph(num_threads).release());
  bundle.build_space_ = bundle.search_space_;
  bundle.build_seconds_ = timer.elapsed_us() / 1e6;
  return bundle;
}

auto build_sq8_partition_bundle(const std::vector<DataType> &vectors,
                                uint32_t vector_num,
                                uint32_t dim,
                                const std::vector<ScalarData> &scalar_data,
                                const std::filesystem::path &db_path,
                                uint32_t num_threads)
    -> PartitionBundle<SQ8SearchSpace, RawBuildSpace> {
  PartitionBundle<SQ8SearchSpace, RawBuildSpace> bundle;
  bundle.name_ = "sq8_mv";

  Timer timer;
  bundle.build_space_ = make_raw_build_space(vectors, vector_num, dim);
  bundle.graph_ = build_hnsw_graph(bundle.build_space_, num_threads);
  bundle.search_space_ = make_search_space<SQ8SearchSpace>(
      vectors, vector_num, dim, scalar_data, db_path, {"id"}, nullptr);
  bundle.build_seconds_ = timer.elapsed_us() / 1e6;
  return bundle;
}

auto build_sq4_partition_bundle(const std::vector<DataType> &vectors,
                                uint32_t vector_num,
                                uint32_t dim,
                                const std::vector<ScalarData> &scalar_data,
                                const std::filesystem::path &db_path,
                                uint32_t num_threads)
    -> PartitionBundle<SQ4SearchSpace, RawBuildSpace> {
  PartitionBundle<SQ4SearchSpace, RawBuildSpace> bundle;
  bundle.name_ = "sq4_mv";

  Timer timer;
  bundle.build_space_ = make_raw_build_space(vectors, vector_num, dim);
  bundle.graph_ = build_hnsw_graph(bundle.build_space_, num_threads);
  bundle.search_space_ = make_search_space<SQ4SearchSpace>(
      vectors, vector_num, dim, scalar_data, db_path, {"id"}, nullptr);
  bundle.build_seconds_ = timer.elapsed_us() / 1e6;
  return bundle;
}

auto build_rabitq_partition_bundle(const std::vector<DataType> &vectors,
                                   uint32_t vector_num,
                                   uint32_t dim,
                                   const std::vector<ScalarData> &scalar_data,
                                   const std::filesystem::path &db_path)
    -> PartitionBundle<RaBitQSearchSpace, RaBitQSearchSpace> {
  PartitionBundle<RaBitQSearchSpace, RaBitQSearchSpace> bundle;
  bundle.name_ = "rabitq_mv";

#if !defined(__AVX512F__)
  (void)vectors;
  (void)vector_num;
  (void)dim;
  (void)scalar_data;
  (void)db_path;
  bundle.available_ = false;
  bundle.note_ = "requires AVX512";
  return bundle;
#else
  Timer timer;
  RocksDBConfig config;
  config.db_path_ = db_path.string();
  config.indexed_fields_ = {"id"};
  bundle.search_space_ =
      std::make_shared<RaBitQSearchSpace>(vector_num, dim, MetricType::L2, config);
  bundle.search_space_->fit(vectors.data(), vector_num, scalar_data.data());
  QGBuilder<RaBitQSearchSpace> qg_builder(bundle.search_space_);
  qg_builder.build_graph();
  bundle.build_space_ = bundle.search_space_;
  bundle.build_seconds_ = timer.elapsed_us() / 1e6;
  return bundle;
#endif
}

void decode_item_ids_to_global_ids(const std::vector<std::string> &item_ids, IDType *ids_out) {
  std::fill(ids_out, ids_out + item_ids.size(), std::numeric_limits<IDType>::max());
  for (size_t i = 0; i < item_ids.size(); ++i) {
    if (item_ids[i].empty()) {
      continue;
    }
    ids_out[i] = static_cast<IDType>(std::stoul(item_ids[i]));
  }
}

auto benchmark_global_bruteforce(const Dataset &ds,
                                 const MetadataFilter &filter,
                                 const std::shared_ptr<RawSearchSpace> &space,
                                 double build_seconds) -> BaselineResult {
  BaselineResult result;
  result.build_seconds_ = build_seconds;
  result.ids_.resize(static_cast<size_t>(ds.query_num_) * kTopK, std::numeric_limits<IDType>::max());

  auto job = std::make_shared<GraphHybridSearchJob<RawSearchSpace, RawSearchSpace>>(
      space, make_dummy_graph(), space);

  std::vector<std::string> item_ids(kTopK);
  Timer timer;
  for (uint32_t i = 0; i < ds.query_num_; ++i) {
    auto *query = const_cast<DataType *>(ds.queries_.data() + static_cast<size_t>(i) * ds.dim_);
    auto *out = result.ids_.data() + static_cast<size_t>(i) * kTopK;
    job->hybrid_search_brute_force_solo(query, out, kTopK, filter, item_ids.data());
  }

  result.query_seconds_ = timer.elapsed_us() / 1e6;
  result.avg_ms_per_query_ = result.query_seconds_ * 1000.0 / static_cast<double>(ds.query_num_);
  result.qps_ = static_cast<double>(ds.query_num_) / result.query_seconds_;
  return result;
}

template <typename SearchSpaceType, typename BuildSpaceType>
auto benchmark_partition_bundle(const Dataset &ds,
                                const MetadataFilter &residual_filter,
                                const std::vector<IDType> &gt,
                                const PartitionBundle<SearchSpaceType, BuildSpaceType> &bundle,
                                double bf_query_seconds) -> BenchmarkResult {
  BenchmarkResult result;
  result.name_ = bundle.name_;
  result.build_seconds_ = bundle.build_seconds_;
  result.available_ = bundle.available_;
  result.note_ = bundle.note_;

  if (!result.available_) {
    return result;
  }

  std::vector<IDType> all_ids(static_cast<size_t>(ds.query_num_) * kTopK,
                              std::numeric_limits<IDType>::max());
  std::vector<IDType> local_ids(kTopK, std::numeric_limits<IDType>::max());
  std::vector<std::string> item_ids(kTopK);

  Timer timer;
  if constexpr (is_rabitq_space_v<SearchSpaceType>) {
    auto job = std::make_shared<GraphHybridSearchJob<SearchSpaceType>>(bundle.search_space_);
    for (uint32_t i = 0; i < ds.query_num_; ++i) {
      auto *query = const_cast<DataType *>(ds.queries_.data() + static_cast<size_t>(i) * ds.dim_);
      std::fill(local_ids.begin(), local_ids.end(), std::numeric_limits<IDType>::max());
      job->rabitq_hybrid_search_solo(
          query, kTopK, local_ids.data(), kEfSearch, residual_filter, item_ids.data());
      decode_item_ids_to_global_ids(item_ids, all_ids.data() + static_cast<size_t>(i) * kTopK);
    }
  } else {
    auto job = std::make_shared<GraphHybridSearchJob<SearchSpaceType, BuildSpaceType>>(
        bundle.search_space_, bundle.graph_, bundle.build_space_);
    for (uint32_t i = 0; i < ds.query_num_; ++i) {
      auto *query = const_cast<DataType *>(ds.queries_.data() + static_cast<size_t>(i) * ds.dim_);
      std::fill(local_ids.begin(), local_ids.end(), std::numeric_limits<IDType>::max());
      job->hybrid_search_solo(
          query, local_ids.data(), kTopK, kEfSearch, residual_filter, item_ids.data());
      decode_item_ids_to_global_ids(item_ids, all_ids.data() + static_cast<size_t>(i) * kTopK);
    }
  }

  result.query_seconds_ = timer.elapsed_us() / 1e6;
  result.avg_ms_per_query_ = result.query_seconds_ * 1000.0 / static_cast<double>(ds.query_num_);
  result.qps_ = static_cast<double>(ds.query_num_) / result.query_seconds_;
  result.speedup_vs_bf_ = bf_query_seconds / result.query_seconds_;
  result.recall_ = calc_recall(all_ids.data(), gt.data(), ds.query_num_, kTopK, kTopK);
  return result;
}

void print_results(const Dataset &ds,
                   uint32_t target_count,
                   const FilterScenario &scenario,
                   const BaselineResult &baseline,
                   const std::vector<BenchmarkResult> &results) {
  std::cout << "\n===== Simple MV vs Brute Force =====\n";
  std::cout << fmt::format("dataset={} data={} dim={} topk={} ef={} queries={}\n",
                           ds.name_,
                           ds.data_num_,
                           ds.dim_,
                           kTopK,
                           kEfSearch,
                           ds.query_num_);
  std::cout << fmt::format("target_partition=label=target_label size={} ({:.3f}%)\n",
                           target_count,
                           static_cast<double>(target_count) * 100.0 /
                               static_cast<double>(ds.data_num_));
  std::cout << "mv_model=manual route on label partition, residual filter evaluated inside partition\n";
  std::cout << fmt::format("filter={}\n", scenario.label_);
  std::cout << fmt::format("bf_exact\tbuild_s={:.3f}\tquery_s={:.3f}\tavg_ms={:.3f}\tqps={:.1f}\n",
                           baseline.build_seconds_,
                           baseline.query_seconds_,
                           baseline.avg_ms_per_query_,
                           baseline.qps_);
  std::cout << "quant\tmv_build_s\tmv_query_s\tavg_ms\tqps\tspeedup_vs_bf\trecall_vs_bf\tnote\n";
  for (const auto &result : results) {
    if (!result.available_) {
      std::cout << fmt::format("{}\t-\t-\t-\t-\t-\t-\t{}\n",
                               result.name_,
                               result.note_);
      continue;
    }
    std::cout << fmt::format("{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.1f}\t{:.2f}x\t{:.4f}\t{}\n",
                             result.name_,
                             result.build_seconds_,
                             result.query_seconds_,
                             result.avg_ms_per_query_,
                             result.qps_,
                             result.speedup_vs_bf_,
                             result.recall_,
                             result.note_);
  }
  std::cout << "====================================\n";
}

}  // namespace

class HybridQuantizationPerformanceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    settings_ = load_benchmark_settings();
    previous_log_level_ = spdlog::default_logger()->level();
    spdlog::set_level(spdlog::level::warn);

    auto data_dir = resolve_data_dir();
    config_ = deep1m(data_dir);
    config_.max_query_num_ = settings_.query_limit_;
    if (settings_.data_limit_ != 0) {
      config_.max_data_num_ = settings_.data_limit_;
    }
    ds_ = load_dataset(config_);

    if (ds_.data_num_ <= kTopK) {
      throw std::runtime_error("dataset too small for topk benchmark");
    }

    settings_.target_count_ = settings_.target_count_ == 0
                                  ? std::max<uint32_t>(kTopK, ds_.data_num_ / 100)
                                  : settings_.target_count_;
    settings_.target_count_ =
        std::max<uint32_t>(kTopK, std::min<uint32_t>(settings_.target_count_, ds_.data_num_ - 1));

    if (settings_.filter_counts_.empty()) {
      settings_.filter_counts_ = {settings_.target_count_,
                                  settings_.target_count_ / 2,
                                  settings_.target_count_ / 5,
                                  settings_.target_count_ / 10};
    }
    normalize_counts(settings_.filter_counts_, kTopK, settings_.target_count_);

    scalar_data_ = build_scalar_data(ds_.data_num_, settings_.target_count_);
    target_vectors_ = slice_vectors(ds_.data_, settings_.target_count_, ds_.dim_);
    target_scalar_data_ = slice_scalar_data(scalar_data_, settings_.target_count_);
    scenarios_ =
        build_filter_scenarios(settings_.filter_counts_, settings_.target_count_, ds_.data_num_);

    work_dir_ = std::filesystem::temp_directory_path() / "alayalite_simple_mv_perf";
    std::filesystem::remove_all(work_dir_);
    std::filesystem::create_directories(work_dir_);
  }

  void TearDown() override {
    spdlog::set_level(previous_log_level_);
    if (std::filesystem::exists(work_dir_)) {
      std::filesystem::remove_all(work_dir_);
    }
  }

  BenchmarkSettings settings_;
  DatasetConfig config_;
  Dataset ds_;
  std::vector<ScalarData> scalar_data_;
  std::vector<DataType> target_vectors_;
  std::vector<ScalarData> target_scalar_data_;
  std::vector<FilterScenario> scenarios_;
  std::filesystem::path work_dir_;
  spdlog::level::level_enum previous_log_level_ = spdlog::level::info;
};

TEST_F(HybridQuantizationPerformanceTest, CompareSimpleMvAgainstBruteForceAcrossQuantization) {
  ASSERT_FALSE(scenarios_.empty());
  ASSERT_GE(settings_.target_count_, kTopK);

  double raw_full_build_seconds = 0.0;
  auto raw_full_space = make_search_space<RawSearchSpace>(
      ds_.data_, ds_.data_num_, ds_.dim_, scalar_data_, work_dir_ / "raw_full_db", {"label", "id"},
      &raw_full_build_seconds);

  auto raw_bundle = build_raw_partition_bundle(target_vectors_,
                                               settings_.target_count_,
                                               ds_.dim_,
                                               target_scalar_data_,
                                               work_dir_ / "raw_partition_db",
                                               std::max<uint32_t>(1, std::thread::hardware_concurrency()));
  auto sq8_bundle = build_sq8_partition_bundle(target_vectors_,
                                               settings_.target_count_,
                                               ds_.dim_,
                                               target_scalar_data_,
                                               work_dir_ / "sq8_partition_db",
                                               std::max<uint32_t>(1, std::thread::hardware_concurrency()));
  auto sq4_bundle = build_sq4_partition_bundle(target_vectors_,
                                               settings_.target_count_,
                                               ds_.dim_,
                                               target_scalar_data_,
                                               work_dir_ / "sq4_partition_db",
                                               std::max<uint32_t>(1, std::thread::hardware_concurrency()));
  auto rabitq_bundle = build_rabitq_partition_bundle(target_vectors_,
                                                     settings_.target_count_,
                                                     ds_.dim_,
                                                     target_scalar_data_,
                                                     work_dir_ / "rabitq_partition_db");

  for (const auto &scenario : scenarios_) {
    auto baseline = benchmark_global_bruteforce(
        ds_, scenario.full_filter_, raw_full_space, raw_full_build_seconds);

    std::vector<BenchmarkResult> results;
    results.push_back(benchmark_partition_bundle(
        ds_, scenario.residual_filter_, baseline.ids_, raw_bundle, baseline.query_seconds_));
    results.push_back(benchmark_partition_bundle(
        ds_, scenario.residual_filter_, baseline.ids_, sq8_bundle, baseline.query_seconds_));
    results.push_back(benchmark_partition_bundle(
        ds_, scenario.residual_filter_, baseline.ids_, sq4_bundle, baseline.query_seconds_));
    results.push_back(benchmark_partition_bundle(
        ds_, scenario.residual_filter_, baseline.ids_, rabitq_bundle, baseline.query_seconds_));

    print_results(ds_, settings_.target_count_, scenario, baseline, results);

    EXPECT_GT(baseline.query_seconds_, 0.0);
    EXPECT_GT(baseline.qps_, 0.0);
    for (const auto &result : results) {
      if (!result.available_) {
        continue;
      }
      EXPECT_GT(result.query_seconds_, 0.0);
      EXPECT_GT(result.qps_, 0.0);
      EXPECT_GE(result.recall_, 0.0F);
    }
  }

  raw_full_space->close_db();
  raw_bundle.search_space_->close_db();
  sq8_bundle.search_space_->close_db();
  sq4_bundle.search_space_->close_db();
  if (rabitq_bundle.available_ && rabitq_bundle.search_space_ != nullptr) {
    rabitq_bundle.search_space_->close_db();
  }
}

}  // namespace alaya
