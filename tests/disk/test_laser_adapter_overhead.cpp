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
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "index/disk/disk_collection.hpp"
#include "index/disk/segment_factory.hpp"
#include "index/disk/types.hpp"
#include "index/graph/laser/qg/qg.hpp"
#include "utils/metric_type.hpp"

#ifndef ALAYA_LASER_FIXTURE_DIR
  #define ALAYA_LASER_FIXTURE_DIR ""
#endif

#ifndef ALAYA_LASER_FIXTURE_PREFIX
  #define ALAYA_LASER_FIXTURE_PREFIX "dsqg_seg_00000001"
#endif

namespace alaya::disk {
namespace {

constexpr uint32_t kCount = 2048;
constexpr uint32_t kDim = 128;
constexpr uint32_t kR = 64;
constexpr uint32_t kQueries = 50;
constexpr uint32_t kWarmup = 5;
constexpr uint32_t kTopK = 10;
constexpr uint32_t kEf = 64;
constexpr uint32_t kBeamWidth = 4;
constexpr float kDramBudgetGb = 0.5F;

template <typename T>
concept HasImportLaserSegment = requires(T &collection,
                                         const std::filesystem::path &src_dir,
                                         const uint64_t *labels,
                                         uint64_t n) {
  collection.import_laser_segment(src_dir, labels, n);
};

struct Fixture {
  std::filesystem::path dir;
  std::string prefix;
  std::vector<float> vectors;
};

struct Metrics {
  double p50_us = 0.0;
  double p95_us = 0.0;
  double p99_us = 0.0;
  double qps = 0.0;
  double recall_at_10 = 0.0;
  std::vector<std::vector<uint64_t>> labels;
};

auto read_fbin(const std::filesystem::path &path, uint32_t expected_count, uint32_t expected_dim)
    -> std::vector<float> {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open fbin fixture: " + path.string());
  }
  int32_t count = 0;
  int32_t dim = 0;
  input.read(reinterpret_cast<char *>(&count), sizeof(count));
  input.read(reinterpret_cast<char *>(&dim), sizeof(dim));
  if (count != static_cast<int32_t>(expected_count) || dim != static_cast<int32_t>(expected_dim)) {
    throw std::runtime_error("unexpected fbin header in " + path.string());
  }
  std::vector<float> out(static_cast<size_t>(count) * dim);
  input.read(reinterpret_cast<char *>(out.data()),
             static_cast<std::streamsize>(out.size() * sizeof(float)));
  if (!input) {
    throw std::runtime_error("short fbin fixture read: " + path.string());
  }
  return out;
}

auto load_fixture(const std::filesystem::path &dir, const std::string &prefix) -> Fixture {
  const auto input_path = dir / (prefix + "_input.fbin");
  return Fixture{dir, prefix, read_fbin(input_path, kCount, kDim)};
}

auto fixture_has_required_files(const std::filesystem::path &dir, const std::string &prefix)
    -> bool {
  const auto index = dir / (prefix + "_R64_MD128.index");
  const std::vector<std::filesystem::path> required = {
      dir / (prefix + "_input.fbin"),
      index,
      std::filesystem::path(index.string() + "_rotator"),
      std::filesystem::path(index.string() + "_cache_ids"),
      std::filesystem::path(index.string() + "_cache_nodes"),
  };
  std::error_code ec;
  return std::all_of(required.begin(), required.end(), [&](const auto &path) {
    return std::filesystem::is_regular_file(path, ec) && !ec &&
           std::filesystem::file_size(path, ec) > 0 && !ec;
  });
}

auto identity_labels() -> std::vector<uint64_t> {
  std::vector<uint64_t> out(kCount);
  std::iota(out.begin(), out.end(), 0);
  return out;
}

auto make_queries(const std::vector<float> &vectors) -> std::vector<float> {
  std::vector<float> queries((kQueries + kWarmup) * kDim);
  for (uint32_t q = 0; q < kQueries + kWarmup; ++q) {
    const uint32_t row = (q * 37U + 11U) % kCount;
    std::copy_n(vectors.data() + static_cast<size_t>(row) * kDim,
                kDim,
                queries.data() + static_cast<size_t>(q) * kDim);
  }
  return queries;
}

auto l2_sqr(const float *a, const float *b, uint32_t dim) -> float {
  float acc = 0.0F;
  for (uint32_t d = 0; d < dim; ++d) {
    const float delta = a[d] - b[d];
    acc += delta * delta;
  }
  return acc;
}

auto truth_topk(const std::vector<float> &vectors, const float *query)
    -> std::unordered_set<uint64_t> {
  std::vector<std::pair<float, uint64_t>> all;
  all.reserve(kCount);
  for (uint64_t i = 0; i < kCount; ++i) {
    all.emplace_back(l2_sqr(query, vectors.data() + static_cast<size_t>(i) * kDim, kDim), i);
  }
  std::partial_sort(all.begin(), all.begin() + kTopK, all.end());
  std::unordered_set<uint64_t> out;
  out.reserve(kTopK);
  for (uint32_t i = 0; i < kTopK; ++i) {
    out.insert(all[i].second);
  }
  return out;
}

auto recall_against_truth(const std::vector<uint64_t> &labels,
                          const std::unordered_set<uint64_t> &truth) -> double {
  uint32_t matched = 0;
  for (auto label : labels) {
    matched += truth.contains(label) ? 1U : 0U;
  }
  return static_cast<double>(matched) / static_cast<double>(truth.size());
}

auto percentile_us(std::vector<double> values, double p) -> double {
  std::sort(values.begin(), values.end());
  const size_t idx =
      std::min(values.size() - 1, static_cast<size_t>(std::llround(p * (values.size() - 1))));
  return values[idx];
}

auto finish_metrics(const std::vector<double> &latencies_us,
                    double recall_sum,
                    std::vector<std::vector<uint64_t>> labels) -> Metrics {
  const double total_us = std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
  Metrics metrics;
  metrics.p50_us = percentile_us(latencies_us, 0.50);
  metrics.p95_us = percentile_us(latencies_us, 0.95);
  metrics.p99_us = percentile_us(latencies_us, 0.99);
  metrics.qps = static_cast<double>(kQueries) * 1000000.0 / std::max(total_us, 1.0);
  metrics.recall_at_10 = recall_sum / static_cast<double>(kQueries);
  metrics.labels = std::move(labels);
  return metrics;
}

auto run_native_per_query(const Fixture &fixture, const std::vector<float> &queries) -> Metrics {
  alaya::laser::QuantizedGraph graph(kCount, kR, kDim, kDim);
  const auto prefix_path = fixture.dir / fixture.prefix;
  graph.load_disk_index(prefix_path.string().c_str(), kDramBudgetGb);
  graph.set_params(kEf, 1, static_cast<int>(kBeamWidth));

  std::vector<double> latencies_us;
  latencies_us.reserve(kQueries);
  std::vector<std::vector<uint64_t>> labels;
  labels.reserve(kQueries);
  double recall_sum = 0.0;
  std::vector<uint32_t> pids(kTopK);

  for (uint32_t q = 0; q < kQueries + kWarmup; ++q) {
    const float *query = queries.data() + static_cast<size_t>(q) * kDim;
    const auto t0 = std::chrono::steady_clock::now();
    graph.search(query, kTopK, pids.data());
    const auto t1 = std::chrono::steady_clock::now();
    if (q < kWarmup) {
      continue;
    }
    latencies_us.push_back(
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
        1000.0);
    std::vector<uint64_t> row_labels;
    row_labels.reserve(kTopK);
    for (auto pid : pids) {
      row_labels.push_back(pid);
    }
    recall_sum += recall_against_truth(row_labels, truth_topk(fixture.vectors, query));
    labels.push_back(std::move(row_labels));
  }
  return finish_metrics(latencies_us, recall_sum, std::move(labels));
}

auto run_native_batch(const Fixture &fixture, const std::vector<float> &queries) -> Metrics {
  alaya::laser::QuantizedGraph graph(kCount, kR, kDim, kDim);
  const auto prefix_path = fixture.dir / fixture.prefix;
  graph.load_disk_index(prefix_path.string().c_str(), kDramBudgetGb);
  graph.set_params(kEf, 1, static_cast<int>(kBeamWidth));

  std::vector<uint32_t> warmup(kWarmup * kTopK);
  graph.batch_search(queries.data(), kTopK, warmup.data(), kWarmup);

  std::vector<uint32_t> pids(kQueries * kTopK);
  const float *measured_queries = queries.data() + static_cast<size_t>(kWarmup) * kDim;
  const auto t0 = std::chrono::steady_clock::now();
  graph.batch_search(measured_queries, kTopK, pids.data(), kQueries);
  const auto t1 = std::chrono::steady_clock::now();
  const double total_us =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
      1000.0;
  const double per_query_us = total_us / static_cast<double>(kQueries);

  double recall_sum = 0.0;
  std::vector<std::vector<uint64_t>> labels;
  labels.reserve(kQueries);
  for (uint32_t q = 0; q < kQueries; ++q) {
    std::vector<uint64_t> row_labels;
    row_labels.reserve(kTopK);
    for (uint32_t k = 0; k < kTopK; ++k) {
      row_labels.push_back(pids[static_cast<size_t>(q) * kTopK + k]);
    }
    const float *query = measured_queries + static_cast<size_t>(q) * kDim;
    recall_sum += recall_against_truth(row_labels, truth_topk(fixture.vectors, query));
    labels.push_back(std::move(row_labels));
  }

  Metrics metrics;
  metrics.p50_us = per_query_us;
  metrics.p95_us = per_query_us;
  metrics.p99_us = per_query_us;
  metrics.qps = static_cast<double>(kQueries) * 1000000.0 / std::max(total_us, 1.0);
  metrics.recall_at_10 = recall_sum / static_cast<double>(kQueries);
  metrics.labels = std::move(labels);
  return metrics;
}

template <typename Collection>
auto run_adapter_per_query(const Fixture &fixture, const std::vector<float> &queries)
    -> std::optional<Metrics> {
  if constexpr (!HasImportLaserSegment<Collection>) {
    return std::nullopt;
  } else {
    const auto root = std::filesystem::temp_directory_path() /
                      ("alaya_laser_adapter_overhead_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root);

    Collection collection(root, kDim, MetricType::L2, DiskIndexType::Laser);
    auto labels_in = identity_labels();
    collection.import_laser_segment(fixture.dir, labels_in.data(), labels_in.size());

    DiskSearchOptions opts;
    opts.top_k = kTopK;
    opts.ef = kEf;
    opts.beam_width = kBeamWidth;

    std::vector<double> latencies_us;
    latencies_us.reserve(kQueries);
    std::vector<std::vector<uint64_t>> labels;
    labels.reserve(kQueries);
    double recall_sum = 0.0;

    for (uint32_t q = 0; q < kQueries + kWarmup; ++q) {
      const float *query = queries.data() + static_cast<size_t>(q) * kDim;
      const auto t0 = std::chrono::steady_clock::now();
      auto hits = collection.search(query, opts);
      const auto t1 = std::chrono::steady_clock::now();
      if (q < kWarmup) {
        continue;
      }
      latencies_us.push_back(
          static_cast<double>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
          1000.0);
      std::vector<uint64_t> row_labels;
      row_labels.reserve(hits.size());
      for (const auto &hit : hits) {
        row_labels.push_back(hit.label);
      }
      recall_sum += recall_against_truth(row_labels, truth_topk(fixture.vectors, query));
      labels.push_back(std::move(row_labels));
    }

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    return finish_metrics(latencies_us, recall_sum, std::move(labels));
  }
}

void print_report_line(const char *path,
                       const Metrics &metrics,
                       double overhead_pct,
                       double overhead_us) {
  std::cout << path << "," << metrics.p50_us << "," << metrics.p95_us << "," << metrics.p99_us
            << "," << metrics.qps << "," << metrics.recall_at_10 << "," << overhead_pct << ","
            << overhead_us << "\n";
}

}  // namespace

TEST(LaserAdapterOverheadTest, native_laser_vs_adapter_equivalence) {
  if constexpr (!HasImportLaserSegment<DiskCollection>) {
    GTEST_SKIP() << "DiskCollection::import_laser_segment is not available yet";
  }
  if (!engine_supported_v1(DiskIndexType::Laser)) {
    GTEST_SKIP() << "disk_laser is not registered in this build";
  }

  const auto fixture_dir = std::filesystem::path(ALAYA_LASER_FIXTURE_DIR);
  const auto fixture_prefix = std::string(ALAYA_LASER_FIXTURE_PREFIX);
  if (fixture_dir.empty() || !fixture_has_required_files(fixture_dir, fixture_prefix)) {
    GTEST_SKIP() << "LASER fixture is missing or incomplete under " << fixture_dir;
  }

  const auto fixture = load_fixture(fixture_dir, fixture_prefix);
  const auto queries = make_queries(fixture.vectors);

  const auto native = run_native_per_query(fixture, queries);
  const auto adapter_or_skip = run_adapter_per_query<DiskCollection>(fixture, queries);
  if (!adapter_or_skip.has_value()) {
    GTEST_SKIP() << "DiskCollection::import_laser_segment is not available yet";
  }
  const auto &adapter = *adapter_or_skip;
  const auto native_batch = run_native_batch(fixture, queries);

  const double overhead_us = adapter.p50_us - native.p50_us;
  const double overhead_pct = overhead_us / std::max(native.p50_us, 1.0);

  std::cout << "path,p50_us,p95_us,p99_us,qps,recall_at_10,overhead_pct,overhead_us\n";
  print_report_line("native_per_query", native, 0.0, 0.0);
  print_report_line("adapter_per_query", adapter, overhead_pct, overhead_us);
  print_report_line("native_batch_context", native_batch, 0.0, 0.0);

  EXPECT_LE(std::abs(native.recall_at_10 - adapter.recall_at_10), 0.001);
  EXPECT_TRUE(overhead_pct <= 0.03 || overhead_us <= 20.0)
      << "native p50=" << native.p50_us << "us adapter p50=" << adapter.p50_us
      << "us overhead_us=" << overhead_us << " overhead_pct=" << overhead_pct;
}

}  // namespace alaya::disk
