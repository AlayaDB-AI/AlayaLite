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
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "index/disk/disk_collection.hpp"
#include "index/disk/types.hpp"
#include "index/graph/vamana/vamana_greedy_search.hpp"
#include "index/graph/vamana/vamana_reader.hpp"
#include "simd/distance_l2.hpp"
#include "utils/metric_type.hpp"

namespace alaya::disk {
namespace {

auto make_vectors(uint64_t n, uint32_t dim, uint32_t seed) -> std::vector<float> {
  std::vector<float> out(n * dim);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  for (auto &v : out) {
    v = dist(rng);
  }
  return out;
}

auto identity_labels(uint64_t n) -> std::vector<uint64_t> {
  std::vector<uint64_t> out(n);
  std::iota(out.begin(), out.end(), 0);
  return out;
}

auto percentile_us(std::vector<double> values, double p) -> double {
  std::sort(values.begin(), values.end());
  const size_t idx = std::min(values.size() - 1,
                              static_cast<size_t>(std::llround(p * (values.size() - 1))));
  return values[idx];
}

auto truth_topk(const std::vector<float> &vectors,
                const std::vector<float> &query,
                uint32_t dim,
                uint32_t top_k) -> std::unordered_set<uint64_t> {
  const uint64_t n = vectors.size() / dim;
  std::vector<std::pair<float, uint64_t>> all;
  all.reserve(n);
  for (uint64_t i = 0; i < n; ++i) {
    const float d = alaya::simd::l2_sqr<float, float>(
        query.data(), vectors.data() + static_cast<size_t>(i) * dim, dim);
    all.emplace_back(d, i);
  }
  std::partial_sort(all.begin(), all.begin() + top_k, all.end());
  std::unordered_set<uint64_t> out;
  for (uint32_t i = 0; i < top_k; ++i) {
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

}  // namespace

TEST(VamanaAdapterOverheadTest, direct_greedy_vs_disk_collection_adapter) {
  constexpr uint32_t kDim = 16;
  constexpr uint32_t kN = 2048;
  constexpr uint32_t kQueries = 50;
  constexpr uint32_t kWarmup = 5;
  constexpr uint32_t kTopK = 10;
  constexpr uint32_t kEf = 64;

  const auto root = std::filesystem::temp_directory_path() /
                    ("alaya_vamana_overhead_" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  const auto coll_path = root / "coll";

  auto vectors = make_vectors(kN, kDim, 42);
  auto ids = identity_labels(kN);
  std::vector<std::vector<float>> queries;
  queries.reserve(kQueries + kWarmup);
  auto qbuf = make_vectors(kQueries + kWarmup, kDim, 43);
  for (uint32_t q = 0; q < kQueries + kWarmup; ++q) {
    queries.emplace_back(qbuf.begin() + static_cast<size_t>(q) * kDim,
                         qbuf.begin() + static_cast<size_t>(q + 1) * kDim);
  }

  DiskCollection collection(coll_path, kDim, MetricType::L2, DiskIndexType::Vamana);
  collection.add_batch(vectors.data(), ids.data(), ids.size());
  collection.flush();

  const auto graph_path = coll_path / "segments" / "seg_00000001" / "graph.index";
  alaya::vamana::VamanaReader reader(graph_path);
  alaya::vamana::VamanaGreedySearch direct(reader, vectors.data(), kDim);

  DiskSearchOptions opts;
  opts.top_k = kTopK;
  opts.ef = kEf;

  std::vector<double> direct_us;
  std::vector<double> adapter_us;
  direct_us.reserve(kQueries);
  adapter_us.reserve(kQueries);
  double recall_direct = 0.0;
  double recall_adapter = 0.0;
  uint32_t equivalent = 0;

  for (uint32_t q = 0; q < kQueries + kWarmup; ++q) {
    const auto &query = queries[q];

    auto t0 = std::chrono::steady_clock::now();
    auto direct_hits = direct.search(query.data(), kTopK, kEf);
    auto t1 = std::chrono::steady_clock::now();
    auto adapter_hits = collection.search(query.data(), opts);
    auto t2 = std::chrono::steady_clock::now();

    if (q < kWarmup) {
      continue;
    }

    direct_us.push_back(
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) /
        1000.0);
    adapter_us.push_back(
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) /
        1000.0);

    auto truth = truth_topk(vectors, query, kDim, kTopK);
    std::vector<uint64_t> direct_labels;
    direct_labels.reserve(direct_hits.size());
    for (const auto &hit : direct_hits) {
      direct_labels.push_back(hit.id);
    }
    std::vector<uint64_t> adapter_labels;
    adapter_labels.reserve(adapter_hits.size());
    for (const auto &hit : adapter_hits) {
      adapter_labels.push_back(hit.label);
    }
    recall_direct += recall_against_truth(direct_labels, truth);
    recall_adapter += recall_against_truth(adapter_labels, truth);

    std::sort(direct_labels.begin(), direct_labels.end());
    std::sort(adapter_labels.begin(), adapter_labels.end());
    if (direct_labels == adapter_labels) {
      equivalent++;
    }
  }

  recall_direct /= kQueries;
  recall_adapter /= kQueries;
  const double p50_direct = percentile_us(direct_us, 0.50);
  const double p95_direct = percentile_us(direct_us, 0.95);
  const double p99_direct = percentile_us(direct_us, 0.99);
  const double p50_adapter = percentile_us(adapter_us, 0.50);
  const double p95_adapter = percentile_us(adapter_us, 0.95);
  const double p99_adapter = percentile_us(adapter_us, 0.99);
  const double direct_total_us =
      std::accumulate(direct_us.begin(), direct_us.end(), 0.0);
  const double adapter_total_us =
      std::accumulate(adapter_us.begin(), adapter_us.end(), 0.0);
  const double direct_qps =
      static_cast<double>(kQueries) * 1000000.0 / std::max(direct_total_us, 1.0);
  const double adapter_qps =
      static_cast<double>(kQueries) * 1000000.0 / std::max(adapter_total_us, 1.0);
  const double delta_us = p50_adapter - p50_direct;
  const double ratio = delta_us / std::max(p50_direct, 1.0);

  std::cout << "direct recall@10=" << recall_direct << " p50_us=" << p50_direct
            << " p95_us=" << p95_direct << " p99_us=" << p99_direct
            << " qps=" << direct_qps << "\n";
  std::cout << "adapter recall@10=" << recall_adapter << " p50_us=" << p50_adapter
            << " p95_us=" << p95_adapter << " p99_us=" << p99_adapter
            << " qps=" << adapter_qps << " delta_us=" << delta_us
            << " ratio=" << ratio << "\n";

  EXPECT_LE(std::abs(recall_direct - recall_adapter), 0.001);
  EXPECT_TRUE(ratio <= 0.03 || delta_us <= 20.0)
      << "p50 direct=" << p50_direct << "us adapter=" << p50_adapter
      << "us delta=" << delta_us << "us ratio=" << ratio;
  EXPECT_GE(equivalent, static_cast<uint32_t>(0.9 * kQueries));

  std::error_code ec;
  std::filesystem::remove_all(root, ec);
}

}  // namespace alaya::disk
