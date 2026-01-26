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

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "simd/distance_l2.hpp"

namespace {

constexpr size_t kWarmupIterations = 1000;
constexpr size_t kBenchmarkIterations = 100000;

void fill_random(std::vector<float>& v, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  for (auto& x : v) {
    x = dist(rng);
  }
}

struct BenchResult {
  double ns_per_call_ = 0.0;
  double speedup_ = 0.0;
};

struct DimResults {
  size_t dim_ = 0;
  BenchResult generic_;
  BenchResult avx2_;
  BenchResult avx512_;
  BenchResult best_;  // get_l2_sqr_func() with auto dispatch
  bool has_avx2_ = false;
  bool has_avx512_ = false;
};

template <typename Func>
auto run_benchmark(Func func, const float* x, const float* y, size_t dim,
                   size_t iterations) -> double {
  volatile float sink = 0;
  for (size_t i = 0; i < kWarmupIterations; ++i) {
    sink = func(x, y, dim);
  }
  (void)sink;

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    sink = func(x, y, dim);
  }
  auto end = std::chrono::high_resolution_clock::now();

  auto duration_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return static_cast<double>(duration_ns) / static_cast<double>(iterations);
}

auto run_benchmarks_for_dim(size_t dim) -> DimResults {
  DimResults results;
  results.dim_ = dim;

  std::vector<float> x(dim);
  std::vector<float> y(dim);
  fill_random(x, 42);
  fill_random(y, 123);

  // Generic (baseline)
  results.generic_.ns_per_call_ =
      run_benchmark(alaya::simd::l2_sqr_generic, x.data(), y.data(), dim,
                    kBenchmarkIterations);
  results.generic_.speedup_ = 1.0;
  double baseline_ns = results.generic_.ns_per_call_;

#ifdef ALAYA_X86
  const auto& features = alaya::simd::get_cpu_features();

  // AVX2
  if (features.avx2_ && features.fma_) {
    results.has_avx2_ = true;
    results.avx2_.ns_per_call_ = run_benchmark(
        alaya::simd::l2_sqr_avx2, x.data(), y.data(), dim, kBenchmarkIterations);
    results.avx2_.speedup_ = baseline_ns / results.avx2_.ns_per_call_;
  }

  // AVX-512
  if (features.avx512f_) {
    results.has_avx512_ = true;
    results.avx512_.ns_per_call_ = run_benchmark(
        alaya::simd::l2_sqr_avx512, x.data(), y.data(), dim, kBenchmarkIterations);
    results.avx512_.speedup_ = baseline_ns / results.avx512_.ns_per_call_;
  }
#endif

  // Best (get_l2_sqr_func with auto dispatch including fixed dimensions)
  results.best_.ns_per_call_ = run_benchmark(alaya::simd::get_l2_sqr_func(),
                                             x.data(), y.data(), dim,
                                             kBenchmarkIterations);
  results.best_.speedup_ = baseline_ns / results.best_.ns_per_call_;

  return results;
}

void print_comparison_table(const std::vector<DimResults>& all_results) {
  std::cout << "\n## L2 SIMD Distance Performance Comparison\n\n";
  std::cout << "| Dimension | Generic (baseline) | AVX2 | AVX-512 | AUTO |\n";
  std::cout << "|-----------|-------------------|------|---------|------|\n";

  for (const auto& r : all_results) {
    std::cout << "| " << r.dim_ << " | ";

    // Generic (baseline)
    std::cout << std::fixed << std::setprecision(2) << r.generic_.ns_per_call_
              << " ns (1.00x) | ";

    // AVX2
    if (r.has_avx2_) {
      if (r.avx2_.speedup_ > 1.05) {
        std::cout << "**" << r.avx2_.ns_per_call_ << " ns (" << r.avx2_.speedup_
                  << "x)** | ";
      } else {
        std::cout << r.avx2_.ns_per_call_ << " ns (" << r.avx2_.speedup_
                  << "x) | ";
      }
    } else {
      std::cout << "N/A | ";
    }

    // AVX-512
    if (r.has_avx512_) {
      if (r.avx512_.speedup_ > 1.05) {
        std::cout << "**" << r.avx512_.ns_per_call_ << " ns ("
                  << r.avx512_.speedup_ << "x)** | ";
      } else {
        std::cout << r.avx512_.ns_per_call_ << " ns (" << r.avx512_.speedup_
                  << "x) | ";
      }
    } else {
      std::cout << "N/A | ";
    }

    // Best (get_l2_sqr_func with dispatch)
    if (r.best_.speedup_ > 1.05) {
      std::cout << "**" << r.best_.ns_per_call_ << " ns (" << r.best_.speedup_
                << "x)** |";
    } else {
      std::cout << r.best_.ns_per_call_ << " ns (" << r.best_.speedup_ << "x) |";
    }

    std::cout << '\n';
  }

  // Print summary
  std::cout << "\n## Summary\n\n";
  std::cout << "- **Bold** indicates >5% speedup over Generic baseline\n";
  std::cout << "- **Best** = get_l2_sqr_func() with auto dispatch (uses Fixed for known dims)\n";
  std::cout << "- SIMD Level: " << alaya::simd::get_simd_level_name() << '\n';
  std::cout << "- Iterations per test: " << kBenchmarkIterations << '\n';
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  std::cout << "# L2 SIMD Distance Benchmark\n\n";

  // Default: ANN mainstream dataset dimensions
  std::vector<size_t> dims = {96, 128, 256, 384, 512, 768, 960, 1024, 1536};

  // Allow custom dimensions from command line
  if (argc > 1) {
    dims.clear();
    for (int i = 1; i < argc; ++i) {
      dims.push_back(std::stoull(argv[i]));
    }
  }

  std::cout << "Running benchmarks for dimensions: ";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << dims[i];
  }
  std::cout << "\n\n";

  std::vector<DimResults> all_results;
  for (size_t dim : dims) {
    std::cout << "Benchmarking dim=" << dim << "..." << std::flush;
    all_results.push_back(run_benchmarks_for_dim(dim));
    std::cout << " done\n";
  }

  print_comparison_table(all_results);

  return 0;
}
