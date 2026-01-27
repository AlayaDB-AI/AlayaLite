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

/**
 * @brief Benchmark for L2 SQ4 SIMD distance functions
 *
 * SQ4 stores 2 values per byte (4 bits each):
 *   - Low nibble (bits 0-3) = even index
 *   - High nibble (bits 4-7) = odd index
 *
 * Usage: ./l2_sqr_sq4_benchmark [dim1 dim2 ...]
 * If no dimensions are provided, defaults to common ANN dataset dimensions.
 */
namespace {

constexpr size_t kWarmupIterations = 1000;
constexpr size_t kBenchmarkIterations = 100000;

// Pack 4-bit values into bytes: low nibble = even index, high nibble = odd index
void pack_sq4(std::vector<uint8_t>& packed, size_t dim, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 15);

  size_t num_bytes = (dim + 1) / 2;
  packed.resize(num_bytes);

  for (size_t i = 0; i < dim; i += 2) {
    auto lo = static_cast<uint8_t>(dist(rng));
    uint8_t hi = (i + 1 < dim) ? static_cast<uint8_t>(dist(rng)) : 0;
    packed[i / 2] = lo | (hi << 4);
  }
}

void fill_min_max(std::vector<float>& min_vals,
                  std::vector<float>& max_vals,
                  size_t dim,
                  unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-10.0F, 10.0F);
  min_vals.resize(dim);
  max_vals.resize(dim);
  for (size_t i = 0; i < dim; ++i) {
    float a = dist(rng);
    float b = dist(rng);
    min_vals[i] = std::min(a, b);
    max_vals[i] = std::max(a, b) + 0.1F;
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
  BenchResult best_;
  bool has_avx2_ = false;
  bool has_avx512_ = false;
};

template <typename Func>
auto run_benchmark(Func func,
                   const uint8_t* x,
                   const uint8_t* y,
                   const float* min,
                   const float* max,
                   size_t dim,
                   size_t iterations) -> double {
  volatile float sink = 0;
  for (size_t i = 0; i < kWarmupIterations; ++i) {
    sink = func(x, y, dim, min, max);
  }
  (void)sink;

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    sink = func(x, y, dim, min, max);
  }
  auto end = std::chrono::high_resolution_clock::now();

  auto duration_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return static_cast<double>(duration_ns) / static_cast<double>(iterations);
}

auto run_benchmarks_for_dim(size_t dim) -> DimResults {
  DimResults results;
  results.dim_ = dim;

  std::vector<uint8_t> x;
  std::vector<uint8_t> y;
  std::vector<float> min_vals;
  std::vector<float> max_vals;

  pack_sq4(x, dim, 42);
  pack_sq4(y, dim, 123);
  fill_min_max(min_vals, max_vals, dim, 456);

  // Generic (baseline)
  results.generic_.ns_per_call_ =
      run_benchmark(alaya::simd::l2_sqr_sq4_generic, x.data(), y.data(),
                    min_vals.data(), max_vals.data(), dim, kBenchmarkIterations);
  results.generic_.speedup_ = 1.0;
  double baseline_ns = results.generic_.ns_per_call_;

#ifdef ALAYA_ARCH_X86
  const auto& features = alaya::simd::get_cpu_features();

  // AVX2
  if (features.avx2_ && features.fma_) {
    results.has_avx2_ = true;
    results.avx2_.ns_per_call_ =
        run_benchmark(alaya::simd::l2_sqr_sq4_avx2, x.data(), y.data(),
                      min_vals.data(), max_vals.data(), dim, kBenchmarkIterations);
    results.avx2_.speedup_ = baseline_ns / results.avx2_.ns_per_call_;
  }

  // AVX-512
  if (features.avx512f_) {
    results.has_avx512_ = true;
    results.avx512_.ns_per_call_ =
        run_benchmark(alaya::simd::l2_sqr_sq4_avx512, x.data(), y.data(),
                      min_vals.data(), max_vals.data(), dim, kBenchmarkIterations);
    results.avx512_.speedup_ = baseline_ns / results.avx512_.ns_per_call_;
  }
#endif

  // Best (get_l2_sqr_sq4_func with auto dispatch)
  results.best_.ns_per_call_ =
      run_benchmark(alaya::simd::get_l2_sqr_sq4_func(), x.data(), y.data(),
                    min_vals.data(), max_vals.data(), dim, kBenchmarkIterations);
  results.best_.speedup_ = baseline_ns / results.best_.ns_per_call_;

  return results;
}

void print_comparison_table(const std::vector<DimResults>& all_results) {
  std::cout << "\n## SQ4 L2 SIMD Distance Performance Comparison\n\n";
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

    // Best (get_l2_sqr_sq4_func with dispatch)
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
  std::cout << "- **AUTO** = get_l2_sqr_sq4_func() with auto dispatch\n";
  std::cout << "- SQ4 packs 2 values per byte (4 bits each)\n";
  std::cout << "- SIMD Level: " << alaya::simd::get_simd_level_name() << '\n';
  std::cout << "- Iterations per test: " << kBenchmarkIterations << '\n';
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  std::cout << "# SQ4 L2 SIMD Distance Benchmark\n\n";

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
