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

#include "simd/fht.hpp"

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

struct SizeResults {
  size_t log_n_ = 0;
  size_t n_ = 0;
  BenchResult generic_;
  BenchResult avx2_;
  BenchResult avx512_;
  BenchResult dispatch_;  // auto dispatch (helper_float_N)
  BenchResult fht_float_; // unified API (fht_float)
  bool has_avx2_ = false;
  bool has_avx512_ = false;
};

template <typename Func>
auto run_benchmark(Func func, std::vector<float>& buf, size_t iterations) -> double {
  std::vector<float> backup = buf;
  volatile float sink = 0;

  // Warmup
  for (size_t i = 0; i < kWarmupIterations; ++i) {
    buf = backup;
    func(buf.data());
    sink = buf[0];
  }
  (void)sink;

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    buf = backup;
    func(buf.data());
  }
  auto end = std::chrono::high_resolution_clock::now();

  auto duration_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return static_cast<double>(duration_ns) / static_cast<double>(iterations);
}

auto run_benchmarks_for_size(size_t log_n) -> SizeResults {
  SizeResults results;
  results.log_n_ = log_n;
  results.n_ = 1ULL << log_n;

  std::vector<float> buf(results.n_);
  fill_random(buf, 42);

  // Generic (baseline) - use template based on log_n
  alaya::simd::FHT_Helper_Func generic_func = nullptr;
  switch (log_n) {
    case 6:
      generic_func = alaya::simd::fwht_generic_template<6>;
      break;
    case 7:
      generic_func = alaya::simd::fwht_generic_template<7>;
      break;
    case 8:
      generic_func = alaya::simd::fwht_generic_template<8>;
      break;
    case 9:
      generic_func = alaya::simd::fwht_generic_template<9>;
      break;
    case 10:
      generic_func = alaya::simd::fwht_generic_template<10>;
      break;
    case 11:
      generic_func = alaya::simd::fwht_generic_template<11>;
      break;
    default:
      break;
  }
  if (generic_func != nullptr) {
    results.generic_.ns_per_call_ = run_benchmark(generic_func, buf, kBenchmarkIterations);
  }
  results.generic_.speedup_ = 1.0;
  double baseline_ns = results.generic_.ns_per_call_;

#ifdef ALAYA_ARCH_X86
  const auto& features = alaya::simd::get_cpu_features();

  // Get AVX2 and AVX512 functions based on size
  alaya::simd::FHT_Helper_Func avx2_func = nullptr;
  alaya::simd::FHT_Helper_Func avx512_func = nullptr;
  alaya::simd::FHT_Helper_Func dispatch_func = nullptr;

  switch (log_n) {
    case 6:
      avx2_func = alaya::simd::helper_float_6_avx2;
      avx512_func = alaya::simd::helper_float_6_avx512;
      dispatch_func = alaya::simd::helper_float_6;
      break;
    case 7:
      avx2_func = alaya::simd::helper_float_7_avx2;
      avx512_func = alaya::simd::helper_float_7_avx512;
      dispatch_func = alaya::simd::helper_float_7;
      break;
    case 8:
      avx2_func = alaya::simd::helper_float_8_avx2;
      avx512_func = alaya::simd::helper_float_8_avx512;
      dispatch_func = alaya::simd::helper_float_8;
      break;
    case 9:
      avx2_func = alaya::simd::helper_float_9_avx2;
      avx512_func = alaya::simd::helper_float_9_avx512;
      dispatch_func = alaya::simd::helper_float_9;
      break;
    case 10:
      avx2_func = alaya::simd::helper_float_10_avx2;
      avx512_func = alaya::simd::helper_float_10_avx512;
      dispatch_func = alaya::simd::helper_float_10;
      break;
    case 11:
      avx2_func = alaya::simd::helper_float_11_avx2;
      avx512_func = alaya::simd::helper_float_11_avx512;
      dispatch_func = alaya::simd::helper_float_11;
      break;
    default:
      break;
  }

  // AVX2
  if (features.avx2_ && avx2_func != nullptr) {
    results.has_avx2_ = true;
    results.avx2_.ns_per_call_ = run_benchmark(avx2_func, buf, kBenchmarkIterations);
    results.avx2_.speedup_ = baseline_ns / results.avx2_.ns_per_call_;
  }

  // AVX-512
  if (features.avx512f_ && avx512_func != nullptr) {
    results.has_avx512_ = true;
    results.avx512_.ns_per_call_ = run_benchmark(avx512_func, buf, kBenchmarkIterations);
    results.avx512_.speedup_ = baseline_ns / results.avx512_.ns_per_call_;
  }

  // Dispatch (auto select best)
  if (dispatch_func != nullptr) {
    results.dispatch_.ns_per_call_ = run_benchmark(dispatch_func, buf, kBenchmarkIterations);
    results.dispatch_.speedup_ = baseline_ns / results.dispatch_.ns_per_call_;
  }

  // fht_float unified API
  auto fht_float_wrapper = [log_n](float* data) -> void { alaya::simd::fht_float(data, static_cast<int>(log_n)); };
  results.fht_float_.ns_per_call_ = run_benchmark(fht_float_wrapper, buf, kBenchmarkIterations);
  results.fht_float_.speedup_ = baseline_ns / results.fht_float_.ns_per_call_;
#else
  // Non-x86: only dispatch function available
  alaya::simd::FHT_Helper_Func dispatch_func = nullptr;
  switch (log_n) {
    case 6:
      dispatch_func = alaya::simd::helper_float_6;
      break;
    case 7:
      dispatch_func = alaya::simd::helper_float_7;
      break;
    case 8:
      dispatch_func = alaya::simd::helper_float_8;
      break;
    case 9:
      dispatch_func = alaya::simd::helper_float_9;
      break;
    case 10:
      dispatch_func = alaya::simd::helper_float_10;
      break;
    case 11:
      dispatch_func = alaya::simd::helper_float_11;
      break;
    default:
      break;
  }
  if (dispatch_func != nullptr) {
    results.dispatch_.ns_per_call_ = run_benchmark(dispatch_func, buf, kBenchmarkIterations);
    results.dispatch_.speedup_ = baseline_ns / results.dispatch_.ns_per_call_;
  }
#endif

  return results;
}

void print_comparison_table(const std::vector<SizeResults>& all_results) {
  std::cout << "\n## FHT SIMD Performance Comparison\n\n";
  std::cout << "| Size (2^N) | Generic (baseline) | AVX2 | AVX-512 | AUTO |\n";
  std::cout << "|------------|-------------------|------|---------|------|\n";

  for (const auto& r : all_results) {
    std::cout << "| 2^" << r.log_n_ << " (" << r.n_ << ") | ";

    // Generic (baseline)
    std::cout << std::fixed << std::setprecision(2) << r.generic_.ns_per_call_
              << " ns (1.00x) | ";

    // AVX2
    if (r.has_avx2_) {
      if (r.avx2_.speedup_ > 1.05) {
        std::cout << "**" << r.avx2_.ns_per_call_ << " ns (" << r.avx2_.speedup_
                  << "x)** | ";
      } else {
        std::cout << r.avx2_.ns_per_call_ << " ns (" << r.avx2_.speedup_ << "x) | ";
      }
    } else {
      std::cout << "N/A | ";
    }

    // AVX-512
    if (r.has_avx512_) {
      if (r.avx512_.speedup_ > 1.05) {
        std::cout << "**" << r.avx512_.ns_per_call_ << " ns (" << r.avx512_.speedup_
                  << "x)** | ";
      } else {
        std::cout << r.avx512_.ns_per_call_ << " ns (" << r.avx512_.speedup_ << "x) | ";
      }
    } else {
      std::cout << "N/A | ";
    }

    // Dispatch (auto)
    if (r.dispatch_.ns_per_call_ > 0) {
      if (r.dispatch_.speedup_ > 1.05) {
        std::cout << "**" << r.dispatch_.ns_per_call_ << " ns (" << r.dispatch_.speedup_
                  << "x)** |";
      } else {
        std::cout << r.dispatch_.ns_per_call_ << " ns (" << r.dispatch_.speedup_
                  << "x) |";
      }
    } else {
      std::cout << "N/A |";
    }

    std::cout << '\n';
  }

  // Print summary
  std::cout << "\n## Summary\n\n";
  std::cout << "- **Bold** indicates >5% speedup over Generic baseline\n";
  std::cout << "- **AUTO** = helper_float_N() with auto dispatch\n";
  std::cout << "- SIMD Level: " << alaya::simd::get_simd_level_name() << '\n';
  std::cout << "- Iterations per test: " << kBenchmarkIterations << '\n';
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  std::cout << "# FHT (Fast Hadamard Transform) SIMD Benchmark\n\n";

  // Default: test sizes 2^6 to 2^11 (64 to 2048)
  std::vector<size_t> log_sizes = {6, 7, 8, 9, 10, 11};

  // Allow custom sizes from command line
  if (argc > 1) {
    log_sizes.clear();
    for (int i = 1; i < argc; ++i) {
      log_sizes.push_back(std::stoull(argv[i]));
    }
  }

  std::cout << "Running benchmarks for sizes: ";
  for (size_t i = 0; i < log_sizes.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << "2^" << log_sizes[i] << " (" << (1ULL << log_sizes[i]) << ")";
  }
  std::cout << "\n\n";

  std::vector<SizeResults> all_results;
  for (size_t log_n : log_sizes) {
    std::cout << "Benchmarking size=2^" << log_n << "..." << std::flush;
    all_results.push_back(run_benchmarks_for_size(log_n));
    std::cout << " done\n";
  }

  print_comparison_table(all_results);

  return 0;
}
