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

#pragma once

//
// build_dispatch.hpp — Vamana build dispatch library. Shared between the
// tools/build_vamana_index CLI and the alayalite.vamana Python binding so
// both entry points share one execution path, one set of log messages, and
// one set of defaults. See proposal integrate-vamana-into-laser-pipeline
// (decision D2/D4) for the rationale.
//

#include <omp.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "index/graph/vamana/budget_estimator.hpp"
#include "index/graph/vamana/kmeans_partition.hpp"
#include "index/graph/vamana/shard_assigner.hpp"
#include "index/graph/vamana/shard_merger.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_writer.hpp"
#include "utils/log.hpp"
#include "utils/timer.hpp"

namespace alaya::vamana {

// BuildVamanaParams — all inputs to a Vamana build invocation.
//
// Path fields are `std::string_view` so the struct is a literal type and
// `kDefaultVamanaBuildParams` below can be declared `constexpr`. Callers
// MUST keep the underlying string storage alive for the duration of the
// build_vamana() call (typical lifetime: parse_args or pybind body frame).
//
// Numeric fields carry the canonical defaults in their in-class
// initializers. The struct-as-literal constant `kDefaultVamanaBuildParams`
// is the single source of truth — every other call site MUST initialize
// from this constant or from `BuildVamanaParams{}`, not from a duplicated
// literal.
struct BuildVamanaParams {
  std::string_view data_path{};
  std::string_view output_path{};
  uint32_t R = 64;
  uint32_t L = 200;
  float alpha = 1.2F;
  uint32_t num_threads = 0;  // 0 → omp_get_num_procs() at call time
  uint64_t seed = 1234;
  float build_dram_budget_gb = 32.0F;
  // Partition kmeans sampling rate. Negative is the sentinel meaning
  // "auto" — the partition path resolves it to
  // `min(1.0, MAX_PQ_TRAINING_SET_SIZE_F / N)` (mirrors DiskANN's
  // `disk_utils.cpp:1291`). Discovered scope on
  // `align-diskann-sharded-with-upstream`: the historic literal 0.01
  // was numerically unstable on small datasets (synth_100k_512d at
  // budget=0.05 ran the growth loop into the `max_num_parts=1024`
  // safety cap). The sentinel preserves byte-equality with DiskANN
  // builds at matched seeds while keeping callers who explicitly pass
  // a sampling rate in (0, 1] on the historic 0.01.
  float sampling_rate = -1.0F;
};

// Single source of truth for Vamana build defaults. Both the CLI's argv
// parser and the Python binding's `py::arg` defaults MUST reference the
// fields of this constant; no duplicate literals.
inline constexpr BuildVamanaParams kDefaultVamanaBuildParams{};

namespace detail {

// Frozen defaults table mirrored from the BuildVamanaParams in-class
// initializers. The static_asserts below fail to compile if any literal
// in BuildVamanaParams drifts away from this table — the lock-step pair
// is why the spec requires a "separately-declared constexpr table".
//
// Limitation: a new field added to BuildVamanaParams is not detected here
// unless it's also added to kFrozenDefaults; the CLI / binding / test
// fixtures are expected to update in lockstep.
struct VamanaNumericDefaults {
  uint32_t R;
  uint32_t L;
  float alpha;
  uint32_t num_threads;
  uint64_t seed;
  float build_dram_budget_gb;
  float sampling_rate;
};
inline constexpr VamanaNumericDefaults kFrozenDefaults{
    .R = 64,
    .L = 200,
    .alpha = 1.2F,
    .num_threads = 0,
    .seed = 1234,
    .build_dram_budget_gb = 32.0F,
    .sampling_rate = -1.0F,
};
static_assert(kDefaultVamanaBuildParams.R == kFrozenDefaults.R,
              "Vamana default R drifted — update CLI / binding / fixtures together");
static_assert(kDefaultVamanaBuildParams.L == kFrozenDefaults.L,
              "Vamana default L drifted — update CLI / binding / fixtures together");
static_assert(kDefaultVamanaBuildParams.alpha == kFrozenDefaults.alpha,
              "Vamana default alpha drifted — update CLI / binding / fixtures together");
static_assert(kDefaultVamanaBuildParams.num_threads == kFrozenDefaults.num_threads,
              "Vamana default num_threads drifted — update CLI / binding / fixtures together");
static_assert(kDefaultVamanaBuildParams.seed == kFrozenDefaults.seed,
              "Vamana default seed drifted — update CLI / binding / fixtures together");
static_assert(kDefaultVamanaBuildParams.build_dram_budget_gb ==
                  kFrozenDefaults.build_dram_budget_gb,
              "Vamana default dram_budget drifted — update CLI / binding / fixtures together");
static_assert(kDefaultVamanaBuildParams.sampling_rate == kFrozenDefaults.sampling_rate,
              "Vamana default sampling_rate drifted — update CLI / binding / fixtures together");

inline bool is_finite_float(float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(value));
  return (bits & 0x7F800000U) != 0x7F800000U;
}

// read_fbin_header — read the 8-byte (num, dim) header of a .fbin without
// loading any vectors. Used by the partition path to decide dispatch
// without materializing the full dataset (critical at 100M scale).
inline void read_fbin_header(const std::string &path, uint32_t &num, uint32_t &dim) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("cannot open --data_path: " + path);
  }
  in.read(reinterpret_cast<char *>(&num), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
  if (!in.good() || num == 0 || dim == 0) {
    throw std::invalid_argument("corrupt .fbin header: " + path);
  }
}

// DiskANN .fbin loader: header `uint32 num, uint32 dim` then `num × dim`
// little-endian float32 values. Reads all vectors into a flat row-major
// buffer.
inline void load_fbin(const std::string &path,
                      std::vector<float> &data,
                      uint32_t &num,
                      uint32_t &dim) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("cannot open --data_path: " + path);
  }
  in.read(reinterpret_cast<char *>(&num), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
  if (!in.good() || num == 0 || dim == 0) {
    throw std::invalid_argument("corrupt .fbin header: " + path);
  }
  const size_t total = static_cast<size_t>(num) * dim;
  data.resize(total);
  in.read(reinterpret_cast<char *>(data.data()),
          static_cast<std::streamsize>(total * sizeof(float)));
  if (static_cast<size_t>(in.gcount()) != total * sizeof(float)) {
    throw std::runtime_error("short read on .fbin data: " + path);
  }
}

// gen_random_slice — streaming Bernoulli sampler. Reads the .fbin one
// vector at a time, retains each with probability `rate`, writes the
// retained vectors into `out` (flat row-major). The caller-owned `rng`
// drives the sampling so two successive calls with the same engine
// produce disjoint/independent draws (matches DiskANN's train/test
// double-sampling pattern; see `partition.cpp:536-541`).
inline void gen_random_slice(const std::string &path,
                             double rate,
                             std::mt19937_64 &rng,
                             std::vector<float> &out,
                             size_t &out_num) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("gen_random_slice: cannot open " + path);
  }
  uint32_t num = 0;
  uint32_t dim = 0;
  in.read(reinterpret_cast<char *>(&num), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
  if (!in.good() || num == 0 || dim == 0) {
    throw std::invalid_argument("gen_random_slice: corrupt .fbin header: " + path);
  }
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::vector<float> buf(dim);
  out.clear();
  out_num = 0;
  for (uint32_t i = 0; i < num; ++i) {
    in.read(reinterpret_cast<char *>(buf.data()),
            static_cast<std::streamsize>(dim) * sizeof(float));
    if (u01(rng) < rate) {
      out.insert(out.end(), buf.begin(), buf.end());
      ++out_num;
    }
  }
}

// estimate_single_shard_gb — thin wrapper around the shared RAM formula in
// budget_estimator.hpp. Historically this used a simplified formula (no
// ROUND_UP(dim,8), no per-node locks, no 1.1× overhead) that could disagree
// with the inner `partition_with_ram_budget` loop near the budget boundary
// and produce a different dispatch decision than DiskANN would make. The
// wrapper guarantees top-level dispatch and the inner budget loop use the
// same bytes-per-shard estimate.
inline double estimate_single_shard_gb(uint32_t num, uint32_t dim, uint32_t R) {
  return alaya::vamana::estimate_ram_usage_gib(static_cast<size_t>(num), dim, sizeof(float), R);
}

// run_single_shard — Phase 1 path. Loads full .fbin, builds Vamana, saves.
inline void run_single_shard(const BuildVamanaParams &args) {
  const std::string data_path(args.data_path);
  const std::string out_path(args.output_path);

  std::vector<float> data;
  uint32_t num = 0;
  uint32_t dim = 0;
  alaya::Timer load_timer;
  load_fbin(data_path, data, num, dim);
  LOG_INFO("loaded {} vectors x {} dims in {}s", num, dim, load_timer.elapsed_s());

  alaya::vamana::VamanaBuildParams params;
  params.R = args.R;
  params.L = args.L;
  params.alpha = args.alpha;
  params.num_threads = args.num_threads;
  params.seed = args.seed;

  alaya::vamana::VamanaBuilder builder(data.data(), static_cast<size_t>(num), dim, params);
  alaya::Timer build_timer;
  builder.build();
  LOG_INFO("total build time: {}s", build_timer.elapsed_s());

  alaya::Timer save_timer;
  alaya::vamana::save_graph(builder.graph(), out_path, args.R, builder.medoid());
  LOG_INFO("wrote {} in {}s", out_path, save_timer.elapsed_s());
}

// run_partition_merge — Phase 2 path. Samples train/test from .fbin,
// grows num_parts until per-shard RAM fits the budget, streams shard
// assignments to per-shard data files, builds each shard's Vamana graph
// in-memory, then merges (union-shuffle-cut) into a single output file.
inline void run_partition_merge(const BuildVamanaParams &args, uint32_t num, uint32_t dim) {
  // Resolve sampling rate. Negative `args.sampling_rate` means "auto":
  // mirror DiskANN's `disk_utils.cpp:1291` (`min(1.0, 256000/N)`) so the
  // patched-DiskANN Tier A reference and AlayaLite agree on sample size
  // at matched seeds. A positive override forces the historic 0.01
  // path (or any user-supplied rate in (0, 1]) for callers who depend on it.
  constexpr double kMaxPqTrainingSetSize = 256000.0;
  const double kSamplingRate =
      (args.sampling_rate > 0.0F) ? static_cast<double>(args.sampling_rate)
                                  : std::min(1.0, kMaxPqTrainingSetSize / static_cast<double>(num));
  const std::string data_path(args.data_path);
  const std::string out_path_str(args.output_path);

  const std::filesystem::path out_path(out_path_str);
  const std::filesystem::path work_dir =
      out_path.parent_path() / (out_path.filename().string() + "_shard_work");
  std::filesystem::create_directories(work_dir);
  const std::string shard_prefix = (work_dir / "s").string();
  LOG_INFO("partition path: shard work dir = {}", work_dir.string());

  std::mt19937_64 sampling_rng(args.seed);
  std::vector<float> train_sample;
  size_t num_train = 0;
  alaya::Timer sample_timer;
  gen_random_slice(data_path, kSamplingRate, sampling_rng, train_sample, num_train);
  std::vector<float> test_sample;
  size_t num_test = 0;
  gen_random_slice(data_path, kSamplingRate, sampling_rng, test_sample, num_test);
  LOG_INFO("sampled train={} test={} (rate {}) in {}s",
           num_train,
           num_test,
           kSamplingRate,
           sample_timer.elapsed_s());
  if (num_train < 3 || num_test < 3) {
    throw std::runtime_error(
        "partition path: sampled train/test too small (>=3 required). "
        "Increase sampling_rate or dataset size.");
  }

  // DiskANN's build_disk_index builds each shard at 2*R/3 (disk_utils.cpp:691,714),
  // not the final R. The merge is a random-shuffle-cut to R, so building shards
  // with fewer edges means a larger fraction of pruned edges survive the cut.
  const uint32_t shard_R = std::max<uint32_t>(1U, 2U * args.R / 3U);

  // Budget growth loop uses shard-level degree for RAM estimate so num_parts
  // matches DiskANN at the same budget (disk_utils.cpp:691 passes 2*R/3 to
  // partition_with_ram_budget).
  alaya::vamana::BudgetLoopParams bp;
  bp.graph_degree = shard_R;
  bp.dtype_size = sizeof(float);
  bp.k_base = 2;
  bp.sampling_rate = kSamplingRate;
  bp.ram_budget_gib = static_cast<double>(args.build_dram_budget_gb);
  bp.base_kmeans.seed = args.seed;
  std::vector<float> pivots;
  const size_t num_parts = alaya::vamana::determine_num_parts_with_ram_budget(train_sample.data(),
                                                                              num_train,
                                                                              test_sample.data(),
                                                                              num_test,
                                                                              dim,
                                                                              bp,
                                                                              pivots);
  LOG_INFO("partition path: frozen num_parts={}", num_parts);

  // Write pivots alongside the merged output. DiskANN's
  // `build_merged_vamana_index` (disk_utils.cpp:694) renames the pivot file
  // from `<tempFiles>_centroids.bin` into `<centroids_file>`; we write
  // directly to `<output>_centroids.bin` — same byte layout as
  // `diskann::save_bin<float>` (utils.h:719): int32 num_parts,
  // int32 dim, float32[num_parts * dim]. Downstream tools
  // (e.g. DiskANN's `search_memory_index` sharded init) rely on this file.
  const std::filesystem::path centroids_path =
      out_path.parent_path() / (out_path.filename().string() + "_centroids.bin");
  {
    std::ofstream cf(centroids_path, std::ios::binary | std::ios::trunc);
    if (!cf.is_open()) {
      throw std::runtime_error("cannot open centroids file: " + centroids_path.string());
    }
    const int32_t np = static_cast<int32_t>(num_parts);
    const int32_t dd = static_cast<int32_t>(dim);
    cf.write(reinterpret_cast<const char *>(&np), sizeof(int32_t));
    cf.write(reinterpret_cast<const char *>(&dd), sizeof(int32_t));
    cf.write(reinterpret_cast<const char *>(pivots.data()),
             static_cast<std::streamsize>(num_parts) * static_cast<std::streamsize>(dim) *
                 sizeof(float));
    if (!cf.good()) {
      throw std::runtime_error("write error on centroids file: " + centroids_path.string());
    }
  }
  LOG_INFO("wrote centroids: {} ({} × {})", centroids_path.string(), num_parts, dim);

  // Free sample memory before streaming assignment (which may allocate a
  // 512MB read buffer).
  train_sample = {};
  test_sample = {};

  alaya::Timer assign_timer;
  auto assign = alaya::vamana::shard_data_by_centroids(data_path,
                                                       pivots.data(),
                                                       num_parts,
                                                       bp.k_base,
                                                       shard_prefix);
  LOG_INFO("shard assignment done in {}s", assign_timer.elapsed_s());

  std::vector<std::filesystem::path> shard_graphs(num_parts);
  for (size_t s = 0; s < num_parts; ++s) {
    alaya::Timer shard_timer;
    LOG_INFO("building shard {}/{}: {} points", s + 1, num_parts, assign.counts[s]);
    std::vector<float> shard_data;
    uint32_t snum = 0;
    uint32_t sdim = 0;
    load_fbin(assign.data_paths[s], shard_data, snum, sdim);
    if (sdim != dim) {
      throw std::runtime_error("shard " + std::to_string(s) + " dim mismatch");
    }

    alaya::vamana::VamanaBuildParams vp;
    vp.R = shard_R;  // matches DiskANN disk_utils.cpp:714 low_degree_params
    vp.L = args.L;
    vp.alpha = args.alpha;
    vp.num_threads = args.num_threads;
    vp.seed = args.seed;
    alaya::vamana::VamanaBuilder b(shard_data.data(), snum, sdim, vp);
    b.build();

    // File naming matches DiskANN's `build_merged_vamana_index`
    // (`disk_utils.cpp:712`): `<prefix>_subshard-<i>_mem.index`. Using the
    // same suffix lets DiskANN's `search_memory_index` and AlayaLite's
    // `diff_vamana_index.py` consume these per-shard graphs without a
    // naming shim.
    const std::filesystem::path graph_path =
        work_dir / (std::string("s_subshard-") + std::to_string(s) + "_mem.index");
    alaya::vamana::save_graph(b.graph(), graph_path, shard_R, b.medoid());
    shard_graphs[s] = graph_path;
    LOG_INFO("shard {}/{} built + saved in {}s -> {}",
             s + 1,
             num_parts,
             shard_timer.elapsed_s(),
             graph_path.string());
  }

  alaya::Timer medoid_timer;
  const uint32_t global_medoid = alaya::vamana::compute_medoid_streaming(data_path);
  LOG_INFO("global medoid = {} (streaming pass {}s)", global_medoid, medoid_timer.elapsed_s());

  std::vector<std::filesystem::path> idmaps;
  idmaps.reserve(num_parts);
  for (const auto &p : assign.idmap_paths) {
    idmaps.emplace_back(p);
  }
  alaya::Timer merge_timer;
  alaya::vamana::merge_shards(shard_graphs, idmaps, out_path_str, args.R, global_medoid, args.seed);
  LOG_INFO("merge done in {}s", merge_timer.elapsed_s());
  LOG_INFO("partition-merge complete: {} (N={}, num_parts={})", out_path_str, num, num_parts);
}

}  // namespace detail

// build_vamana — top-level entry point. Validates parameters, reads the
// .fbin header, and dispatches to either the single-shard path or the
// partition+merge path based on estimated RAM vs the budget.
//
// Throws:
//   std::invalid_argument — parameter validation failure, malformed .fbin header
//   std::runtime_error    — filesystem errors, short reads, other build failures
//
// Callers MUST keep storage backing `params.data_path` and
// `params.output_path` alive for the duration of the call.
inline void build_vamana(const BuildVamanaParams &params_in) {
  BuildVamanaParams params = params_in;

  if (params.data_path.empty()) {
    throw std::invalid_argument("missing required field data_path");
  }
  if (params.output_path.empty()) {
    throw std::invalid_argument("missing required field output_path");
  }
  if (params.R == 0) {
    throw std::invalid_argument("R (max_degree) must be > 0");
  }
  if (params.L < params.R) {
    throw std::invalid_argument("L (lbuild) must be >= R (max_degree)");
  }
  if (!detail::is_finite_float(params.alpha) || params.alpha < 1.0F) {
    throw std::invalid_argument("alpha must be finite and >= 1.0");
  }
  if (!(params.sampling_rate < 0.0F ||
        (params.sampling_rate > 0.0F && params.sampling_rate <= 1.0F))) {
    throw std::invalid_argument("sampling_rate must be negative for auto or in (0, 1]");
  }
  if (params.num_threads == 0) {
    params.num_threads = static_cast<uint32_t>(omp_get_num_procs());
  }

  const std::string data_path(params.data_path);
  const std::string out_path(params.output_path);

  LOG_INFO("build_vamana_index: data={}, out={}, R={}, L={}, alpha={}, threads={}, seed={}",
           data_path,
           out_path,
           params.R,
           params.L,
           params.alpha,
           params.num_threads,
           params.seed);

  uint32_t num = 0;
  uint32_t dim = 0;
  detail::read_fbin_header(data_path, num, dim);
  LOG_INFO(".fbin header: N={}, dim={}", num, dim);

  const double estimated_gb = detail::estimate_single_shard_gb(num, dim, params.R);
  LOG_INFO("estimated single-shard RAM: {:.3f} GiB (budget {:.3f} GiB)",
           estimated_gb,
           params.build_dram_budget_gb);

  if (estimated_gb <= params.build_dram_budget_gb) {
    detail::run_single_shard(params);
  } else {
    detail::run_partition_merge(params, num, dim);
  }
}

}  // namespace alaya::vamana
