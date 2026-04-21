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

//
// build_vamana_index — produce a Vamana .index file in DiskANN's single-file
// binary layout for downstream consumption by any DiskANN-format reader
// (including DiskANN's own `search_memory_index`).
//
// Dispatches between two paths based on estimated single-shard RAM vs
// --build_dram_budget:
//   * single-shard in-memory build (Phase 1), or
//   * k-means partition → per-shard build → union-shuffle-cut merge
//     (Phase 2), streaming the base data where possible to avoid loading
//     the full 51GB BIGANN-100M file into RAM.
//

#include <omp.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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

namespace {

struct Args {
  std::string data_path;
  std::string index_path_prefix;
  uint32_t R = 64;
  uint32_t L = 100;
  float alpha = 1.2f;
  uint32_t num_threads = 0;  // 0 → omp_get_num_procs()
  uint64_t seed = 1234;
  float build_dram_budget_gb = 32.0f;
  bool show_help = false;
};

void print_help(std::ostream &os) {
  os << "build_vamana_index — produce a DiskANN-format Vamana .index file\n"
     << "\n"
     << "Required:\n"
     << "  --data_path <path>             Input .fbin file (float32: u32 num, u32 dim, data)\n"
     << "  --index_path_prefix <path>     Output graph path (DiskANN .index single-file)\n"
     << "\n"
     << "Optional:\n"
     << "  -R, --max_degree <uint32>      Graph degree bound (default 64)\n"
     << "  -L, --lbuild <uint32>          Build-time beam width (default 100)\n"
     << "      --alpha <float>            α-RNG pruning parameter (default 1.2)\n"
     << "  -T, --num_threads <uint32>     OpenMP thread count (default: omp_get_num_procs())\n"
     << "      --seed <uint64>            RNG seed (default 1234)\n"
     << "      --build_dram_budget <GB>   Single-shard budget in GiB (default 32.0)\n"
     << "  -h, --help                     Show this message\n"
     << "\n"
     << "Dispatch:\n"
     << "  * single-shard in-memory build when estimated RAM ≤ --build_dram_budget\n"
     << "  * k-means partition + union-shuffle-cut merge otherwise, writing\n"
     << "    per-shard intermediates next to --index_path_prefix under a\n"
     << "    `<prefix>_shard_work/` directory (left on disk; remove manually\n"
     << "    when no longer needed).\n";
}

[[noreturn]] void die(const std::string &msg) {
  std::cerr << "error: " << msg << "\n";
  std::cerr << "Run with --help for usage.\n";
  std::exit(2);
}

uint32_t parse_u32(const std::string &s, const std::string &flag) {
  try {
    size_t pos = 0;
    unsigned long long v = std::stoull(s, &pos);
    if (pos != s.size() || v > std::numeric_limits<uint32_t>::max()) {
      throw std::invalid_argument("out of range");
    }
    return static_cast<uint32_t>(v);
  } catch (const std::exception &) {
    die("invalid uint32 for " + flag + ": '" + s + "'");
  }
}

uint64_t parse_u64(const std::string &s, const std::string &flag) {
  try {
    size_t pos = 0;
    unsigned long long v = std::stoull(s, &pos);
    if (pos != s.size()) {
      throw std::invalid_argument("trailing garbage");
    }
    return static_cast<uint64_t>(v);
  } catch (const std::exception &) {
    die("invalid uint64 for " + flag + ": '" + s + "'");
  }
}

float parse_f32(const std::string &s, const std::string &flag) {
  try {
    size_t pos = 0;
    float v = std::stof(s, &pos);
    if (pos != s.size()) {
      throw std::invalid_argument("trailing garbage");
    }
    return v;
  } catch (const std::exception &) {
    die("invalid float for " + flag + ": '" + s + "'");
  }
}

Args parse_args(int argc, char **argv) {
  Args a;
  auto need_value = [&](int &i) -> std::string {
    if (i + 1 >= argc) {
      die(std::string("flag '") + argv[i] + "' requires a value");
    }
    ++i;
    return argv[i];
  };

  for (int i = 1; i < argc; ++i) {
    std::string_view flag = argv[i];
    if (flag == "-h" || flag == "--help") {
      a.show_help = true;
    } else if (flag == "--data_path") {
      a.data_path = need_value(i);
    } else if (flag == "--index_path_prefix") {
      a.index_path_prefix = need_value(i);
    } else if (flag == "-R" || flag == "--max_degree") {
      a.R = parse_u32(need_value(i), std::string(flag));
    } else if (flag == "-L" || flag == "--lbuild") {
      a.L = parse_u32(need_value(i), std::string(flag));
    } else if (flag == "--alpha") {
      a.alpha = parse_f32(need_value(i), std::string(flag));
    } else if (flag == "-T" || flag == "--num_threads") {
      a.num_threads = parse_u32(need_value(i), std::string(flag));
    } else if (flag == "--seed") {
      a.seed = parse_u64(need_value(i), std::string(flag));
    } else if (flag == "--build_dram_budget") {
      a.build_dram_budget_gb = parse_f32(need_value(i), std::string(flag));
    } else {
      die(std::string("unknown flag '") + argv[i] + "'");
    }
  }
  return a;
}

// read_fbin_header — read just the 8-byte (num, dim) header of a .fbin
// without loading any vectors. Used by the partition path to decide
// dispatch without materializing the full dataset (critical at 100M).
void read_fbin_header(const std::string &path, uint32_t &num, uint32_t &dim) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    die("cannot open --data_path: " + path);
  }
  in.read(reinterpret_cast<char *>(&num), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
  if (!in.good() || num == 0 || dim == 0) {
    die("corrupt .fbin header: " + path);
  }
}

// DiskANN .fbin loader: header `uint32 num, uint32 dim` then `num × dim`
// little-endian float32 values. Reads all vectors into a flat row-major
// buffer.
void load_fbin(const std::string &path,
               std::vector<float> &data,
               uint32_t &num,
               uint32_t &dim) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    die("cannot open --data_path: " + path);
  }
  in.read(reinterpret_cast<char *>(&num), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
  if (!in.good() || num == 0 || dim == 0) {
    die("corrupt .fbin header: " + path);
  }
  const size_t total = static_cast<size_t>(num) * dim;
  data.resize(total);
  in.read(reinterpret_cast<char *>(data.data()),
          static_cast<std::streamsize>(total * sizeof(float)));
  if (static_cast<size_t>(in.gcount()) != total * sizeof(float)) {
    die("short read on .fbin data: " + path);
  }
}

// gen_random_slice — streaming Bernoulli sampler. Reads the .fbin one
// vector at a time, retains each with probability `rate`, writes the
// retained vectors into `out` (flat row-major). The caller-owned `rng`
// drives the sampling so two successive calls with the same engine
// produce disjoint/independent draws (matches DiskANN's train/test
// double-sampling pattern; see `partition.cpp:536-541`).
void gen_random_slice(const std::string &path,
                      double rate,
                      std::mt19937_64 &rng,
                      std::vector<float> &out,
                      size_t &out_num) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    die("gen_random_slice: cannot open " + path);
  }
  uint32_t num = 0;
  uint32_t dim = 0;
  in.read(reinterpret_cast<char *>(&num), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
  if (!in.good() || num == 0 || dim == 0) {
    die("gen_random_slice: corrupt .fbin header: " + path);
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

// Per DiskANN: single-shard RAM ≈ data_bytes + R * N * sizeof(uint32_t)
// plus ~1.3× transient graph overhead during link. We use the same
// formula + a 1.3 safety factor to decide if we must partition.
double estimate_single_shard_gb(uint32_t num, uint32_t dim, uint32_t R) {
  const double data_bytes =
      static_cast<double>(num) * static_cast<double>(dim) * sizeof(float);
  const double graph_bytes =
      static_cast<double>(num) * static_cast<double>(R) * sizeof(uint32_t) * 1.3;
  return (data_bytes + graph_bytes) / (1024.0 * 1024.0 * 1024.0);
}

// run_single_shard — legacy Phase 1 path. Loads full .fbin, builds
// Vamana, saves. Unchanged logic, wrapped for dispatch clarity.
int run_single_shard(const Args &args, uint32_t /*num_hint*/, uint32_t /*dim_hint*/) {
  std::vector<float> data;
  uint32_t num = 0;
  uint32_t dim = 0;
  alaya::Timer load_timer;
  load_fbin(args.data_path, data, num, dim);
  LOG_INFO("loaded {} vectors x {} dims in {}s", num, dim, load_timer.elapsed_s());

  alaya::vamana::VamanaBuildParams params;
  params.R = args.R;
  params.L = args.L;
  params.alpha = args.alpha;
  params.num_threads = args.num_threads;
  params.seed = args.seed;

  alaya::vamana::VamanaBuilder builder(data.data(),
                                       static_cast<size_t>(num),
                                       dim,
                                       params);
  alaya::Timer build_timer;
  builder.build();
  LOG_INFO("total build time: {}s", build_timer.elapsed_s());

  alaya::Timer save_timer;
  alaya::vamana::save_graph(builder.graph(),
                            args.index_path_prefix,
                            args.R,
                            builder.medoid());
  LOG_INFO("wrote {} in {}s", args.index_path_prefix, save_timer.elapsed_s());
  return 0;
}

// run_partition_merge — Phase 2 path. Samples train/test from .fbin,
// grows num_parts until per-shard RAM fits the budget, streams shard
// assignments to per-shard data files, builds each shard's Vamana graph
// in-memory, then merges (union-shuffle-cut) into a single output file.
int run_partition_merge(const Args &args, uint32_t num, uint32_t dim) {
  constexpr double kSamplingRate = 0.01;

  // Intermediate artifacts live next to the output so they're on the same
  // filesystem (fast rename/IO). User can `rm -rf` the work dir after the
  // final .index lands — we don't auto-delete, to aid post-mortem on
  // alignment failures.
  const std::filesystem::path out_path(args.index_path_prefix);
  const std::filesystem::path work_dir =
      out_path.parent_path() / (out_path.filename().string() + "_shard_work");
  std::filesystem::create_directories(work_dir);
  const std::string shard_prefix = (work_dir / "s").string();
  LOG_INFO("partition path: shard work dir = {}", work_dir.string());

  // 1. Sample train and test from the .fbin.
  std::mt19937_64 sampling_rng(args.seed);
  std::vector<float> train_sample;
  size_t num_train = 0;
  alaya::Timer sample_timer;
  gen_random_slice(args.data_path, kSamplingRate, sampling_rng, train_sample, num_train);
  std::vector<float> test_sample;
  size_t num_test = 0;
  gen_random_slice(args.data_path, kSamplingRate, sampling_rng, test_sample, num_test);
  LOG_INFO("sampled train={} test={} (rate {}) in {}s",
           num_train, num_test, kSamplingRate, sample_timer.elapsed_s());
  if (num_train < 3 || num_test < 3) {
    die("partition path: sampled train/test too small (≥3 required). "
        "Increase sampling_rate or dataset size.");
  }

  // Per-shard build degree. DiskANN's build_disk_index builds each shard at
  // 2*R/3 (disk_utils.cpp:691,714), not the final R. Rationale: the merge is a
  // random-shuffle-cut to R, so building shards with fewer edges means a
  // larger fraction of pruned edges survive the cut (≈76% at 2R/3 vs 50% at
  // full R for k_base=2). This is required for partitioned recall to track
  // single-shard recall within ~1pp.
  const uint32_t shard_R = std::max<uint32_t>(1u, 2u * args.R / 3u);

  // 2. Budget growth loop → frozen num_parts + pivots. Uses the shard-level
  // degree for the RAM estimate so num_parts matches DiskANN at the same
  // budget (disk_utils.cpp:691 passes 2*R/3 to partition_with_ram_budget).
  alaya::vamana::BudgetLoopParams bp;
  bp.graph_degree = shard_R;
  bp.dtype_size = sizeof(float);
  bp.k_base = 2;
  bp.sampling_rate = kSamplingRate;
  bp.ram_budget_gib = static_cast<double>(args.build_dram_budget_gb);
  bp.base_kmeans.seed = args.seed;
  std::vector<float> pivots;
  const size_t num_parts = alaya::vamana::determine_num_parts_with_ram_budget(
      train_sample.data(), num_train,
      test_sample.data(), num_test,
      dim, bp, pivots);
  LOG_INFO("partition path: frozen num_parts={}", num_parts);

  // Free sample memory before streaming assignment (which may allocate a
  // 512MB read buffer).
  train_sample = {};
  test_sample = {};

  // 3. Stream-assign all base points to shards.
  alaya::Timer assign_timer;
  auto assign = alaya::vamana::shard_data_by_centroids(
      args.data_path, pivots.data(), num_parts, bp.k_base, shard_prefix);
  LOG_INFO("shard assignment done in {}s", assign_timer.elapsed_s());

  // 4. Per-shard Vamana build. Each shard fits in RAM by construction
  // (budget loop picked num_parts for exactly that reason). Build one at
  // a time so peak RSS is bounded by `max(shard_size)` rather than
  // `sum(shard_sizes)`.
  std::vector<std::filesystem::path> shard_graphs(num_parts);
  for (size_t s = 0; s < num_parts; ++s) {
    alaya::Timer shard_timer;
    LOG_INFO("building shard {}/{}: {} points", s + 1, num_parts, assign.counts[s]);
    std::vector<float> shard_data;
    uint32_t snum = 0;
    uint32_t sdim = 0;
    load_fbin(assign.data_paths[s], shard_data, snum, sdim);
    if (sdim != dim) {
      die("shard " + std::to_string(s) + " dim mismatch");
    }

    alaya::vamana::VamanaBuildParams vp;
    vp.R = shard_R;  // matches DiskANN disk_utils.cpp:714 low_degree_params
    vp.L = args.L;
    vp.alpha = args.alpha;
    vp.num_threads = args.num_threads;
    vp.seed = args.seed;
    alaya::vamana::VamanaBuilder b(shard_data.data(), snum, sdim, vp);
    b.build();

    const std::filesystem::path graph_path =
        work_dir /
        (std::string("s_subshard-") + std::to_string(s) + "_graph.index");
    alaya::vamana::save_graph(b.graph(), graph_path, shard_R, b.medoid());
    shard_graphs[s] = graph_path;
    LOG_INFO("shard {}/{} built + saved in {}s → {}",
             s + 1, num_parts, shard_timer.elapsed_s(), graph_path.string());
  }

  // 5. Streaming medoid on the global id space.
  alaya::Timer medoid_timer;
  const uint32_t global_medoid =
      alaya::vamana::compute_medoid_streaming(args.data_path);
  LOG_INFO("global medoid = {} (streaming pass {}s)",
           global_medoid, medoid_timer.elapsed_s());

  // 6. Merge.
  std::vector<std::filesystem::path> idmaps;
  idmaps.reserve(num_parts);
  for (const auto &p : assign.idmap_paths) {
    idmaps.emplace_back(p);
  }
  alaya::Timer merge_timer;
  alaya::vamana::merge_shards(shard_graphs, idmaps,
                              args.index_path_prefix,
                              args.R, global_medoid, args.seed);
  LOG_INFO("merge done in {}s", merge_timer.elapsed_s());
  LOG_INFO("partition-merge complete: {} (N={}, num_parts={})",
           args.index_path_prefix, num, num_parts);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  Args args = parse_args(argc, argv);
  if (args.show_help) {
    print_help(std::cout);
    return 0;
  }
  if (args.data_path.empty()) {
    die("missing required flag --data_path");
  }
  if (args.index_path_prefix.empty()) {
    die("missing required flag --index_path_prefix");
  }
  if (args.R == 0) {
    die("--max_degree must be > 0");
  }
  if (args.L < args.R) {
    die("--lbuild must be >= --max_degree");
  }
  if (args.alpha < 1.0f) {
    die("--alpha must be >= 1.0");
  }
  if (args.num_threads == 0) {
    args.num_threads = static_cast<uint32_t>(omp_get_num_procs());
  }

  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

  LOG_INFO("build_vamana_index: data={}, out={}, R={}, L={}, alpha={}, threads={}, seed={}",
           args.data_path,
           args.index_path_prefix,
           args.R,
           args.L,
           args.alpha,
           args.num_threads,
           args.seed);

  uint32_t num = 0;
  uint32_t dim = 0;
  read_fbin_header(args.data_path, num, dim);
  LOG_INFO(".fbin header: N={}, dim={}", num, dim);

  const double estimated_gb = estimate_single_shard_gb(num, dim, args.R);
  LOG_INFO("estimated single-shard RAM: {:.3f} GiB (budget {:.3f} GiB)",
           estimated_gb, args.build_dram_budget_gb);

  if (estimated_gb <= args.build_dram_budget_gb) {
    return run_single_shard(args, num, dim);
  }
  return run_partition_merge(args, num, dim);
}
