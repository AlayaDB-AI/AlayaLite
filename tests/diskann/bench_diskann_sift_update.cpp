// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only
//
// PQ update benchmark for DiskANNIndex on SIFT1M/GIST1M-style fbin datasets.
// The update trace format matches Yi's update runner: each round file contains
// uint32 update_size, followed by update_size delete ids and update_size insert ids.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "coro/sync_wait.hpp"
#include "coro/thread_pool.hpp"
#include "coro/when_all.hpp"
#include "index/graph/diskann/diskann_index.hpp"

namespace {
using alaya::diskann::DiskANNBuildParams;
using alaya::diskann::DiskANNIndex;
using alaya::diskann::DiskANNLoadParams;
using alaya::diskann::DiskANNSearchParams;

constexpr uint32_t kMissingSlot = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kDefaultBenchmarkGraphR = 64;
constexpr uint32_t kDefaultBenchmarkBuildL = 100;
constexpr uint32_t kDefaultBenchmarkUpdateSearchL = 0;  // 0 => R + 32 (Yi's build_k rule)
constexpr uint64_t kDefaultBenchmarkPageCachePages = 262144;
constexpr uint32_t kBenchmarkTopK = 10;
constexpr double kDefaultBenchmarkCacheRatio = 1.0;

enum class MixedMode { Background, SharedQueue };

struct FloatMatrix {
  std::vector<float> data;
  uint32_t n = 0;
  uint32_t dim = 0;
};

struct IntMatrix {
  std::vector<uint32_t> data;
  uint32_t n = 0;
  uint32_t dim = 0;
  const uint32_t *row(uint32_t i) const { return data.data() + static_cast<size_t>(i) * dim; }
};

struct TraceManifest {
  std::string prefix = "round_";
  uint32_t initial_count = 0;
  uint32_t total_count = 0;
  uint32_t rounds = 0;
  uint32_t update_size = 0;
};

struct TraceRound {
  std::vector<uint32_t> deletes;
  std::vector<uint32_t> inserts;
};

struct DatasetFiles {
  std::string name;
  std::filesystem::path base;
  std::filesystem::path queries;
  std::filesystem::path groundtruth;
};

struct Options {
  std::string data_dir = "./sift1m";
  std::string trace_dir = "/tmp/diskann_sift1m_update_trace_20260701";
  std::string index_dir = "/tmp/diskann_sift1m_alaya_update_pq";
  std::string out_csv = "/tmp/diskann_sift1m_alaya_update_pq.csv";
  bool rebuild = false;
  bool deterministic = false;
  bool flush_rounds = true;  ///< flush after each round, outside the timed window (Yi's
                             ///< write-back is likewise excluded from its update QPS)
  bool build_only = false;
  bool eval_only = false;
  bool single_updates = false;
  bool mixed = false;
  bool update_rerank = true;          ///< Yi trace-bench parity (_rerank_flag=true)
  bool update_insert_prune = false;   ///< Yi never alpha-prunes at insert
  bool update_repair = false;         ///< delete-time in-neighbor repair
  bool track_indegree = false;        ///< track live-slot in-degree percentiles
  bool mixed_round0_baseline = true;  ///< Yi UpdateRunner: round 0 runs no updates
  MixedMode mixed_mode = MixedMode::Background;
  uint32_t max_rounds = 0;
  uint32_t nq = 1000;
  uint32_t build_r = kDefaultBenchmarkGraphR;
  uint32_t build_l = kDefaultBenchmarkBuildL;
  uint32_t record_capacity = 0;  ///< on-disk neighbor slots; 0 => R. Yi keeps
                                 ///< MAX_NEIGHBOURS=96 slots over a ~64-degree
                                 ///< graph, so reconnects rarely overflow.
  uint32_t search_l = 100;
  uint32_t rerank_count = 0;
  uint32_t update_l = kDefaultBenchmarkUpdateSearchL;
  uint32_t repair_search_l = 0;  ///< 0 => deleted-node out-neighbors only
  uint32_t garden_budget = 0;    ///< per-round low-in-degree refresh budget
  uint32_t garden_l = 0;         ///< 0 => update_l
  uint32_t beam = 4;
  uint32_t search_threads = 1;
  uint32_t eval_pipeline = 0;  ///< >1: eval via search_pipelined with this many
                               ///< in-flight query coroutines (Yi tasklet-style);
                               ///< 0/1 = classic one-query-per-thread eval
  uint32_t insert_batch = 32;
  uint32_t update_insert_threads = 32;
  uint32_t update_reconnect_threads = 4;
  uint32_t warmup_searches = 0;
  uint32_t updates_per_round = 0;
  uint32_t mixed_query_batch = 10000;
  uint32_t mixed_sleep_ms = 5000;
  double cache_ratio = kDefaultBenchmarkCacheRatio;
  uint64_t page_cache_capacity = kDefaultBenchmarkPageCachePages;
  alaya::diskann::DiskANNUpdateIO update_io = alaya::diskann::DiskANNUpdateIO::kAuto;
  uint32_t update_search_concurrency = 0;  // 0 = library default (4x insert threads)
  bool search_page_cache = true;           // searches peek+fill the shard page cache
  std::string oracle_ids;  ///< when set: batch-rebuild the graph on exactly this live
                           ///< label set (ids.bin format), then eval once — the
                           ///< "best achievable graph on the survivor set" reference
                           ///< that separates intrinsic geometry floor from
                           ///< incremental-process slack.
  uint32_t garden_pass = 0;        ///< when set with --eval_only on a kept index: run
                                   ///< garden_refresh(garden_pass) this many nodes per
                                   ///< iteration and re-eval, to test whether broader
                                   ///< garden COVERAGE (vs a full rebuild) recovers the
                                   ///< residual — coverage lever vs global-structural.
  uint32_t garden_pass_iters = 3;  ///< garden passes to run in --garden_pass mode.
};

DatasetFiles resolve_dataset_files(const std::filesystem::path &data_dir) {
  const std::vector<DatasetFiles> candidates = {
      {"sift",
       data_dir / "sift_base.fbin",
       data_dir / "sift_query.fbin",
       data_dir / "sift_gt.ibin"},
      {"gist",
       data_dir / "gist_base.fbin",
       data_dir / "gist_query.fbin",
       data_dir / "gist_gt.ibin"},
  };
  for (const DatasetFiles &files : candidates) {
    if (!std::filesystem::exists(files.base)) {
      continue;
    }
    if (!std::filesystem::exists(files.queries) || !std::filesystem::exists(files.groundtruth)) {
      throw std::runtime_error("incomplete " + files.name + " dataset under " + data_dir.string());
    }
    return files;
  }
  throw std::runtime_error("cannot find sift_base.fbin or gist_base.fbin under " +
                           data_dir.string());
}

FloatMatrix read_fbin(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path.string());
  }
  int32_t n = 0;
  int32_t dim = 0;
  in.read(reinterpret_cast<char *>(&n), sizeof(n));
  in.read(reinterpret_cast<char *>(&dim), sizeof(dim));
  if (!in || n <= 0 || dim <= 0) {
    throw std::runtime_error("bad fbin header: " + path.string());
  }
  FloatMatrix matrix;
  matrix.n = static_cast<uint32_t>(n);
  matrix.dim = static_cast<uint32_t>(dim);
  matrix.data.resize(static_cast<size_t>(matrix.n) * matrix.dim);
  in.read(reinterpret_cast<char *>(matrix.data.data()),
          static_cast<std::streamsize>(matrix.data.size() * sizeof(float)));
  if (!in) {
    throw std::runtime_error("short fbin read: " + path.string());
  }
  return matrix;
}

IntMatrix read_ibin(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path.string());
  }
  int32_t n = 0;
  int32_t dim = 0;
  in.read(reinterpret_cast<char *>(&n), sizeof(n));
  in.read(reinterpret_cast<char *>(&dim), sizeof(dim));
  if (!in || n <= 0 || dim <= 0) {
    throw std::runtime_error("bad ibin header: " + path.string());
  }
  IntMatrix matrix;
  matrix.n = static_cast<uint32_t>(n);
  matrix.dim = static_cast<uint32_t>(dim);
  matrix.data.resize(static_cast<size_t>(matrix.n) * matrix.dim);
  in.read(reinterpret_cast<char *>(matrix.data.data()),
          static_cast<std::streamsize>(matrix.data.size() * sizeof(uint32_t)));
  if (!in) {
    throw std::runtime_error("short ibin read: " + path.string());
  }
  return matrix;
}

uint32_t parse_u32(const std::string &value, const char *name) {
  const unsigned long parsed = std::stoul(value);
  if (parsed > UINT32_MAX) {
    throw std::invalid_argument(std::string(name) + " overflows uint32");
  }
  return static_cast<uint32_t>(parsed);
}

uint64_t parse_u64(const std::string &value, const char *name) {
  return static_cast<uint64_t>(std::stoull(value));
}

double parse_ratio(const std::string &value, const char *name) {
  const double parsed = std::stod(value);
  if (!std::isfinite(parsed) || parsed < 0.0 || parsed > 1.0) {
    throw std::invalid_argument(std::string(name) + " must be in [0, 1]");
  }
  return parsed;
}

const char *mixed_mode_name(MixedMode mode) {
  switch (mode) {
    case MixedMode::Background:
      return "background";
    case MixedMode::SharedQueue:
      return "shared_queue";
  }
  return "unknown";
}

MixedMode parse_mixed_mode(const std::string &value) {
  if (value == "background") {
    return MixedMode::Background;
  }
  if (value == "shared_queue") {
    return MixedMode::SharedQueue;
  }
  throw std::invalid_argument("bad --mixed_mode: " + value);
}

void print_usage(const char *argv0) {
  std::cerr << "Usage:\n"
            << "  " << argv0 << " [data_dir] [trace_dir] [index_dir] [out_csv] [flags]\n\n"
            << "Flags:\n"
            << "  --rebuild                  force a fresh index build\n"
            << "  --deterministic            deterministic search barriers\n"
            << "  --flush_rounds             flush dirty pages after each round\n"
            << "  --no_flush_rounds          skip per-round dirty-page flush\n"
            << "  --no_update_rerank         disable update rerank\n"
            << "  --update_insert_prune      alpha-prune insert candidates\n"
            << "  --update_repair            enable delete-time in-neighbor repair\n"
            << "  --no_update_repair         disable delete-time in-neighbor repair\n"
            << "  --repair_search_l N        repair search list (0 => deleted-node out-neighbors)\n"
            << "  --track_indegree           track live-slot in-degree percentiles\n"
            << "  --garden_budget N          refresh up to N low-in-degree live nodes per round\n"
            << "  --garden_l N               garden search list (0 => use --update_L)\n"
            << "  --garden_pass N            eval_only: garden_refresh(N) per pass, then re-eval\n"
            << "  --garden_pass_iters N      garden passes in --garden_pass mode (default 3)\n"
            << "  --oracle_ids FILE          batch-rebuild on FILE's ids.bin label set, eval once\n"
            << "  --build_only               build the initial index and exit\n"
            << "  --eval_only                replay live mask and run eval only\n"
            << "  --single_updates           issue single remove/insert calls\n"
            << "  --mixed                    run mixed update/search workload\n"
            << "  --mixed_mode MODE          background or shared_queue\n"
            << "  --rounds N                 cap update rounds\n"
            << "  --nq N                     cap eval queries (0 => all)\n";
}

TraceManifest read_manifest(const std::filesystem::path &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open trace manifest: " + path.string());
  }
  std::unordered_map<std::string, std::string> kv;
  for (std::string line; std::getline(in, line);) {
    const size_t eq = line.find('=');
    if (eq != std::string::npos) {
      kv[line.substr(0, eq)] = line.substr(eq + 1);
    }
  }
  TraceManifest m;
  m.prefix = kv.at("file_prefix");
  m.initial_count = parse_u32(kv.at("initial_count"), "initial_count");
  m.total_count = parse_u32(kv.at("total_count"), "total_count");
  m.rounds = parse_u32(kv.at("rounds"), "rounds");
  m.update_size = parse_u32(kv.at("update_size"), "update_size");
  return m;
}

TraceRound read_round(const std::filesystem::path &path, uint32_t expected_size) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open trace round: " + path.string());
  }
  uint32_t n = 0;
  in.read(reinterpret_cast<char *>(&n), sizeof(n));
  if (!in || n != expected_size) {
    throw std::runtime_error("bad trace round header: " + path.string());
  }
  TraceRound round;
  round.deletes.resize(n);
  round.inserts.resize(n);
  in.read(reinterpret_cast<char *>(round.deletes.data()),
          static_cast<std::streamsize>(round.deletes.size() * sizeof(uint32_t)));
  in.read(reinterpret_cast<char *>(round.inserts.data()),
          static_cast<std::streamsize>(round.inserts.size() * sizeof(uint32_t)));
  if (!in) {
    throw std::runtime_error("short trace round read: " + path.string());
  }
  return round;
}

Options parse_args(int argc, char **argv) {
  Options opt;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--rebuild") {
      opt.rebuild = true;
    } else if (arg == "--deterministic") {
      opt.deterministic = true;
    } else if (arg == "--flush_rounds") {
      opt.flush_rounds = true;
    } else if (arg == "--no_flush_rounds") {
      opt.flush_rounds = false;
    } else if (arg == "--no_update_rerank") {
      opt.update_rerank = false;
    } else if (arg == "--update_insert_prune") {
      opt.update_insert_prune = true;
    } else if (arg == "--update_repair") {
      opt.update_repair = true;
    } else if (arg == "--no_update_repair") {
      opt.update_repair = false;
    } else if (arg == "--track_indegree") {
      opt.track_indegree = true;
    } else if (arg == "--no_mixed_round0") {
      opt.mixed_round0_baseline = false;
    } else if (arg == "--build_only") {
      opt.build_only = true;
    } else if (arg == "--eval_only") {
      opt.eval_only = true;
    } else if (arg == "--oracle_ids" && i + 1 < argc) {
      opt.oracle_ids = argv[++i];
    } else if (arg == "--garden_pass" && i + 1 < argc) {
      opt.garden_pass = parse_u32(argv[++i], "--garden_pass");
      opt.track_indegree = true;
    } else if (arg == "--garden_pass_iters" && i + 1 < argc) {
      opt.garden_pass_iters = parse_u32(argv[++i], "--garden_pass_iters");
    } else if (arg == "--single_updates") {
      opt.single_updates = true;
    } else if (arg == "--mixed") {
      opt.mixed = true;
    } else if (arg == "--mixed_mode" && i + 1 < argc) {
      opt.mixed = true;
      opt.mixed_mode = parse_mixed_mode(argv[++i]);
    } else if (arg == "--rounds" && i + 1 < argc) {
      opt.max_rounds = parse_u32(argv[++i], "--rounds");
    } else if (arg == "--nq" && i + 1 < argc) {
      opt.nq = parse_u32(argv[++i], "--nq");
    } else if (arg == "--R" && i + 1 < argc) {
      opt.build_r = std::max<uint32_t>(1, parse_u32(argv[++i], "--R"));
    } else if (arg == "--capacity" && i + 1 < argc) {
      opt.record_capacity = parse_u32(argv[++i], "--capacity");  // 0 => R
    } else if (arg == "--build_L" && i + 1 < argc) {
      opt.build_l = std::max<uint32_t>(1, parse_u32(argv[++i], "--build_L"));
    } else if (arg == "--L" && i + 1 < argc) {
      opt.search_l = parse_u32(argv[++i], "--L");
    } else if (arg == "--rerank_count" && i + 1 < argc) {
      opt.rerank_count = parse_u32(argv[++i], "--rerank_count");
    } else if (arg == "--update_L" && i + 1 < argc) {
      opt.update_l = parse_u32(argv[++i], "--update_L");  // 0 => R + 32
    } else if (arg == "--repair_search_l" && i + 1 < argc) {
      opt.repair_search_l = parse_u32(argv[++i], "--repair_search_l");
    } else if (arg == "--garden_budget" && i + 1 < argc) {
      opt.garden_budget = parse_u32(argv[++i], "--garden_budget");
    } else if (arg == "--garden_l" && i + 1 < argc) {
      opt.garden_l = parse_u32(argv[++i], "--garden_l");
    } else if (arg == "--beam" && i + 1 < argc) {
      opt.beam = parse_u32(argv[++i], "--beam");
    } else if (arg == "--threads" && i + 1 < argc) {
      opt.search_threads = std::max<uint32_t>(1, parse_u32(argv[++i], "--threads"));
    } else if (arg == "--eval_pipeline" && i + 1 < argc) {
      opt.eval_pipeline = parse_u32(argv[++i], "--eval_pipeline");
    } else if (arg == "--insert_batch" && i + 1 < argc) {
      opt.insert_batch = std::max<uint32_t>(1, parse_u32(argv[++i], "--insert_batch"));
    } else if (arg == "--update_io" && i + 1 < argc) {
      const std::string mode = argv[++i];
      if (mode == "auto") {
        opt.update_io = alaya::diskann::DiskANNUpdateIO::kAuto;
      } else if (mode == "uring") {
        opt.update_io = alaya::diskann::DiskANNUpdateIO::kUring;
      } else if (mode == "blocking") {
        opt.update_io = alaya::diskann::DiskANNUpdateIO::kBlocking;
      } else {
        throw std::invalid_argument("--update_io expects auto|uring|blocking");
      }
    } else if (arg == "--update_search_concurrency" && i + 1 < argc) {
      opt.update_search_concurrency = parse_u32(argv[++i], "--update_search_concurrency");
    } else if (arg == "--update_insert_threads" && i + 1 < argc) {
      opt.update_insert_threads =
          std::max<uint32_t>(1, parse_u32(argv[++i], "--update_insert_threads"));
    } else if (arg == "--update_reconnect_threads" && i + 1 < argc) {
      opt.update_reconnect_threads =
          std::max<uint32_t>(1, parse_u32(argv[++i], "--update_reconnect_threads"));
    } else if (arg == "--warmup_searches" && i + 1 < argc) {
      opt.warmup_searches = parse_u32(argv[++i], "--warmup_searches");
    } else if (arg == "--updates_per_round" && i + 1 < argc) {
      opt.updates_per_round = parse_u32(argv[++i], "--updates_per_round");
    } else if (arg == "--mixed_query_batch" && i + 1 < argc) {
      opt.mixed_query_batch = std::max<uint32_t>(1, parse_u32(argv[++i], "--mixed_query_batch"));
    } else if (arg == "--mixed_sleep_ms" && i + 1 < argc) {
      opt.mixed_sleep_ms = parse_u32(argv[++i], "--mixed_sleep_ms");
    } else if (arg == "--cache_ratio" && i + 1 < argc) {
      opt.cache_ratio = parse_ratio(argv[++i], "--cache_ratio");
    } else if (arg == "--page_cache" && i + 1 < argc) {
      opt.page_cache_capacity = parse_u64(argv[++i], "--page_cache");
    } else if (arg == "--search_page_cache" && i + 1 < argc) {
      opt.search_page_cache = parse_u64(argv[++i], "--search_page_cache") != 0;
    } else {
      if (arg.rfind("--", 0) == 0) {
        throw std::invalid_argument("unknown option: " + arg);
      }
      pos.push_back(arg);
    }
  }
  if (pos.size() > 0) {
    opt.data_dir = pos[0];
  }
  if (pos.size() > 1) {
    opt.trace_dir = pos[1];
  }
  if (pos.size() > 2) {
    opt.index_dir = pos[2];
  }
  if (pos.size() > 3) {
    opt.out_csv = pos[3];
  }
  if (opt.garden_budget > 0) {
    opt.track_indegree = true;
  }
  return opt;
}

TraceRound limit_round_updates(TraceRound round, const Options &opt) {
  if (opt.updates_per_round == 0) {
    return round;
  }
  const size_t count = static_cast<size_t>(opt.updates_per_round);
  if (count > round.deletes.size() || count > round.inserts.size()) {
    throw std::invalid_argument("--updates_per_round exceeds trace round size");
  }
  round.deletes.resize(count);
  round.inserts.resize(count);
  return round;
}

// ids.bin format (matches the on-disk index and fig13_forensics.load_ids):
// [uint64 count][count x uint64 label]. Used to seed the oracle rebuild's live
// label set directly from a kept index's ids.bin.
std::vector<uint64_t> read_ids_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open oracle ids file: " + path);
  }
  uint64_t count = 0;
  in.read(reinterpret_cast<char *>(&count), sizeof(count));
  if (!in) {
    throw std::runtime_error("oracle ids file truncated header: " + path);
  }
  // Validate the header count against the actual file size BEFORE allocating:
  // a corrupt count would otherwise request a giant vector up front.
  const uint64_t payload_bytes =
      static_cast<uint64_t>(std::filesystem::file_size(path)) - sizeof(uint64_t);
  if (count > payload_bytes / sizeof(uint64_t)) {
    throw std::runtime_error("oracle ids file count " + std::to_string(count) +
                             " exceeds file size: " + path);
  }
  std::vector<uint64_t> labels(count);
  in.read(reinterpret_cast<char *>(labels.data()),
          static_cast<std::streamsize>(count * sizeof(uint64_t)));
  if (!in) {
    throw std::runtime_error("oracle ids file truncated body: " + path);
  }
  return labels;
}

void build_initial_index(const Options &opt,
                         const FloatMatrix &base,
                         const TraceManifest &manifest) {
  namespace fs = std::filesystem;
  if (opt.rebuild && fs::exists(opt.index_dir)) {
    fs::remove_all(opt.index_dir);
  }
  if (fs::exists(opt.index_dir)) {
    std::cout << "[update_bench] reusing index " << opt.index_dir << "\n";
    return;
  }
  DiskANNBuildParams bp;
  bp.R = opt.build_r;
  bp.L = opt.build_l;
  bp.record_capacity = opt.record_capacity;
  bp.alpha = 1.2f;
  bp.pq_n_chunks = 32;
  bp.cache_ratio = opt.cache_ratio;
  bp.num_threads = 96;
  bp.seed = 1234;
  bp.verbose = true;

  if (!opt.oracle_ids.empty()) {
    // Oracle: batch-rebuild the graph on exactly the survivor+inserted label
    // set (original numbering preserved), gathering each node's coords from the
    // base by label. Preserving labels lets the same GT + age buckets score it.
    const std::vector<uint64_t> labels = read_ids_file(opt.oracle_ids);
    const uint32_t n = static_cast<uint32_t>(labels.size());
    std::vector<float> coords(static_cast<size_t>(n) * base.dim);
    for (uint32_t i = 0; i < n; ++i) {
      const uint64_t label = labels[i];
      if (label >= base.n) {
        throw std::runtime_error("oracle label out of base range: " + std::to_string(label));
      }
      std::copy_n(base.data.data() + static_cast<size_t>(label) * base.dim,
                  base.dim,
                  coords.data() + static_cast<size_t>(i) * base.dim);
    }
    std::cout << "[update_bench] building ORACLE PQ index (batch rebuild on survivor set), n="
              << n << " dir=" << opt.index_dir << "\n";
    DiskANNIndex::build(opt.index_dir, coords.data(), labels.data(), n, base.dim, bp);
    return;
  }

  std::vector<uint64_t> labels(manifest.initial_count);
  for (uint32_t id = 0; id < manifest.initial_count; ++id) {
    labels[id] = id;
  }
  std::cout << "[update_bench] building initial PQ index, n=" << manifest.initial_count
            << " dir=" << opt.index_dir << "\n";
  DiskANNIndex::build(opt.index_dir,
                      base.data.data(),
                      labels.data(),
                      manifest.initial_count,
                      base.dim,
                      bp);
}

struct RecallResult {
  uint64_t hits = 0;
  uint64_t total = 0;
  double mean_us = 0.0;
  double qps = 0.0;
  double recall() const { return total == 0 ? 0.0 : static_cast<double>(hits) / total; }
};

struct MixedSearchState {
  std::atomic<bool> stop{false};
  std::atomic<uint32_t> ready{0};
  std::atomic<uint32_t> next_query{0};
  std::atomic<uint32_t> window_queries{0};
  std::atomic<bool> window_paused{false};
  std::atomic<uint64_t> queries{0};
  std::atomic<uint64_t> latency_ns{0};
  std::atomic<uint64_t> active_search_ns{0};
  std::vector<std::thread> workers;
  std::chrono::steady_clock::time_point started_at{};
  std::chrono::steady_clock::time_point stopped_at{};
  std::exception_ptr error = nullptr;
  std::mutex error_mutex;
};

struct MixedSearchResult {
  uint64_t queries = 0;
  double mean_us = 0.0;
  double qps = 0.0;
};

DiskANNSearchParams make_search_params(const Options &opt) {
  DiskANNSearchParams sp;
  sp.search_list_size = opt.search_l;
  sp.use_pq = true;
  sp.rerank = true;
  sp.rerank_count = opt.rerank_count == 0 ? opt.search_l : opt.rerank_count;
  sp.deterministic = opt.deterministic;
  return sp;
}

RecallResult evaluate_search(const DiskANNIndex &idx,
                             const FloatMatrix &queries,
                             const IntMatrix &gt,
                             const std::vector<uint8_t> &live,
                             const Options &opt) {
  const uint32_t nq = opt.nq == 0 ? queries.n : std::min<uint32_t>(opt.nq, queries.n);
  const DiskANNSearchParams sp = make_search_params(opt);

  std::vector<uint64_t> labels(static_cast<size_t>(nq) * kBenchmarkTopK);
  std::vector<float> distances(static_cast<size_t>(nq) * kBenchmarkTopK);
  std::vector<double> lat_us(nq, 0.0);
  std::atomic<uint32_t> next{0};
  std::atomic<uint64_t> tot_ios{0};
  std::atomic<uint64_t> tot_page_hits{0};
  std::atomic<uint64_t> tot_bfs_hits{0};
  std::atomic<uint64_t> tot_nodes{0};
  std::atomic<uint64_t> tot_waits{0};
  std::atomic<uint64_t> tot_inflight{0};
  std::atomic<uint64_t> tot_wait_us{0};
  std::atomic<uint64_t> tot_proc_us{0};
  std::atomic<uint64_t> tot_setup_us{0};
  std::atomic<uint64_t> tot_peek_us{0};
  std::atomic<uint64_t> tot_fillpg_us{0};
  std::atomic<uint64_t> tot_rerank_reads{0};
  auto worker = [&]() {
    uint64_t ios = 0;
    uint64_t page_hits = 0;
    uint64_t bfs_hits = 0;
    uint64_t nodes = 0;
    for (;;) {
      const uint32_t qi = next.fetch_add(1, std::memory_order_relaxed);
      if (qi >= nq) {
        break;
      }
      const float *query = queries.data.data() + static_cast<size_t>(qi) * queries.dim;
      alaya::diskann::SearchStats stats;
      auto t0 = std::chrono::steady_clock::now();
      idx.search(query,
                 kBenchmarkTopK,
                 labels.data() + static_cast<size_t>(qi) * kBenchmarkTopK,
                 distances.data() + static_cast<size_t>(qi) * kBenchmarkTopK,
                 sp,
                 &stats);
      auto t1 = std::chrono::steady_clock::now();
      lat_us[qi] = std::chrono::duration<double, std::micro>(t1 - t0).count();
      ios += stats.n_ios;
      page_hits += stats.n_page_cache_hits;
      bfs_hits += stats.n_cache_hits;
      nodes += stats.n_nodes_processed;
      tot_waits.fetch_add(stats.n_waits, std::memory_order_relaxed);
      tot_inflight.fetch_add(stats.inflight_sum, std::memory_order_relaxed);
      tot_wait_us.fetch_add(stats.wait_us, std::memory_order_relaxed);
      tot_proc_us.fetch_add(stats.proc_us, std::memory_order_relaxed);
      tot_setup_us.fetch_add(stats.setup_us, std::memory_order_relaxed);
      tot_peek_us.fetch_add(stats.peek_us, std::memory_order_relaxed);
      tot_fillpg_us.fetch_add(stats.fillpg_us, std::memory_order_relaxed);
      tot_rerank_reads.fetch_add(stats.n_rerank_reads, std::memory_order_relaxed);
    }
    tot_ios.fetch_add(ios, std::memory_order_relaxed);
    tot_page_hits.fetch_add(page_hits, std::memory_order_relaxed);
    tot_bfs_hits.fetch_add(bfs_hits, std::memory_order_relaxed);
    tot_nodes.fetch_add(nodes, std::memory_order_relaxed);
  };
#if defined(__linux__)
  if (opt.eval_pipeline > 1) {
    // Scratch first-touch (~12 B/slot per in-flight slot) must not land in the
    // timed window: at depth 1024 on 10M slots it rivals the queries themselves.
    idx.prewarm_pipeline(opt.search_threads, opt.eval_pipeline);
  }
#endif
  const auto wall0 = std::chrono::steady_clock::now();
#if defined(__linux__)
  if (opt.eval_pipeline > 1) {
    // Yi-style query pipelining: threads keep eval_pipeline queries in flight
    // as coroutines. rerank is forced off — a no-op in this protocol (every
    // retset entry is expanded during traversal; rerank_reads=0 on all legs).
    std::vector<alaya::diskann::SearchStats> qstats(nq);
    DiskANNSearchParams psp = sp;
    psp.rerank = false;
    idx.search_pipelined(queries.data.data(),
                         nq,
                         kBenchmarkTopK,
                         labels.data(),
                         distances.data(),
                         opt.search_threads,
                         opt.eval_pipeline,
                         psp,
                         qstats.data(),
                         lat_us.data());
    for (uint32_t qi = 0; qi < nq; ++qi) {
      const auto &st = qstats[qi];
      tot_ios.fetch_add(st.n_ios, std::memory_order_relaxed);
      tot_page_hits.fetch_add(st.n_page_cache_hits, std::memory_order_relaxed);
      tot_bfs_hits.fetch_add(st.n_cache_hits, std::memory_order_relaxed);
      tot_nodes.fetch_add(st.n_nodes_processed, std::memory_order_relaxed);
      tot_waits.fetch_add(st.n_waits, std::memory_order_relaxed);
      tot_inflight.fetch_add(st.inflight_sum, std::memory_order_relaxed);
      tot_wait_us.fetch_add(st.wait_us, std::memory_order_relaxed);
      tot_proc_us.fetch_add(st.proc_us, std::memory_order_relaxed);
      tot_setup_us.fetch_add(st.setup_us, std::memory_order_relaxed);
      tot_peek_us.fetch_add(st.peek_us, std::memory_order_relaxed);
      tot_fillpg_us.fetch_add(st.fillpg_us, std::memory_order_relaxed);
      tot_rerank_reads.fetch_add(st.n_rerank_reads, std::memory_order_relaxed);
    }
  } else {
#else
  {
#endif
    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < opt.search_threads; ++i) {
      threads.emplace_back(worker);
    }
    for (auto &thread : threads) {
      thread.join();
    }
  }
  const auto wall1 = std::chrono::steady_clock::now();

  std::cout << "[eval io] per-query ios=" << static_cast<double>(tot_ios.load()) / nq
            << " page_hits=" << static_cast<double>(tot_page_hits.load()) / nq
            << " bfs_hits=" << static_cast<double>(tot_bfs_hits.load()) / nq
            << " nodes=" << static_cast<double>(tot_nodes.load()) / nq << " hit_ratio="
            << (static_cast<double>(tot_page_hits.load() + tot_bfs_hits.load()) /
                std::max<double>(1.0,
                                 static_cast<double>(tot_ios.load() + tot_page_hits.load() +
                                                     tot_bfs_hits.load())))
            << " depth="
            << (static_cast<double>(tot_inflight.load()) /
                std::max<double>(1.0, static_cast<double>(tot_waits.load())))
            << " waits=" << static_cast<double>(tot_waits.load()) / nq
            << " wait_us=" << static_cast<double>(tot_wait_us.load()) / nq
            << " proc_us=" << static_cast<double>(tot_proc_us.load()) / nq
            << " setup_us=" << static_cast<double>(tot_setup_us.load()) / nq
            << " peek_us=" << static_cast<double>(tot_peek_us.load()) / nq
            << " fillpg_us=" << static_cast<double>(tot_fillpg_us.load()) / nq
            << " rerank_reads=" << static_cast<double>(tot_rerank_reads.load()) / nq
            << " pipeline=" << std::max<uint32_t>(1, opt.eval_pipeline) << "\n";

  RecallResult result;
  // Recall by GT-id bucket: sequential traces assign ids in insertion order, so
  // the id encodes node age (original build vs round-r insert). The per-bucket
  // split separates "old survivors got eroded" from "new inserts are wired
  // weakly" — indistinguishable in the aggregate number.
  constexpr uint32_t kIdBucketWidth = 500000;
  std::vector<uint64_t> bucket_hits(20, 0);
  std::vector<uint64_t> bucket_total(20, 0);
  for (uint32_t qi = 0; qi < nq; ++qi) {
    std::unordered_set<uint64_t> got;
    const uint64_t *row = labels.data() + static_cast<size_t>(qi) * kBenchmarkTopK;
    for (uint32_t i = 0; i < kBenchmarkTopK; ++i) {
      if (row[i] != DiskANNIndex::kNoLabel) {
        got.insert(row[i]);
      }
    }
    const uint32_t *truth = gt.row(qi);
    for (uint32_t i = 0; i < kBenchmarkTopK && i < gt.dim; ++i) {
      if (truth[i] < live.size() && live[truth[i]] != 0) {
        ++result.total;
        const size_t bucket = std::min<size_t>(truth[i] / kIdBucketWidth, bucket_hits.size() - 1);
        ++bucket_total[bucket];
        if (got.count(truth[i]) != 0) {
          ++result.hits;
          ++bucket_hits[bucket];
        }
      }
    }
    result.mean_us += lat_us[qi];
  }
  std::cout << "[recall_buckets] w=" << kIdBucketWidth;
  for (size_t bucket = 0; bucket < bucket_hits.size(); ++bucket) {
    std::cout << " " << bucket_hits[bucket] << "/" << bucket_total[bucket];
  }
  std::cout << "\n";
  result.mean_us /= nq;
  result.qps = static_cast<double>(nq) / std::chrono::duration<double>(wall1 - wall0).count();
  return result;
}

void record_mixed_search_error(MixedSearchState &state) {
  std::lock_guard<std::mutex> lock(state.error_mutex);
  if (state.error == nullptr) {
    state.error = std::current_exception();
  }
  state.stop.store(true, std::memory_order_release);
}

void sleep_until_next_mixed_window(MixedSearchState &state, uint32_t sleep_ms) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(sleep_ms);
  while (!state.stop.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void maybe_pause_mixed_search_window(MixedSearchState &state, const Options &opt) {
  const uint32_t finished = state.window_queries.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (finished < opt.mixed_query_batch) {
    return;
  }
  bool expected = false;
  if (!state.window_paused.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    return;
  }
  state.window_queries.store(0, std::memory_order_release);
  sleep_until_next_mixed_window(state, opt.mixed_sleep_ms);
  state.window_paused.store(false, std::memory_order_release);
}

void wait_for_mixed_search_window(MixedSearchState &state) {
  while (!state.stop.load(std::memory_order_acquire) &&
         state.window_paused.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void run_mixed_search_query(MixedSearchState &state,
                            const DiskANNIndex &idx,
                            const FloatMatrix &queries,
                            uint32_t query_id,
                            const DiskANNSearchParams &sp) {
  std::array<uint64_t, kBenchmarkTopK> labels{};
  std::array<float, kBenchmarkTopK> distances{};
  const float *query = queries.data.data() + static_cast<size_t>(query_id) * queries.dim;
  const auto t0 = std::chrono::steady_clock::now();
  idx.search(query, kBenchmarkTopK, labels.data(), distances.data(), sp);
  const auto t1 = std::chrono::steady_clock::now();
  const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  state.latency_ns.fetch_add(static_cast<uint64_t>(ns), std::memory_order_relaxed);
  state.queries.fetch_add(1, std::memory_order_relaxed);
}

void start_mixed_search(MixedSearchState &state,
                        const DiskANNIndex &idx,
                        const FloatMatrix &queries,
                        const Options &opt) {
  const uint32_t nq = opt.nq == 0 ? queries.n : std::min<uint32_t>(opt.nq, queries.n);
  if (nq == 0) {
    throw std::invalid_argument("--mixed requires at least one query");
  }
  const DiskANNSearchParams sp = make_search_params(opt);
  const uint32_t workers = std::max<uint32_t>(1, opt.search_threads);
  state.started_at = std::chrono::steady_clock::now();
  state.workers.reserve(workers);
  for (uint32_t worker_id = 0; worker_id < workers; ++worker_id) {
    state.workers.emplace_back([&, worker_id]() {
      (void)worker_id;
      state.ready.fetch_add(1, std::memory_order_release);
      try {
        while (!state.stop.load(std::memory_order_acquire)) {
          wait_for_mixed_search_window(state);
          if (state.stop.load(std::memory_order_acquire)) {
            break;
          }
          const uint32_t seq = state.next_query.fetch_add(1, std::memory_order_relaxed);
          const uint32_t qi = seq % nq;
          run_mixed_search_query(state, idx, queries, qi, sp);
          maybe_pause_mixed_search_window(state, opt);
        }
      } catch (...) {
        record_mixed_search_error(state);
      }
    });
  }
  while (state.ready.load(std::memory_order_acquire) < workers) {
    std::this_thread::yield();
  }
}

void start_shared_queue_mixed_search(MixedSearchState &state,
                                     const DiskANNIndex &idx,
                                     const FloatMatrix &queries,
                                     const Options &opt,
                                     coro::thread_pool &pool) {
  const uint32_t nq = opt.nq == 0 ? queries.n : std::min<uint32_t>(opt.nq, queries.n);
  if (nq == 0) {
    throw std::invalid_argument("--mixed requires at least one query");
  }
  const DiskANNSearchParams sp = make_search_params(opt);
  state.started_at = std::chrono::steady_clock::now();
  state.workers.emplace_back([&, nq, sp]() {
    state.ready.fetch_add(1, std::memory_order_release);
    try {
      auto search_one = [&]() -> coro::task<> {
        co_await pool.schedule();
        if (state.stop.load(std::memory_order_acquire)) {
          co_return;
        }
        const uint32_t seq = state.next_query.fetch_add(1, std::memory_order_relaxed);
        run_mixed_search_query(state, idx, queries, seq % nq, sp);
      };
      sleep_until_next_mixed_window(state, opt.mixed_sleep_ms);
      while (!state.stop.load(std::memory_order_acquire)) {
        auto run_window = [&]() -> coro::task<> {
          std::vector<coro::task<>> tasks;
          tasks.reserve(opt.mixed_query_batch);
          for (uint32_t i = 0; i < opt.mixed_query_batch; ++i) {
            if (state.stop.load(std::memory_order_acquire)) {
              break;
            }
            tasks.emplace_back(search_one());
          }
          co_await coro::when_all(std::move(tasks));
        };
        const auto window_start = std::chrono::steady_clock::now();
        coro::sync_wait(run_window());
        const auto window_end = std::chrono::steady_clock::now();
        const auto active_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(window_end - window_start).count();
        state.active_search_ns.fetch_add(static_cast<uint64_t>(active_ns),
                                         std::memory_order_relaxed);
        sleep_until_next_mixed_window(state, opt.mixed_sleep_ms);
      }
    } catch (...) {
      record_mixed_search_error(state);
    }
  });
  while (state.ready.load(std::memory_order_acquire) < 1) {
    std::this_thread::yield();
  }
}

void request_mixed_search_stop(MixedSearchState &state) {
  state.stop.store(true, std::memory_order_release);
}

void join_mixed_search(MixedSearchState &state) {
  for (auto &worker : state.workers) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  state.stopped_at = std::chrono::steady_clock::now();
}

MixedSearchResult finish_mixed_search(MixedSearchState &state) {
  request_mixed_search_stop(state);
  join_mixed_search(state);
  if (state.error != nullptr) {
    std::rethrow_exception(state.error);
  }
  MixedSearchResult result;
  result.queries = state.queries.load(std::memory_order_relaxed);
  if (result.queries == 0) {
    return result;
  }
  const uint64_t active_search_ns = state.active_search_ns.load(std::memory_order_relaxed);
  const double wall_s =
      active_search_ns == 0
          ? std::chrono::duration<double>(state.stopped_at - state.started_at).count()
          : static_cast<double>(active_search_ns) / 1'000'000'000.0;
  result.mean_us = static_cast<double>(state.latency_ns.load(std::memory_order_relaxed)) /
                   result.queries / 1000.0;
  result.qps = wall_s <= 0.0 ? 0.0 : static_cast<double>(result.queries) / wall_s;
  return result;
}

template <typename MakeTask>
void run_indexed_tasks(uint32_t count, MakeTask make_task) {
  auto run = [&]() -> coro::task<> {
    std::vector<coro::task<>> tasks;
    tasks.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
      tasks.emplace_back(make_task(i));
    }
    co_await coro::when_all(std::move(tasks));
  };
  coro::sync_wait(run());
}

void apply_deletes(DiskANNIndex &idx,
                   const TraceRound &round,
                   const Options &opt,
                   std::vector<uint8_t> &live,
                   std::vector<uint32_t> &label_to_slot) {
  if (opt.single_updates) {
    for (const uint32_t label : round.deletes) {
      if (label >= live.size() || live[label] == 0 || label_to_slot[label] == kMissingSlot) {
        throw std::runtime_error("trace delete id is not live: " + std::to_string(label));
      }
      idx.remove(label_to_slot[label]);
      live[label] = 0;
      label_to_slot[label] = kMissingSlot;
    }
    return;
  }

  std::vector<uint32_t> slots;
  slots.reserve(round.deletes.size());
  for (const uint32_t label : round.deletes) {
    if (label >= live.size() || live[label] == 0 || label_to_slot[label] == kMissingSlot) {
      throw std::runtime_error("trace delete id is not live: " + std::to_string(label));
    }
    slots.push_back(label_to_slot[label]);
  }
  idx.batch_remove(slots.data(), static_cast<uint32_t>(slots.size()));
  for (const uint32_t label : round.deletes) {
    live[label] = 0;
    label_to_slot[label] = kMissingSlot;
  }
}

void apply_deletes_shared(DiskANNIndex &idx,
                          const TraceRound &round,
                          const Options &opt,
                          std::vector<uint8_t> &live,
                          std::vector<uint32_t> &label_to_slot,
                          coro::thread_pool &pool) {
  std::vector<uint32_t> slots;
  slots.reserve(round.deletes.size());
  for (const uint32_t label : round.deletes) {
    if (label >= live.size() || live[label] == 0 || label_to_slot[label] == kMissingSlot) {
      throw std::runtime_error("trace delete id is not live: " + std::to_string(label));
    }
    slots.push_back(label_to_slot[label]);
  }

  if (opt.single_updates) {
    auto remove_one = [&idx, &pool, &slots](uint32_t i) -> coro::task<> {
      co_await pool.schedule();
      idx.remove(slots[i]);
    };
    run_indexed_tasks(static_cast<uint32_t>(slots.size()), remove_one);
  } else {
    idx.batch_remove_with_pool(slots.data(), static_cast<uint32_t>(slots.size()), pool);
  }
  for (const uint32_t label : round.deletes) {
    live[label] = 0;
    label_to_slot[label] = kMissingSlot;
  }
}

void apply_inserts(DiskANNIndex &idx,
                   const FloatMatrix &base,
                   const TraceRound &round,
                   const Options &opt,
                   std::vector<uint8_t> &live,
                   std::vector<uint32_t> &label_to_slot) {
  if (opt.single_updates) {
    for (const uint32_t label : round.inserts) {
      if (label >= live.size() || live[label] != 0) {
        throw std::runtime_error("trace insert id is already live: " + std::to_string(label));
      }
      const float *src = base.data.data() + static_cast<size_t>(label) * base.dim;
      label_to_slot[label] = idx.insert(src, label);
      live[label] = 1;
    }
    return;
  }

  std::vector<float> batch_vectors(static_cast<size_t>(round.inserts.size()) * base.dim);
  std::vector<uint64_t> batch_labels(round.inserts.size());
  for (size_t i = 0; i < round.inserts.size(); ++i) {
    const uint32_t label = round.inserts[i];
    if (label >= live.size() || live[label] != 0) {
      throw std::runtime_error("trace insert id is already live: " + std::to_string(label));
    }
    const float *src = base.data.data() + static_cast<size_t>(label) * base.dim;
    std::copy(src, src + base.dim, batch_vectors.data() + i * base.dim);
    batch_labels[i] = label;
  }
  const std::vector<uint32_t> slots = idx.batch_insert(batch_vectors.data(),
                                                       batch_labels.data(),
                                                       static_cast<uint32_t>(batch_labels.size()),
                                                       opt.insert_batch);
  for (size_t i = 0; i < batch_labels.size(); ++i) {
    const uint32_t label = static_cast<uint32_t>(batch_labels[i]);
    live[label] = 1;
    label_to_slot[label] = slots[i];
  }
}

void apply_inserts_shared(DiskANNIndex &idx,
                          const FloatMatrix &base,
                          const TraceRound &round,
                          const Options &opt,
                          std::vector<uint8_t> &live,
                          std::vector<uint32_t> &label_to_slot,
                          coro::thread_pool &pool) {
  std::vector<float> batch_vectors(static_cast<size_t>(round.inserts.size()) * base.dim);
  std::vector<uint64_t> batch_labels(round.inserts.size());
  for (size_t i = 0; i < round.inserts.size(); ++i) {
    const uint32_t label = round.inserts[i];
    if (label >= live.size() || live[label] != 0) {
      throw std::runtime_error("trace insert id is already live: " + std::to_string(label));
    }
    const float *src = base.data.data() + static_cast<size_t>(label) * base.dim;
    std::copy(src, src + base.dim, batch_vectors.data() + i * base.dim);
    batch_labels[i] = label;
  }

  std::vector<uint32_t> slots(batch_labels.size());
  if (opt.single_updates) {
    auto insert_one =
        [&idx, &pool, &batch_vectors, &batch_labels, &slots, &base](uint32_t i) -> coro::task<> {
      co_await pool.schedule();
      const float *src = batch_vectors.data() + static_cast<size_t>(i) * base.dim;
      slots[i] = idx.insert(src, batch_labels[i]);
    };
    run_indexed_tasks(static_cast<uint32_t>(slots.size()), insert_one);
  } else {
    slots = idx.batch_insert_with_pool(batch_vectors.data(),
                                       batch_labels.data(),
                                       static_cast<uint32_t>(batch_labels.size()),
                                       opt.insert_batch,
                                       pool);
  }
  for (size_t i = 0; i < batch_labels.size(); ++i) {
    const uint32_t label = static_cast<uint32_t>(batch_labels[i]);
    live[label] = 1;
    label_to_slot[label] = slots[i];
  }
}

void apply_live_mask_only(const TraceRound &round, std::vector<uint8_t> &live) {
  for (const uint32_t label : round.deletes) {
    if (label >= live.size() || live[label] == 0) {
      throw std::runtime_error("trace delete id is not live: " + std::to_string(label));
    }
    live[label] = 0;
  }
  for (const uint32_t label : round.inserts) {
    if (label >= live.size() || live[label] != 0) {
      throw std::runtime_error("trace insert id is already live: " + std::to_string(label));
    }
    live[label] = 1;
  }
}

void run_warmup_searches(const DiskANNIndex &idx,
                         const FloatMatrix &queries,
                         const IntMatrix &gt,
                         const std::vector<uint8_t> &live,
                         const Options &opt) {
  for (uint32_t i = 0; i < opt.warmup_searches; ++i) {
    const RecallResult recall = evaluate_search(idx, queries, gt, live, opt);
    std::cout << "[warmup " << i << "] masked_recall@10=" << recall.recall()
              << " search_mean_us=" << recall.mean_us << " search_qps=" << recall.qps << "\n";
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "-h" || arg == "--help") {
        print_usage(argv[0]);
        return 0;
      }
    }
    const Options opt = parse_args(argc, argv);
    const std::filesystem::path data_dir = opt.data_dir;
    const std::filesystem::path trace_dir = opt.trace_dir;
    const TraceManifest manifest = read_manifest(trace_dir / "manifest.txt");
    if (opt.updates_per_round > manifest.update_size) {
      throw std::invalid_argument("--updates_per_round exceeds trace update_size");
    }

    const DatasetFiles dataset = resolve_dataset_files(data_dir);
    FloatMatrix base = read_fbin(dataset.base);
    FloatMatrix queries = read_fbin(dataset.queries);
    IntMatrix gt = read_ibin(dataset.groundtruth);
    if (base.n != manifest.total_count || queries.dim != base.dim || gt.n != queries.n) {
      throw std::runtime_error(dataset.name + " data and trace manifest are inconsistent");
    }

    build_initial_index(opt, base, manifest);
    if (opt.build_only) {
      std::cout << "[update_bench] build_only complete: " << opt.index_dir << "\n";
      return 0;
    }
    DiskANNLoadParams lp;
    lp.num_threads = std::max<uint32_t>(opt.search_threads, 1);
    lp.beam_width = opt.beam;
    lp.updatable = true;
    lp.update_search_l = opt.update_l;
    lp.update_rerank = opt.update_rerank;
    lp.update_insert_prune = opt.update_insert_prune;
    lp.update_repair = opt.update_repair;
    lp.repair_search_l = opt.repair_search_l;
    lp.track_in_degree = opt.track_indegree;
    lp.garden_search_l = opt.garden_l;
    lp.update_insert_threads = opt.update_insert_threads;
    lp.update_reconnect_threads = opt.update_reconnect_threads;
    lp.page_cache_capacity = opt.page_cache_capacity;
    lp.update_io = opt.update_io;
    lp.update_search_concurrency = opt.update_search_concurrency;
    lp.search_page_cache = opt.search_page_cache;
    DiskANNIndex idx;
    idx.load(opt.index_dir, lp);

    std::vector<uint8_t> live(manifest.total_count, 0);
    std::vector<uint32_t> label_to_slot(manifest.total_count, kMissingSlot);
    if (!opt.oracle_ids.empty()) {
      // Oracle: the live set is exactly the rebuilt label set. Mark them alive so
      // the same masked-recall + age buckets score the batch-rebuilt graph.
      const std::vector<uint64_t> oracle_labels = read_ids_file(opt.oracle_ids);
      for (const uint64_t label : oracle_labels) {
        if (label < live.size()) {
          live[label] = 1;
        }
      }
      run_warmup_searches(idx, queries, gt, live, opt);
      const RecallResult recall = evaluate_search(idx, queries, gt, live, opt);
      std::cout << "[oracle] n=" << oracle_labels.size()
                << " masked_recall@10=" << recall.recall()
                << " search_mean_us=" << recall.mean_us << " search_qps=" << recall.qps << "\n";
      return 0;
    }
    for (uint32_t id = 0; id < manifest.initial_count; ++id) {
      live[id] = 1;
      label_to_slot[id] = id;
    }

    const uint32_t rounds =
        opt.max_rounds == 0 ? manifest.rounds : std::min<uint32_t>(opt.max_rounds, manifest.rounds);
    if (opt.eval_only) {
      for (uint32_t round_id = 0; round_id < rounds; ++round_id) {
        const auto path = trace_dir / (manifest.prefix + std::to_string(round_id));
        apply_live_mask_only(limit_round_updates(read_round(path, manifest.update_size), opt),
                             live);
      }
      run_warmup_searches(idx, queries, gt, live, opt);
      const RecallResult recall = evaluate_search(idx, queries, gt, live, opt);
      std::cout << "[eval_only] rounds=" << rounds << " masked_recall@10=" << recall.recall()
                << " search_mean_us=" << recall.mean_us << " search_qps=" << recall.qps << "\n";
      // Coverage probe: does gardening the WHOLE survivor set (vs garden's 50k/
      // round tail budget) close the residual toward the batch-rebuild ceiling?
      // If it plateaus below, the slack is global-structural (rebuild-only); if
      // it climbs, garden just needs more coverage budget.
      for (uint32_t it = 0; opt.garden_pass > 0 && it < opt.garden_pass_iters; ++it) {
        const auto gs = idx.garden_refresh(opt.garden_pass);
        const RecallResult r = evaluate_search(idx, queries, gt, live, opt);
        std::cout << "[garden_pass " << it << "] refreshed=" << gs.refreshed
                  << " p10 " << gs.indeg_p10_before << "->" << gs.indeg_p10_after
                  << " p50 " << gs.indeg_p50_before << "->" << gs.indeg_p50_after
                  << " ms=" << static_cast<double>(gs.elapsed_us) / 1000.0
                  << " masked_recall@10=" << r.recall() << "\n";
      }
      return 0;
    }

    run_warmup_searches(idx, queries, gt, live, opt);

    std::ofstream csv(opt.out_csv);
    csv << "round,deletes,inserts,update_ms,delete_ms,insert_ms,update_qps,"
           "mixed_search_qps,mixed_search_mean_us,mixed_search_queries,"
           "masked_recall_at_10,recall_hits,recall_total,search_mean_us,search_qps,"
           "live_count,tombstones,free_slots,cache_ratio,flush_ms\n";

    // Yi's UpdateRunner runs NO updates in mixed round 0 (search-only baseline
    // eval); replicate by prepending a baseline round and shifting trace files
    // to rounds 1..N.
    const bool round0_baseline = opt.mixed && opt.mixed_round0_baseline;
    const uint32_t total_rounds = rounds + (round0_baseline ? 1U : 0U);
    for (uint32_t round_id = 0; round_id < total_rounds; ++round_id) {
      const bool baseline_round = round0_baseline && round_id == 0;
      TraceRound round;
      if (!baseline_round) {
        const uint32_t trace_idx = round0_baseline ? round_id - 1 : round_id;
        const auto path = trace_dir / (manifest.prefix + std::to_string(trace_idx));
        round = limit_round_updates(read_round(path, manifest.update_size), opt);
      }
      MixedSearchState mixed_state;
      bool mixed_started = false;
      std::unique_ptr<coro::thread_pool> mixed_pool;
      const auto t0 = std::chrono::steady_clock::now();
      try {
        if (opt.mixed && !baseline_round) {
          if (opt.mixed_mode == MixedMode::SharedQueue) {
            const uint32_t mixed_workers = std::max({uint32_t{1},
                                                     opt.search_threads,
                                                     opt.update_insert_threads,
                                                     opt.update_reconnect_threads});
            mixed_pool = std::make_unique<coro::thread_pool>(
                coro::thread_pool::options{.thread_count = mixed_workers,
                                           .on_thread_start_functor = nullptr,
                                           .on_thread_stop_functor = nullptr});
            start_shared_queue_mixed_search(mixed_state, idx, base, opt, *mixed_pool);
          } else {
            start_mixed_search(mixed_state, idx, base, opt);
          }
          mixed_started = true;
        }
        if (!baseline_round) {
          if (opt.mixed && opt.mixed_mode == MixedMode::SharedQueue) {
            apply_deletes_shared(idx, round, opt, live, label_to_slot, *mixed_pool);
          } else {
            apply_deletes(idx, round, opt, live, label_to_slot);
          }
        }
        const auto t_delete = std::chrono::steady_clock::now();
        if (!baseline_round) {
          if (opt.mixed && opt.mixed_mode == MixedMode::SharedQueue) {
            apply_inserts_shared(idx, base, round, opt, live, label_to_slot, *mixed_pool);
          } else {
            apply_inserts(idx, base, round, opt, live, label_to_slot);
          }
        }
        const auto t_insert = std::chrono::steady_clock::now();
        const auto t1 = std::chrono::steady_clock::now();
        MixedSearchResult mixed_search;
        if (mixed_started) {
          mixed_search = finish_mixed_search(mixed_state);
        }
        if (opt.garden_budget > 0 && !baseline_round) {
          coro::thread_pool *garden_pool = mixed_pool ? mixed_pool.get() : nullptr;
          const auto g0 = std::chrono::steady_clock::now();
          auto gs = idx.garden_refresh(opt.garden_budget, garden_pool);
          const auto garden_wall_us =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - g0)
                  .count();
          if (gs.elapsed_us == 0 && garden_wall_us > 0) {
            gs.elapsed_us = static_cast<uint64_t>(garden_wall_us);
          }
          std::cout << "[garden] refreshed=" << gs.refreshed << " selected=" << gs.selected
                    << " p10 " << gs.indeg_p10_before << "->" << gs.indeg_p10_after
                    << " p50 " << gs.indeg_p50_before << "->" << gs.indeg_p50_after
                    << " ms=" << static_cast<double>(gs.elapsed_us) / 1000.0 << "\n";
        }
        if (opt.track_indegree) {
          const auto ip = idx.in_degree_percentiles();
          std::cout << "[indeg] p10=" << ip.p10 << " p50=" << ip.p50
                    << " p90=" << ip.p90 << "\n";
        }
        if (mixed_pool) {
          mixed_pool->shutdown();
        }
        // Persist outside the timed window: Yi's dirty-page write-back is
        // likewise excluded from its update QPS (background writeback_one /
        // round-boundary writeback_remaining). flush_pages() is the light
        // dirty-page write-back; the full checkpoint runs once after all
        // rounds.
        double flush_ms = 0.0;
        if (opt.flush_rounds && !baseline_round) {
          const auto f0 = std::chrono::steady_clock::now();
          idx.flush_pages();
          flush_ms =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - f0)
                  .count();
        }
        const double update_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double delete_ms = std::chrono::duration<double, std::milli>(t_delete - t0).count();
        const double insert_ms =
            std::chrono::duration<double, std::milli>(t_insert - t_delete).count();
        const auto stg = idx.take_update_stage_stats();
        if (stg.inserts > 0) {
          const auto per = [&](uint64_t us) {
            return static_cast<double>(us) / 1000.0 / static_cast<double>(stg.inserts);
          };
          std::cout << "[stage per-insert ms] gate=" << per(stg.gate_us)
                    << " greedy=" << per(stg.greedy_us) << " search=" << per(stg.search_us)
                    << " alloc=" << per(stg.alloc_us) << " prefetch=" << per(stg.prefetch_us)
                    << " write=" << per(stg.write_us) << " reconnect=" << per(stg.reconnect_us)
                    << " | rc(prefetch=" << per(stg.rc_prefetch_us)
                    << " lock=" << per(stg.rc_lock_us) << " impl=" << per(stg.rc_impl_us)
                    << ") reconnects/insert="
                    << static_cast<double>(stg.reconnects) / static_cast<double>(stg.inserts)
                    << " repair_ms=" << static_cast<double>(stg.repair_us) / 1000.0
                    << " repairs=" << stg.repairs
                    << " garden_ms=" << static_cast<double>(stg.garden_us) / 1000.0
                    << " gardens=" << stg.gardens << "\n";
        }
        const size_t round_updates = round.deletes.size() + round.inserts.size();
        const double update_qps =
            round_updates > 0 ? static_cast<double>(round_updates) / (update_ms / 1000.0) : 0.0;
        const RecallResult recall = evaluate_search(idx, queries, gt, live, opt);
        std::cout << "[round " << round_id << "] update_qps=" << update_qps
                  << " delete_ms=" << delete_ms << " insert_ms=" << insert_ms
                  << " flush_ms=" << flush_ms << " mixed_mode=" << mixed_mode_name(opt.mixed_mode)
                  << " mixed_search_qps=" << mixed_search.qps
                  << " mixed_search_mean_us=" << mixed_search.mean_us
                  << " mixed_search_queries=" << mixed_search.queries
                  << " masked_recall@10=" << recall.recall() << " eval_qps=" << recall.qps
                  << " eval_mean_us=" << recall.mean_us << " live=" << idx.live_count()
                  << " tombstones=" << idx.tombstone_count()
                  << (baseline_round ? " (baseline)" : "") << "\n";
        csv << round_id << "," << round.deletes.size() << "," << round.inserts.size() << ","
            << update_ms << "," << delete_ms << "," << insert_ms << "," << update_qps << ","
            << mixed_search.qps << "," << mixed_search.mean_us << "," << mixed_search.queries << ","
            << recall.recall() << "," << recall.hits << "," << recall.total << "," << recall.mean_us
            << "," << recall.qps << "," << idx.live_count() << "," << idx.tombstone_count() << ","
            << idx.free_slot_count() << "," << opt.cache_ratio << "," << flush_ms << "\n";
      } catch (...) {
        if (mixed_started) {
          request_mixed_search_stop(mixed_state);
          join_mixed_search(mixed_state);
        }
        if (mixed_pool) {
          mixed_pool->shutdown();
        }
        throw;
      }
    }
    idx.flush();
    std::cout << "[update_bench] wrote " << opt.out_csv << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[update_bench] ERROR: " << e.what() << "\n";
    return 1;
  }
}
