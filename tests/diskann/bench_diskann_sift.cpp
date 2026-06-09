// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only
//
// Standalone SIFT1M benchmark for alaya::diskann::DiskANNIndex.
//
// Builds (or reuses) a disk index on SIFT1M, then sweeps the search list size L
// at a fixed beam width and reports, per L: recall@1, recall@10, mean / p99
// query latency, and mean disk reads per query (graph traversal vs PQ rerank,
// reported separately). The metrics and the ground-truth file match the
// official DiskANN `search_disk_index` tool so the two can be compared directly.
// Results are written as CSV.
//
// Usage:
//   bench_diskann_sift [data_dir] [index_dir] [out_csv] [flags...]
//   --rebuild         force a fresh index build (otherwise reuse index_dir)
//   --nopq            exact disk greedy search (use_pq=false) instead of PQ+rerank
//   --deterministic   per-expansion (No-PQ) / per-beam (PQ) barrier, batch==sequential
//   --nq N            cap query count (0 => all queries)
//   --io_depth N      No-PQ async pipeline depth (0 => tuned default 32)
//   --threads N       concurrent search workers (reports aggregate QPS)
//   --only_l L        run a single L instead of the full sweep
//   --time_pq         time PQ train+encode at --threads then exit (no search)
//   --pq_n N          cap train-set size for --time_pq
// Defaults: data_dir=./sift1m  index_dir=/tmp/diskann_sift1m_alaya  out_csv=diskann_sift1m_alaya.csv
// data_dir must hold sift_base.fbin / sift_query.fbin / sift_gt.ibin (override via argv[1]).
// The on-disk layout always stores full-precision coords, so --nopq runs on the
// same PQ-built index (PQ codes are simply ignored).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "index/graph/diskann/beam_search.hpp"  // SearchStats
#include "index/graph/diskann/diskann_index.hpp"

namespace {
using alaya::diskann::DiskANNBuildParams;
using alaya::diskann::DiskANNIndex;
using alaya::diskann::DiskANNSearchParams;
using alaya::diskann::SearchStats;

// .fbin / .fvecs sibling format used by the DiskANN tooling:
//   [int32 npts][int32 dim][npts * dim * float32]
struct FloatMatrix {
  std::vector<float> data;
  uint32_t n = 0;
  uint32_t dim = 0;
};

FloatMatrix read_fbin(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("cannot open " + path);
  }
  int32_t n = 0;
  int32_t dim = 0;
  f.read(reinterpret_cast<char *>(&n), 4);
  f.read(reinterpret_cast<char *>(&dim), 4);
  if (n <= 0 || dim <= 0) {
    throw std::runtime_error("bad fbin header: " + path);
  }
  FloatMatrix m;
  m.n = static_cast<uint32_t>(n);
  m.dim = static_cast<uint32_t>(dim);
  m.data.resize(static_cast<size_t>(n) * dim);
  f.read(reinterpret_cast<char *>(m.data.data()),
         static_cast<std::streamsize>(m.data.size() * sizeof(float)));
  if (!f) {
    throw std::runtime_error("short read: " + path);
  }
  return m;
}

// .ibin ground truth: [int32 nq][int32 k][nq * k * uint32 ids].
struct IntMatrix {
  std::vector<uint32_t> data;
  uint32_t n = 0;
  uint32_t dim = 0;
  const uint32_t *row(uint32_t i) const { return data.data() + static_cast<size_t>(i) * dim; }
};

IntMatrix read_ibin(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("cannot open " + path);
  }
  int32_t n = 0;
  int32_t dim = 0;
  f.read(reinterpret_cast<char *>(&n), 4);
  f.read(reinterpret_cast<char *>(&dim), 4);
  if (n <= 0 || dim <= 0) {
    throw std::runtime_error("bad ibin header: " + path);
  }
  IntMatrix m;
  m.n = static_cast<uint32_t>(n);
  m.dim = static_cast<uint32_t>(dim);
  m.data.resize(static_cast<size_t>(n) * dim);
  f.read(reinterpret_cast<char *>(m.data.data()),
         static_cast<std::streamsize>(m.data.size() * sizeof(uint32_t)));
  if (!f) {
    throw std::runtime_error("short read: " + path);
  }
  return m;
}

double percentile(std::vector<double> v, double p) {
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  size_t idx = static_cast<size_t>(std::ceil(p * static_cast<double>(v.size()))) - 1;
  if (idx >= v.size()) {
    idx = v.size() - 1;
  }
  return v[idx];
}
}  // namespace

int main(int argc, char **argv) {
  try {
    std::string data_dir = "./sift1m";  // override via argv[1]; see usage above
    std::string index_dir = "/tmp/diskann_sift1m_alaya";
    std::string out_csv = "diskann_sift1m_alaya.csv";
    bool rebuild = false;
    bool nopq = false;          // --nopq: exact disk greedy search (use_pq=false)
    bool deterministic = false;  // --deterministic: per-expansion/-beam barrier
    uint32_t max_nq = 0;         // --nq N: cap query count (0 => all queries)
    uint32_t io_depth = 0;       // --io_depth N: No-PQ async pipeline depth (0 => 2*beam)
    uint32_t threads = 1;        // --threads N: concurrent search workers (aggregate QPS)
    uint32_t only_l = 0;         // --only_l L: run a single L instead of the full sweep
    bool time_pq = false;        // --time_pq: time PQ train+encode at --threads, then exit
    uint64_t pq_n = 0;           // --pq_n N: cap train-set size for --time_pq (0 => all base)

    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--rebuild") {
        rebuild = true;
      } else if (a == "--nopq") {
        nopq = true;
      } else if (a == "--deterministic") {
        deterministic = true;
      } else if (a == "--nq" && i + 1 < argc) {
        max_nq = static_cast<uint32_t>(std::stoul(argv[++i]));
      } else if (a == "--io_depth" && i + 1 < argc) {
        io_depth = static_cast<uint32_t>(std::stoul(argv[++i]));
      } else if (a == "--threads" && i + 1 < argc) {
        threads = std::max<uint32_t>(1, static_cast<uint32_t>(std::stoul(argv[++i])));
      } else if (a == "--only_l" && i + 1 < argc) {
        only_l = static_cast<uint32_t>(std::stoul(argv[++i]));
      } else if (a == "--time_pq") {
        time_pq = true;
      } else if (a == "--pq_n" && i + 1 < argc) {
        pq_n = static_cast<uint64_t>(std::stoull(argv[++i]));
      } else {
        pos.push_back(a);
      }
    }
    if (pos.size() >= 1) {
      data_dir = pos[0];
    }
    if (pos.size() >= 2) {
      index_dir = pos[1];
    }
    if (pos.size() >= 3) {
      out_csv = pos[2];
    }

    const std::string base_path = data_dir + "/sift_base.fbin";
    const std::string query_path = data_dir + "/sift_query.fbin";
    const std::string gt_path = data_dir + "/sift_gt.ibin";

    std::cout << "[bench] loading SIFT1M from " << data_dir << "\n";
    FloatMatrix base = read_fbin(base_path);
    FloatMatrix query = read_fbin(query_path);
    IntMatrix gt = read_ibin(gt_path);
    std::cout << "[bench] base " << base.n << "x" << base.dim << "  query " << query.n << "x"
              << query.dim << "  gt " << gt.n << "x" << gt.dim << "\n";
    if (base.dim != query.dim) {
      throw std::runtime_error("dim mismatch base/query");
    }
    if (gt.n != query.n) {
      throw std::runtime_error("gt/query count mismatch");
    }

    const uint64_t n = base.n;
    const uint64_t dim = base.dim;

    // --- PQ training timing mode (parallel speedup measurement). ---
    if (time_pq) {
      const uint64_t pn = (pq_n > 0 && pq_n < n) ? pq_n : n;
      std::cout << "[bench] --time_pq: PQ train+encode (n_chunks=32, iters=15) on " << pn
                << " vectors, threads=" << threads << "\n";
      alaya::diskann::PQTable pq;
      auto t0 = std::chrono::steady_clock::now();
      pq.train(base.data.data(), pn, dim, 32, 15, 1234, threads);
      auto t1 = std::chrono::steady_clock::now();
      pq.encode(base.data.data(), pn, threads);
      auto t2 = std::chrono::steady_clock::now();
      const double train_s = std::chrono::duration<double>(t1 - t0).count();
      const double enc_s = std::chrono::duration<double>(t2 - t1).count();
      std::printf("[time_pq] threads=%u  n=%llu  train=%.2fs  encode=%.2fs  total=%.2fs\n", threads,
                  static_cast<unsigned long long>(pn), train_s, enc_s, train_s + enc_s);
      return 0;
    }

    // --- Build (identity labels: returned label == base id == gt id). ---
    namespace fs = std::filesystem;
    if (rebuild && fs::exists(index_dir)) {
      std::cout << "[bench] --rebuild: removing " << index_dir << "\n";
      fs::remove_all(index_dir);
    }
    // Build params held in scope so they can be echoed for reproducibility.
    DiskANNBuildParams bp;
    bp.R = 64;
    bp.L = 100;
    bp.alpha = 1.2f;
    bp.pq_n_chunks = 32;    // 128 / 32 => 4 dims/chunk => 32 bytes/vector PQ
    bp.cache_ratio = 0.01;  // ~10k BFS-cached nodes; match official --num_nodes_to_cache
    bp.num_threads = 96;
    bp.seed = 1234;
    bp.verbose = true;  // print per-phase build wall-times
    if (!fs::exists(index_dir)) {
      std::vector<uint64_t> labels(n);
      for (uint64_t i = 0; i < n; ++i) {
        labels[i] = i;
      }
      std::cout << "[bench] building index (R=" << bp.R << " L=" << bp.L << " alpha=" << bp.alpha
                << " pq_chunks=" << bp.pq_n_chunks << " cache_ratio=" << bp.cache_ratio
                << " threads=" << bp.num_threads << " seed=" << bp.seed << ") -> " << index_dir
                << "\n";
      auto t0 = std::chrono::steady_clock::now();
      DiskANNIndex::build(index_dir, base.data.data(), labels.data(), n, dim, bp);
      auto t1 = std::chrono::steady_clock::now();
      std::cout << "[bench] build done in " << std::chrono::duration<double>(t1 - t0).count()
                << " s\n";
    } else {
      std::cout << "[bench] reusing existing index at " << index_dir
                << " (pass --rebuild to force)\n";
    }

    // --- Load: fixed beam width across the L sweep. ---
    const uint32_t kBeam = 4;
    const uint32_t pool_threads = std::max<uint32_t>(8, threads);
    DiskANNIndex idx;
    idx.load(index_dir,
             {/*num_threads=*/pool_threads, /*beam_width=*/kBeam, /*nopq_io_depth=*/io_depth});
    // Mirror DiskANNIndex's 0 => 32 resolution for display/CSV.
    const uint32_t eff_depth = io_depth == 0 ? 32u : std::max<uint32_t>(2u * kBeam, io_depth);

    const uint32_t kTopK = 10;
    const std::vector<uint32_t> ls =
        only_l > 0 ? std::vector<uint32_t>{only_l} : std::vector<uint32_t>{10, 20, 30, 50, 75, 100, 150, 200};

    const std::string system =
        std::string("alaya_") + (nopq ? "nopq" : "pq") + (deterministic ? "_det" : "_async");
    std::cout << "[bench] mode: " << system << "  (use_pq=" << (!nopq)
              << " deterministic=" << deterministic
              << (nopq ? "  nopq_io_depth=" + std::to_string(eff_depth) : "")
              << "  threads=" << threads << ")\n";

    const uint32_t nq_run = (max_nq > 0 && max_nq < query.n) ? max_nq : query.n;
    if (nq_run != query.n) {
      std::cout << "[bench] limiting to first " << nq_run << " of " << query.n << " queries\n";
    }

    std::ofstream csv(out_csv);
    csv << "system,L,beam,recall_at_1,recall_at_10,mean_lat_us,p99_lat_us,"
           "mean_ios,mean_rerank_reads,mean_total_reads,qps_1thread,io_depth,threads,agg_qps\n";

    std::printf("\n  %3s  %9s  %9s  %9s  %7s  %4s  %9s\n", "L", "recall@10", "mean us", "p99 us",
                "ios", "thr", "aggQPS");
    std::printf("  ---  ---------  ---------  ---------  -------  ----  ---------\n");

    // Per-query outputs are row-major (nq_run * kTopK) so concurrent workers write
    // disjoint slices; latencies are per-query, throughput is wall-clock.
    std::vector<uint64_t> all_l(static_cast<size_t>(nq_run) * kTopK);
    std::vector<float> all_d(static_cast<size_t>(nq_run) * kTopK);
    std::vector<double> lat_us(nq_run, 0.0);

    for (uint32_t l : ls) {
      DiskANNSearchParams sp;
      sp.search_list_size = l;
      sp.use_pq = !nopq;
      sp.deterministic = deterministic;
      // PQ: rerank the whole explored frontier (all L candidates) by exact L2, the
      // apples-to-apples analogue of official DiskANN returning top-K over every
      // node it read. At termination every retset entry is already visited, so
      // this adds compute but ~0 extra disk reads (n_rerank_reads stays ~0).
      // No-PQ already computes exact L2 for every read node, so rerank is moot.
      sp.rerank = !nopq;
      sp.rerank_count = nopq ? 0 : l;

      std::atomic<uint64_t> a_ios{0};
      std::atomic<uint32_t> next{0};

      // Each worker pulls queries off a shared counter (work-stealing for balance),
      // times each search, and writes into that query's output slice.
      auto worker = [&]() {
        SearchStats lst;
        lst.read_order.reserve(4096);
        for (;;) {
          const uint32_t q = next.fetch_add(1, std::memory_order_relaxed);
          if (q >= nq_run) {
            break;
          }
          const float *qv = query.data.data() + static_cast<size_t>(q) * dim;
          lst.n_ios = 0;
          lst.n_cache_hits = 0;
          lst.n_nodes_processed = 0;
          lst.n_rerank_reads = 0;
          lst.read_order.clear();
          auto a = std::chrono::steady_clock::now();
          idx.search(qv, kTopK, all_l.data() + static_cast<size_t>(q) * kTopK,
                     all_d.data() + static_cast<size_t>(q) * kTopK, sp, &lst);
          auto b = std::chrono::steady_clock::now();
          lat_us[q] = std::chrono::duration<double, std::micro>(b - a).count();
          a_ios.fetch_add(lst.n_ios, std::memory_order_relaxed);
        }
      };

      auto wall0 = std::chrono::steady_clock::now();
      if (threads == 1) {
        worker();
      } else {
        std::vector<std::thread> pool;
        pool.reserve(threads);
        for (uint32_t t = 0; t < threads; ++t) {
          pool.emplace_back(worker);
        }
        for (auto &th : pool) {
          th.join();
        }
      }
      auto wall1 = std::chrono::steady_clock::now();
      const double wall_s = std::chrono::duration<double>(wall1 - wall0).count();
      const double agg_qps = static_cast<double>(nq_run) / wall_s;

      double hit1 = 0;
      double hit10 = 0;
      for (uint32_t q = 0; q < nq_run; ++q) {
        const uint64_t *row = all_l.data() + static_cast<size_t>(q) * kTopK;
        const uint32_t *truth = gt.row(q);
        std::unordered_set<uint32_t> t10(truth, truth + kTopK);
        for (uint32_t i = 0; i < kTopK; ++i) {
          if (row[i] != DiskANNIndex::kNoLabel && t10.count(static_cast<uint32_t>(row[i])) != 0) {
            hit10 += 1.0;
          }
        }
        if (row[0] != DiskANNIndex::kNoLabel && static_cast<uint32_t>(row[0]) == truth[0]) {
          hit1 += 1.0;
        }
      }

      const double recall1 = hit1 / nq_run;
      const double recall10 = hit10 / (static_cast<double>(nq_run) * kTopK);
      double total_us = 0;
      for (double x : lat_us) {
        total_us += x;
      }
      const double mean_us = total_us / nq_run;
      const double p99_us = percentile(lat_us, 0.99);
      const double mean_ios = static_cast<double>(a_ios.load()) / nq_run;
      const double qps_1t = 1e6 / mean_us;

      std::printf("  %3u  %9.4f  %9.1f  %9.1f  %7.1f  %4u  %9.0f\n", l, recall10, mean_us, p99_us,
                  mean_ios, threads, agg_qps);
      // CSV: mean_rerank_reads is 0 (PQ reranks from co-located coords; No-PQ has no
      // rerank), and mean_total_reads == mean_ios for the same reason.
      csv << system << "," << l << "," << kBeam << "," << recall1 << "," << recall10 << ","
          << mean_us << "," << p99_us << "," << mean_ios << ",0," << mean_ios << "," << qps_1t
          << "," << eff_depth << "," << threads << "," << agg_qps << "\n";
    }
    csv.close();
    std::cout << "\n[bench] wrote " << out_csv << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[bench] ERROR: " << e.what() << "\n";
    return 1;
  }
}
