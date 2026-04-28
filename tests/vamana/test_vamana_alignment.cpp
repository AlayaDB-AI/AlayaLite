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
// test_vamana_alignment — three-tier check that an AlayaLite-produced
// Vamana .index is "close enough" to a DiskANN reference built with the
// same parameters. Gate 1 exit criterion for openspec change
// port-diskann-vamana.
//
// Tier L1 — format parse compatibility:
//   Both files parseable as DiskANN's single-file .index layout.
//
// Tier L2 — structural similarity:
//   avg_degree, in-degree histogram (10 bins), BFS diameter from medoid,
//   orphan count all within --stat_threshold (default 5%) of the DiskANN
//   reference.
//
// Tier L3 — recall@K similarity:
//   Invoke DiskANN's `search_memory_index` on both indices with the same
//   query + gt files and a sweep of L_search values. The max |Δrecall@K|
//   across Ls must not exceed --recall_delta_pp (default 0.5 pp for
//   single-shard, 1.0 pp in --force_partition mode).
//

#include <sys/wait.h>  // WIFEXITED / WEXITSTATUS
#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct VamanaIndex {
  std::filesystem::path path;
  uint64_t file_size = 0;
  uint32_t max_observed_degree = 0;
  uint32_t start = 0;
  uint64_t frozen_pts = 0;
  std::vector<std::vector<uint32_t>> adjacency;

  size_t num_nodes() const { return adjacency.size(); }
};

struct Args {
  std::string dataset;
  std::string alaya_index;
  std::string diskann_index;
  std::string data_path;
  std::string query_path;
  std::string gt_path;
  std::string result_path_prefix;  // scratch location for search_memory_index
  std::string search_memory_index_bin =
      "/home/huangliang/alaya-dev/DiskANN/build/apps/search_memory_index";
  uint32_t R = 64;
  uint32_t L = 100;
  float alpha = 1.2f;
  uint32_t recall_at = 10;
  std::vector<uint32_t> l_search = {10, 50, 100};
  float stat_threshold = 0.05f;
  float recall_delta_pp = 0.5f;
  bool force_partition = false;  // Gate 2 mode: relax recall, check num_parts
  // Strict mode: AlayaLite num_parts MUST equal `expected_num_parts`
  // (set via `--expected_num_parts <N>`).
  uint32_t expected_num_parts = 0;  // 0 = disabled
  // Envelope mode: AlayaLite num_parts MUST satisfy
  // `expected_num_parts_lo <= alaya_parts <= expected_num_parts_hi`
  // (set via `--expected_num_parts_envelope <lo> <hi>`). Tier B uses
  // this when the DiskANN reference is unseeded, so its num_parts is
  // a 3-rerun envelope rather than a fixed value. Both bounds must be
  // > 0 for envelope mode to activate; `--expected_num_parts <N>` is
  // shorthand for `--expected_num_parts_envelope <N> <N>`.
  uint32_t expected_num_parts_lo = 0;
  uint32_t expected_num_parts_hi = 0;
  std::string alaya_shard_work_dir;  // default derived from alaya_index
  bool show_help = false;
};

void print_help(std::ostream &os) {
  os << "test_vamana_alignment — gate AlayaLite vs DiskANN Vamana builds.\n"
     << "\n"
     << "Convenience:\n"
     << "  --dataset <name>                Resolve default paths for a known dataset\n"
     << "                                  (sift1m, gist1m, deep10m, bigann100m).\n"
     << "\n"
     << "Explicit inputs (any of these override --dataset defaults):\n"
     << "  --alaya_index <path>            AlayaLite .index output\n"
     << "  --diskann_index <path>          DiskANN reference .index\n"
     << "  --data_path <path>              Base vectors .fbin (for search_memory_index)\n"
     << "  --query_path <path>             Query vectors .fbin\n"
     << "  --gt_path <path>                Ground truth .ibin\n"
     << "  --result_path_prefix <path>     Scratch prefix for search output (default: /tmp/tvam_)\n"
     << "\n"
     << "Params:\n"
     << "  -R <uint32>                     Build-time degree (default 64)\n"
     << "  -L <uint32>                     Build-time lbuild (default 100) — informational\n"
     << "  --alpha <float>                 Build alpha (default 1.2) — informational\n"
     << "  --recall_at <uint32>            K for recall@K (default 10)\n"
     << "  --l_search <list>               Comma-separated L_search values (default 10,50,100)\n"
     << "  --stat_threshold <ratio>        L2 structural threshold (default 0.05 = 5%)\n"
     << "  --recall_delta_pp <pp>          Max |Δrecall@K| (default 0.5)\n"
     << "  --force_partition               Gate 2 mode: relax recall threshold to 1.0pp min,\n"
     << "                                  enable num_parts consistency check (when\n"
     << "                                  --expected_num_parts is provided).\n"
     << "  --expected_num_parts_envelope <lo> <hi>\n"
     << "                                  Tier B mode: AlayaLite num_parts must lie\n"
     << "                                  in [lo, hi]. Use when DiskANN reference is\n"
     << "                                  unseeded — bounds derived from ≥3 reruns.\n"
     << "  --expected_num_parts <uint32>   DiskANN reference's num_parts value; with\n"
     << "                                  --force_partition, the test counts AlayaLite's\n"
     << "                                  shard graph files under `<alaya_index>_shard_work/`\n"
     << "                                  and fails early if the counts differ.\n"
     << "  --alaya_shard_work_dir <path>   Override default shard-work location\n"
     << "                                  (default: `<alaya_index>_shard_work`)\n"
     << "  --search_memory_index_bin <p>   Override DiskANN search binary path\n"
     << "  -h, --help                      This message\n";
}

[[noreturn]] void die(const std::string &msg) {
  std::cerr << "test_vamana_alignment error: " << msg << "\n"
            << "Run with --help for usage.\n";
  std::exit(2);
}

std::vector<uint32_t> parse_csv_u32(const std::string &s) {
  std::vector<uint32_t> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.empty()) {
      continue;
    }
    out.push_back(static_cast<uint32_t>(std::stoul(tok)));
  }
  return out;
}

// Dataset shorthand mapping for convenience flag. See D10 in design.md:
// data under /md1/.../data/<dataset>/, prefixed by a short label.
std::string dataset_base_prefix(const std::string &dataset) {
  if (dataset == "sift1m") return "sift";
  if (dataset == "gist1m") return "gist";
  if (dataset == "deep10m") return "deep";
  if (dataset == "bigann100m") return "bigann";
  return dataset;  // fallback: use name as-is
}

void resolve_dataset_defaults(Args &a) {
  if (a.dataset.empty()) {
    return;
  }
  const std::string prefix = dataset_base_prefix(a.dataset);
  const std::string data_dir = "/md1/huangliang/alaya-dev/data/" + a.dataset;
  const std::string bg_dir = "/md1/huangliang/alaya-dev/build_graph/" + a.dataset;
  const std::string param_tag =
      "R" + std::to_string(a.R) + "_L" + std::to_string(a.L) + "_a" +
      (a.alpha == static_cast<int>(a.alpha)
           ? std::to_string(static_cast<int>(a.alpha))
           : [&]() {
               std::ostringstream os;
               os.precision(1);
               os << std::fixed << a.alpha;
               return os.str();
             }());
  if (a.data_path.empty()) a.data_path = data_dir + "/" + prefix + "_base.fbin";
  if (a.query_path.empty()) a.query_path = data_dir + "/" + prefix + "_query.fbin";
  if (a.gt_path.empty()) a.gt_path = data_dir + "/" + prefix + "_gt.ibin";
  if (a.alaya_index.empty())
    a.alaya_index = bg_dir + "/alaya/" + param_tag + "/graph.index";
  if (a.diskann_index.empty())
    a.diskann_index = bg_dir + "/diskann_gt/" + param_tag + "/graph.index";
  if (a.result_path_prefix.empty())
    a.result_path_prefix = bg_dir + "/diff/" + param_tag + "/search_";
}

Args parse_args(int argc, char **argv) {
  Args a;
  auto need_value = [&](int &i) {
    if (i + 1 >= argc) {
      die(std::string("flag '") + argv[i] + "' requires a value");
    }
    ++i;
    return std::string(argv[i]);
  };
  for (int i = 1; i < argc; ++i) {
    std::string_view f = argv[i];
    if (f == "-h" || f == "--help")
      a.show_help = true;
    else if (f == "--dataset")
      a.dataset = need_value(i);
    else if (f == "--alaya_index")
      a.alaya_index = need_value(i);
    else if (f == "--diskann_index")
      a.diskann_index = need_value(i);
    else if (f == "--data_path")
      a.data_path = need_value(i);
    else if (f == "--query_path")
      a.query_path = need_value(i);
    else if (f == "--gt_path")
      a.gt_path = need_value(i);
    else if (f == "--result_path_prefix")
      a.result_path_prefix = need_value(i);
    else if (f == "--search_memory_index_bin")
      a.search_memory_index_bin = need_value(i);
    else if (f == "-R")
      a.R = static_cast<uint32_t>(std::stoul(need_value(i)));
    else if (f == "-L")
      a.L = static_cast<uint32_t>(std::stoul(need_value(i)));
    else if (f == "--alpha")
      a.alpha = std::stof(need_value(i));
    else if (f == "--recall_at")
      a.recall_at = static_cast<uint32_t>(std::stoul(need_value(i)));
    else if (f == "--l_search")
      a.l_search = parse_csv_u32(need_value(i));
    else if (f == "--stat_threshold")
      a.stat_threshold = std::stof(need_value(i));
    else if (f == "--recall_delta_pp")
      a.recall_delta_pp = std::stof(need_value(i));
    else if (f == "--force_partition")
      a.force_partition = true;
    else if (f == "--expected_num_parts")
      a.expected_num_parts = static_cast<uint32_t>(std::stoul(need_value(i)));
    else if (f == "--expected_num_parts_envelope") {
      a.expected_num_parts_lo = static_cast<uint32_t>(std::stoul(need_value(i)));
      a.expected_num_parts_hi = static_cast<uint32_t>(std::stoul(need_value(i)));
    }
    else if (f == "--alaya_shard_work_dir")
      a.alaya_shard_work_dir = need_value(i);
    else
      die(std::string("unknown flag '") + argv[i] + "'");
  }
  return a;
}

// ---------------- L1: Parse ----------------
VamanaIndex load_index(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("cannot open " + path.string());
  }
  in.seekg(0, std::ios::end);
  const uint64_t actual_size = static_cast<uint64_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  if (actual_size < 24) {
    throw std::runtime_error(path.string() + ": shorter than 24-byte header");
  }
  VamanaIndex idx;
  idx.path = path;
  in.read(reinterpret_cast<char *>(&idx.file_size), sizeof(uint64_t));
  in.read(reinterpret_cast<char *>(&idx.max_observed_degree), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&idx.start), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&idx.frozen_pts), sizeof(uint64_t));
  if (idx.file_size != actual_size) {
    std::cerr << "warn: " << path << " header expected_file_size=" << idx.file_size
              << " but actual size=" << actual_size << " — continuing anyway\n";
  }
  uint64_t pos = 24;
  while (pos < actual_size) {
    uint32_t k = 0;
    in.read(reinterpret_cast<char *>(&k), sizeof(uint32_t));
    if (!in.good()) {
      throw std::runtime_error(path.string() + ": truncated at node " +
                               std::to_string(idx.adjacency.size()));
    }
    std::vector<uint32_t> nbrs(k);
    if (k > 0) {
      in.read(reinterpret_cast<char *>(nbrs.data()),
              static_cast<std::streamsize>(k) * sizeof(uint32_t));
      if (!in.good()) {
        throw std::runtime_error(path.string() + ": truncated neighbors at node " +
                                 std::to_string(idx.adjacency.size()));
      }
    }
    idx.adjacency.push_back(std::move(nbrs));
    pos += sizeof(uint32_t) * (1 + static_cast<uint64_t>(k));
  }
  return idx;
}

// ---------------- L2: Structural ----------------
struct StructuralStats {
  double avg_degree = 0.0;
  uint32_t orphan_count = 0;
  uint32_t bfs_diameter_from_medoid = 0;
  std::array<uint64_t, 10> in_degree_hist{};
  uint32_t in_degree_max = 0;
};

std::vector<uint32_t> compute_in_degree(const VamanaIndex &idx) {
  std::vector<uint32_t> in_deg(idx.num_nodes(), 0);
  for (const auto &nbrs : idx.adjacency) {
    for (uint32_t m : nbrs) {
      if (m < idx.num_nodes()) {
        in_deg[m]++;
      }
    }
  }
  return in_deg;
}

uint32_t bfs_diameter(const VamanaIndex &idx, uint32_t source) {
  if (idx.num_nodes() == 0) return 0;
  std::vector<int32_t> depth(idx.num_nodes(), -1);
  depth[source] = 0;
  std::queue<uint32_t> q;
  q.push(source);
  int32_t max_depth = 0;
  while (!q.empty()) {
    uint32_t n = q.front();
    q.pop();
    for (uint32_t m : idx.adjacency[n]) {
      if (m < idx.num_nodes() && depth[m] < 0) {
        depth[m] = depth[n] + 1;
        if (depth[m] > max_depth) max_depth = depth[m];
        q.push(m);
      }
    }
  }
  return static_cast<uint32_t>(max_depth);
}

StructuralStats compute_stats(const VamanaIndex &idx) {
  StructuralStats s;
  uint64_t total_out = 0;
  for (const auto &nbrs : idx.adjacency) total_out += nbrs.size();
  s.avg_degree = idx.num_nodes() == 0 ? 0.0
                                      : static_cast<double>(total_out) /
                                            static_cast<double>(idx.num_nodes());
  auto in_deg = compute_in_degree(idx);
  for (uint32_t d : in_deg) {
    if (d == 0) s.orphan_count++;
    if (d > s.in_degree_max) s.in_degree_max = d;
  }
  // 10-bin in-degree histogram over [0, in_degree_max].
  if (s.in_degree_max == 0) {
    s.in_degree_hist[0] = in_deg.size();
  } else {
    const double step = (static_cast<double>(s.in_degree_max) + 1.0) / 10.0;
    for (uint32_t d : in_deg) {
      size_t bin = std::min<size_t>(9, static_cast<size_t>(d / step));
      s.in_degree_hist[bin]++;
    }
  }
  s.bfs_diameter_from_medoid = bfs_diameter(idx, idx.start);
  return s;
}

double rel_diff(double a, double b) {
  const double denom = std::max(std::abs(a), std::abs(b));
  return denom < 1e-12 ? 0.0 : std::abs(a - b) / denom;
}

struct StructuralVerdict {
  bool pass = true;
  double avg_degree_rel = 0.0;
  double bfs_diameter_rel = 0.0;
  double orphan_rel = 0.0;
  double max_hist_bin_rel = 0.0;
};

StructuralVerdict compare_stats(const StructuralStats &a, const StructuralStats &b,
                                double threshold) {
  StructuralVerdict v;
  v.avg_degree_rel = rel_diff(a.avg_degree, b.avg_degree);
  v.bfs_diameter_rel =
      rel_diff(a.bfs_diameter_from_medoid, b.bfs_diameter_from_medoid);
  v.orphan_rel = rel_diff(a.orphan_count, b.orphan_count);
  for (size_t i = 0; i < 10; ++i) {
    v.max_hist_bin_rel =
        std::max(v.max_hist_bin_rel, rel_diff(static_cast<double>(a.in_degree_hist[i]),
                                              static_cast<double>(b.in_degree_hist[i])));
  }
  v.pass = (v.avg_degree_rel <= threshold) && (v.bfs_diameter_rel <= threshold) &&
           (v.orphan_rel <= threshold) && (v.max_hist_bin_rel <= threshold);
  return v;
}

// ---------------- L3: Recall via search_memory_index ----------------
//
// We shell out to DiskANN's existing search binary rather than linking its
// library — this keeps AlayaLite's build system free of a DiskANN dependency
// while still giving a byte-accurate parse test (if DiskANN's loader
// accepts AlayaLite's file, so will the library underneath).

struct SearchResult {
  bool binary_loaded_file = false;  // L1 check by DiskANN's loader
  std::vector<double> recall_at_k;  // recall@K one entry per L_search
};

std::string shell_escape(const std::string &s) {
  if (s.find('\'') == std::string::npos) return "'" + s + "'";
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\' || c == '$' || c == '`') out += '\\';
    out += c;
  }
  out += '"';
  return out;
}

// Run `cmd`, capture stdout. Returns exit code; fills `stdout_out`.
int run_capture(const std::string &cmd, std::string &stdout_out) {
  FILE *pipe = popen(cmd.c_str(), "r");
  if (!pipe) return -1;
  char buf[4096];
  stdout_out.clear();
  while (fgets(buf, sizeof(buf), pipe)) {
    stdout_out += buf;
  }
  int status = pclose(pipe);
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  return -1;
}

// Parse search_memory_index stdout for the Recall@K column.
//
// DiskANN's header row has multi-word column names ("Avg dist cmps",
// "Mean Latency (mus)"), so whitespace tokens in the header don't line
// up with whitespace tokens in the data rows. Recall@K is always the
// last column in the output, so we read the last numeric token of each
// subsequent data row. Data rows are identified by starting with an
// integer (the Ls value the user asked for) and the separator line of
// '=' characters is skipped.
std::vector<double> parse_recall_column(const std::string &output, uint32_t k) {
  std::vector<double> recalls;
  const std::string wanted = "Recall@" + std::to_string(k);
  std::istringstream iss(output);
  std::string line;
  bool header_seen = false;
  while (std::getline(iss, line)) {
    if (!header_seen) {
      if (line.find(wanted) != std::string::npos) {
        header_seen = true;
      }
      continue;
    }
    if (line.find_first_not_of(" =\t\r\n") == std::string::npos) {
      continue;  // blank or separator row
    }
    std::istringstream ls(line);
    std::vector<std::string> toks;
    std::string tok;
    while (ls >> tok) toks.push_back(tok);
    if (toks.size() < 2) continue;
    try {
      (void)std::stoi(toks[0]);
    } catch (...) {
      continue;  // first token not an integer -> not a data row
    }
    try {
      recalls.push_back(std::stod(toks.back()));
    } catch (...) {
      continue;
    }
  }
  return recalls;
}

SearchResult run_search(const Args &args, const std::string &index_path,
                        const std::string &result_prefix, std::string &stdout_out) {
  std::string l_values;
  for (size_t i = 0; i < args.l_search.size(); ++i) {
    if (i) l_values += ' ';
    l_values += std::to_string(args.l_search[i]);
  }
  std::string cmd = shell_escape(args.search_memory_index_bin) +
                    " --data_type float --dist_fn l2"
                    " --index_path_prefix " +
                    shell_escape(index_path) + " --query_file " +
                    shell_escape(args.query_path) + " --gt_file " +
                    shell_escape(args.gt_path) + " --result_path " +
                    shell_escape(result_prefix) + " -K " +
                    std::to_string(args.recall_at) + " -L " + l_values + " 2>&1";
  int rc = run_capture(cmd, stdout_out);
  SearchResult r;
  r.binary_loaded_file = (rc == 0);
  if (!r.binary_loaded_file) return r;
  r.recall_at_k = parse_recall_column(stdout_out, args.recall_at);
  return r;
}

// ---------------- Main driver ----------------
struct Verdict {
  bool l1_pass = false;
  bool l2_pass = false;
  bool l3_pass = false;
  bool num_parts_pass = true;  // vacuously true unless --force_partition + --expected_num_parts
};

int final_exit_code(const Verdict &v) {
  if (!v.num_parts_pass) return 9;  // partition count mismatch → fail earliest
  if (!v.l1_pass) return 10;
  if (!v.l2_pass) return 11;
  if (!v.l3_pass) return 12;
  return 0;
}

// Count AlayaLite shard-graph files `<prefix>_subshard-<i>_mem.index`
// written by build_vamana_index's partition path. The naming convention
// mirrors DiskANN's `build_merged_vamana_index` (`disk_utils.cpp:712`),
// so DiskANN's `search_memory_index` can also consume these files.
// Returns the number of matching files found.
uint32_t count_alaya_shards(const std::filesystem::path &work_dir) {
  if (!std::filesystem::exists(work_dir) || !std::filesystem::is_directory(work_dir)) {
    return 0;
  }
  uint32_t n = 0;
  for (const auto &entry : std::filesystem::directory_iterator(work_dir)) {
    if (!entry.is_regular_file()) continue;
    const std::string name = entry.path().filename().string();
    // The `_mem.index` suffix matches DiskANN's shard graph naming; the
    // `_subshard-` infix is the reliable discriminator against unrelated
    // `.index` files the work dir might accumulate.
    const std::string suffix = "_mem.index";
    if (name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0 &&
        name.find("_subshard-") != std::string::npos) {
      ++n;
    }
  }
  return n;
}

}  // namespace

int main(int argc, char **argv) {
  Args args = parse_args(argc, argv);
  if (args.show_help) {
    print_help(std::cout);
    return 0;
  }
  resolve_dataset_defaults(args);
  if (args.alaya_index.empty() || args.diskann_index.empty() ||
      args.query_path.empty() || args.gt_path.empty()) {
    die("missing required paths (pass --dataset or the explicit flags)");
  }

  const double effective_recall_delta_pp =
      args.force_partition ? std::max(args.recall_delta_pp, 1.0f)
                           : args.recall_delta_pp;

  std::cout << "## test_vamana_alignment\n"
            << "  dataset=" << args.dataset << " R=" << args.R << " L=" << args.L
            << " alpha=" << args.alpha << "\n"
            << "  alaya_index  : " << args.alaya_index << "\n"
            << "  diskann_index: " << args.diskann_index << "\n"
            << "  query        : " << args.query_path << "\n"
            << "  gt           : " << args.gt_path << "\n"
            << "  recall@" << args.recall_at << " threshold: " << effective_recall_delta_pp
            << " pp\n"
            << "  structural threshold: " << (args.stat_threshold * 100.0) << " %\n\n";

  Verdict verdict;

  // Gate 2 pre-check — num_parts consistency. Runs before L1/L2/L3 so that
  // a partition-count mismatch fails the harness with a clear diagnostic
  // rather than surfacing as a downstream recall divergence. Active when
  // `--force_partition` is set together with either `--expected_num_parts`
  // (strict mode) or `--expected_num_parts_envelope <lo> <hi>` (Tier B
  // mode, used with unseeded DiskANN references). Strict supersedes
  // envelope when both are provided so existing harness invocations stay
  // bit-exact.
  uint32_t lo = args.expected_num_parts;
  uint32_t hi = args.expected_num_parts;
  if (args.expected_num_parts == 0 && args.expected_num_parts_lo > 0) {
    lo = args.expected_num_parts_lo;
    hi = args.expected_num_parts_hi;
  }
  if (args.force_partition && lo > 0) {
    std::filesystem::path work_dir = args.alaya_shard_work_dir.empty()
        ? std::filesystem::path(args.alaya_index + "_shard_work")
        : std::filesystem::path(args.alaya_shard_work_dir);
    const uint32_t alaya_parts = count_alaya_shards(work_dir);
    std::cout << "Gate2 num_parts check:\n"
              << "  alaya_shard_work_dir = " << work_dir << "\n"
              << "  alaya num_parts      = " << alaya_parts << "\n"
              << "  expected envelope    = [" << lo << ", " << hi << "]\n";
    if (alaya_parts == 0) {
      std::cerr << "  FAIL: no AlayaLite shard graphs found in " << work_dir
                << " (partition path not run, or build_vamana_index work "
                << "dir was cleaned up?)\n";
      verdict.num_parts_pass = false;
      return final_exit_code(verdict);
    }
    if (alaya_parts < lo || alaya_parts > hi) {
      std::cerr << "  FAIL: num_parts " << alaya_parts
                << " outside expected envelope [" << lo << ", " << hi << "]\n";
      verdict.num_parts_pass = false;
      return final_exit_code(verdict);
    }
    std::cout << "  PASS\n\n";
  }

  // L1 — parse compatibility (local reader as a byte-level sanity check;
  // DiskANN's own loader is exercised later in L3 via search_memory_index).
  VamanaIndex alaya;
  VamanaIndex diskann;
  try {
    alaya = load_index(args.alaya_index);
    diskann = load_index(args.diskann_index);
    verdict.l1_pass = true;
    std::cout << "L1 parse: PASS\n"
              << "  alaya:   N=" << alaya.num_nodes() << " max_deg=" << alaya.max_observed_degree
              << " start=" << alaya.start << "\n"
              << "  diskann: N=" << diskann.num_nodes()
              << " max_deg=" << diskann.max_observed_degree << " start=" << diskann.start
              << "\n\n";
  } catch (const std::exception &e) {
    std::cerr << "L1 parse: FAIL — " << e.what() << "\n";
    return final_exit_code(verdict);
  }

  if (alaya.max_observed_degree != args.R) {
    std::cerr << "L1 warn: alaya max_observed_degree=" << alaya.max_observed_degree
              << " != --R " << args.R
              << " (downstream DiskANN-format readers may assert)\n";
  }
  if (alaya.num_nodes() != diskann.num_nodes()) {
    std::cerr << "L2 fail: alaya N=" << alaya.num_nodes()
              << " != diskann N=" << diskann.num_nodes() << "\n";
    return final_exit_code(verdict);
  }

  // L2 — structural similarity
  auto sa = compute_stats(alaya);
  auto sd = compute_stats(diskann);
  auto v2 = compare_stats(sa, sd, args.stat_threshold);
  verdict.l2_pass = v2.pass;
  std::cout << "L2 structural: " << (v2.pass ? "PASS" : "FAIL") << "\n"
            << "  avg_degree   A=" << sa.avg_degree << " B=" << sd.avg_degree
            << " rel_diff=" << (v2.avg_degree_rel * 100.0) << "%\n"
            << "  BFS diameter A=" << sa.bfs_diameter_from_medoid
            << " B=" << sd.bfs_diameter_from_medoid
            << " rel_diff=" << (v2.bfs_diameter_rel * 100.0) << "%\n"
            << "  orphans      A=" << sa.orphan_count << " B=" << sd.orphan_count
            << " rel_diff=" << (v2.orphan_rel * 100.0) << "%\n"
            << "  in-degree hist max_bin_rel_diff=" << (v2.max_hist_bin_rel * 100.0) << "%\n\n";

  if (!verdict.l2_pass) {
    return final_exit_code(verdict);
  }

  // L3 — recall via DiskANN search
  // Ensure the scratch result directory exists so --result_path can write.
  std::filesystem::create_directories(
      std::filesystem::path(args.result_path_prefix).parent_path());

  std::string out_alaya;
  std::string out_diskann;
  auto r_alaya = run_search(args, args.alaya_index, args.result_path_prefix + "alaya_", out_alaya);
  auto r_diskann =
      run_search(args, args.diskann_index, args.result_path_prefix + "diskann_", out_diskann);

  if (!r_alaya.binary_loaded_file) {
    std::cerr << "L3 fail: DiskANN search_memory_index could not load AlayaLite .index\n"
              << "---- stderr/stdout ----\n"
              << out_alaya << "\n";
    return final_exit_code(verdict);
  }
  if (!r_diskann.binary_loaded_file) {
    std::cerr << "L3 fail: DiskANN search_memory_index could not load DiskANN .index (!)\n"
              << "---- stderr/stdout ----\n"
              << out_diskann << "\n";
    return final_exit_code(verdict);
  }
  if (r_alaya.recall_at_k.size() != r_diskann.recall_at_k.size() ||
      r_alaya.recall_at_k.size() != args.l_search.size()) {
    std::cerr << "L3 fail: parsed " << r_alaya.recall_at_k.size() << " / "
              << r_diskann.recall_at_k.size() << " recall rows for "
              << args.l_search.size() << " L values\n"
              << "---- alaya output ----\n"
              << out_alaya << "\n---- diskann output ----\n"
              << out_diskann << "\n";
    return final_exit_code(verdict);
  }

  double max_abs_delta = 0.0;
  std::cout << "L3 recall@" << args.recall_at << " (pp):\n";
  for (size_t i = 0; i < args.l_search.size(); ++i) {
    const double delta = std::abs(r_alaya.recall_at_k[i] - r_diskann.recall_at_k[i]);
    max_abs_delta = std::max(max_abs_delta, delta);
    std::cout << "  L=" << args.l_search[i] << "   alaya=" << r_alaya.recall_at_k[i]
              << " diskann=" << r_diskann.recall_at_k[i] << " |Δ|=" << delta << "\n";
  }
  verdict.l3_pass = max_abs_delta <= effective_recall_delta_pp;
  std::cout << "  max |Δ| = " << max_abs_delta << " pp  threshold=" << effective_recall_delta_pp
            << " pp  -> " << (verdict.l3_pass ? "PASS" : "FAIL") << "\n\n";

  std::cout << "OVERALL: "
            << (verdict.num_parts_pass && verdict.l1_pass && verdict.l2_pass &&
                        verdict.l3_pass
                    ? "PASS"
                    : "FAIL")
            << "\n";
  return final_exit_code(verdict);
}
