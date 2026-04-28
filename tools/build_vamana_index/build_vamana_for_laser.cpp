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
// Dispatch logic and defaults live in `build_dispatch.hpp`; this file is a
// thin argv → BuildVamanaParams translator and a top-level try/catch that
// maps exceptions to exit(2).
//

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "index/graph/vamana/build_dispatch.hpp"
#include "utils/log.hpp"

namespace {

using alaya::vamana::BuildVamanaParams;
using alaya::vamana::kDefaultVamanaBuildParams;

// CLI-local state. `data_path_str` / `output_path_str` own the string
// storage; `params.data_path` / `params.output_path` are string_views into
// them and remain valid for the lifetime of this struct.
struct Cli {
  BuildVamanaParams params = kDefaultVamanaBuildParams;
  std::string data_path_str;
  std::string output_path_str;
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
     << "  -R, --max_degree <uint32>      Graph degree bound (default "
     << kDefaultVamanaBuildParams.R << ")\n"
     << "  -L, --lbuild <uint32>          Build-time beam width (default "
     << kDefaultVamanaBuildParams.L << ")\n"
     << "      --alpha <float>            α-RNG pruning parameter (default "
     << kDefaultVamanaBuildParams.alpha << ")\n"
     << "  -T, --num_threads <uint32>     OpenMP thread count (default: omp_get_num_procs())\n"
     << "      --seed <uint64>            RNG seed (default "
     << kDefaultVamanaBuildParams.seed << ")\n"
     << "      --build_dram_budget <GB>   Single-shard budget in GiB (default "
     << kDefaultVamanaBuildParams.build_dram_budget_gb << ")\n"
     << "      --sampling_rate <float>    Partition kmeans sampling rate (default "
     << "auto = min(1.0, 256000 / N))\n"
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

Cli parse_args(int argc, char **argv) {
  Cli c;
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
      c.show_help = true;
    } else if (flag == "--data_path") {
      c.data_path_str = need_value(i);
    } else if (flag == "--index_path_prefix") {
      c.output_path_str = need_value(i);
    } else if (flag == "-R" || flag == "--max_degree") {
      c.params.R = parse_u32(need_value(i), std::string(flag));
    } else if (flag == "-L" || flag == "--lbuild") {
      c.params.L = parse_u32(need_value(i), std::string(flag));
    } else if (flag == "--alpha") {
      c.params.alpha = parse_f32(need_value(i), std::string(flag));
    } else if (flag == "-T" || flag == "--num_threads") {
      c.params.num_threads = parse_u32(need_value(i), std::string(flag));
    } else if (flag == "--seed") {
      c.params.seed = parse_u64(need_value(i), std::string(flag));
    } else if (flag == "--build_dram_budget") {
      c.params.build_dram_budget_gb = parse_f32(need_value(i), std::string(flag));
    } else if (flag == "--sampling_rate") {
      c.params.sampling_rate = parse_f32(need_value(i), std::string(flag));
    } else {
      die(std::string("unknown flag '") + argv[i] + "'");
    }
  }
  return c;
}

}  // namespace

int main(int argc, char **argv) {
  Cli cli = parse_args(argc, argv);
  if (cli.show_help) {
    print_help(std::cout);
    return 0;
  }
  if (cli.data_path_str.empty()) {
    die("missing required flag --data_path");
  }
  if (cli.output_path_str.empty()) {
    die("missing required flag --index_path_prefix");
  }

  // Bind string_views in params to the strings owned by `cli`. `cli` lives
  // for the rest of main, so the views stay valid through build_vamana().
  cli.params.data_path = cli.data_path_str;
  cli.params.output_path = cli.output_path_str;

  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

  try {
    alaya::vamana::build_vamana(cli.params);
  } catch (const std::exception &e) {
    die(e.what());
  }
  return 0;
}
