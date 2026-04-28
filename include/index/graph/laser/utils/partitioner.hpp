/**
 * @file partitioner.hpp
 * @brief Graph partitioning for disk-layout optimization.
 *
 * Implements LDG (Linear Deterministic Greedy) graph partitioning to group
 * related nodes together on disk pages, minimizing I/O during graph traversal.
 * Nodes that frequently access each other are placed in the same partition.
 */

#pragma once

#include <omp.h>
#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <ostream>
#include <queue>
#include <random>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "freq_relayout.hpp"

#ifndef INF
  #define INF 0xffffffff
#endif  // INF
#ifndef READ_U64
  #define READ_U64(stream, val) stream.read(reinterpret_cast<char *>(&val), sizeof(_u64))
#endif  // !READ_U64
#ifndef ROUND_UP
  #define ROUND_UP(X, Y) \
    (((static_cast<uint64_t>(X) / (Y)) + (static_cast<uint64_t>(X) % (Y) != 0)) * (Y))
#endif  // !ROUND_UP
#ifndef SECTOR_LEN
  #define SECTOR_LEN (_u64)4096
#endif  // !SECTOR_LEN

template <typename T>
class ConcurrentBoundedQueue {
 public:
  void clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    std::queue<T> empty_queue;
    std::swap(queue_, empty_queue);
  }
  void push(const T &value) {
    std::unique_lock<std::mutex> lock(mtx_);
    queue_.push(value);
    cond_empty_.notify_one();
  }
  void pop(T &value) {
    std::unique_lock<std::mutex> lock(mtx_);
    cond_empty_.wait(lock, [this]() {
      return !queue_.empty();
    });
    value = queue_.front();
    queue_.pop();
  }

 private:
  std::queue<T> queue_;
  std::mutex mtx_;
  std::condition_variable cond_empty_;
};

namespace gp {

using concurrent_queue = ConcurrentBoundedQueue<unsigned>;
namespace fs = std::filesystem;
using _u64 = uint64_t;
using _u32 = unsigned int;
using VecT = uint8_t;

inline size_t get_file_size(const std::string &fname) {
  std::ifstream reader(fname, std::ios::binary | std::ios::ate);
  if (!reader.fail() && reader.is_open()) {
    size_t end_pos = reader.tellg();
    reader.close();
    return end_pos;
  }
  std::cout << "Could not open file: " << fname << std::endl;
  return 0;
}

// DiskANN changes how the meta is stored in the first sector of
// the _disk.index file after commit id 8bb74ff637cb2a77c99b71368ade68c62b7ca8e0
// (exclusive) It returns <is_new_version, vector of metas uint64_ts>
inline std::pair<bool, std::vector<_u64>> get_disk_index_meta(const std::string &path) {
  std::ifstream fin(path, std::ios::binary);

  int meta_n;
  int meta_dim;
  const int expected_new_meta_n = 9;
  const int expected_new_meta_n_with_reorder_data = 12;
  const int old_meta_n = 11;
  bool is_new_version = true;
  std::vector<_u64> metas;

  fin.read(reinterpret_cast<char *>(&meta_n), sizeof(int));
  fin.read(reinterpret_cast<char *>(&meta_dim), sizeof(int));

  if (meta_n == expected_new_meta_n || meta_n == expected_new_meta_n_with_reorder_data) {
    metas.resize(meta_n);
    fin.read(reinterpret_cast<char *>(metas.data()),
             static_cast<std::streamsize>(sizeof(_u64) * meta_n));
  } else {
    is_new_version = false;
    metas.resize(old_meta_n);
    fin.seekg(0, std::ios::beg);
    fin.read(reinterpret_cast<char *>(metas.data()), sizeof(_u64) * old_meta_n);
  }
  fin.close();
  return {is_new_version, metas};
}

class GraphPartitioner {
 public:
  GraphPartitioner(const char *indexName,
                   uint64_t num_nodes,
                   size_t dim,
                   uint64_t max_node_len,
                   uint64_t node_per_page,
                   unsigned cut,
                   const char *data_type = "float",
                   bool visual = false,
                   const std::string & /*freq_file*/ = std::string(""))
      : dim_(dim), nd_(num_nodes), max_node_len_(max_node_len), C_(node_per_page) {
    visual_ = visual;
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // check file size
    size_t actual_size = get_file_size(indexName);
    size_t expected_size;
    auto meta_pair = get_disk_index_meta(indexName);
    if (meta_pair.first) {
      expected_size = meta_pair.second.back();
    } else {
      expected_size = meta_pair.second.front();
    }
    if (actual_size != expected_size) {
      throw std::runtime_error(
          "Partitioner: index file size mismatch vs. expected header metadata");
    }

    rd_ = new std::random_device();
    gen_ = new std::mt19937((*rd_)());
    dis_ = new std::uniform_real_distribution<>(0, 1);
    load_vamana(indexName);
    cursize_ = nd_ / 1000;

    // copy to direct_graph
    direct_graph_.clear();
    direct_graph_.resize(full_graph_.size());
#pragma omp parallel for
    for (unsigned i = 0; i < nd_; i++) {
      direct_graph_[i].assign(full_graph_[i].begin(), full_graph_[i].end());
    }
    // cut graph
    if (cut != INF) {
      std::cout << "direct graph will be cut, it degree become " << cut << std::endl;
    }
#pragma omp parallel for
    for (unsigned i = 0; i < nd_; i++) {
      if (cut < direct_graph_[i].size()) {
        direct_graph_[i].resize(cut);
      }
    }
    // reverse graph
    std::vector<std::mutex> ms(nd_);
    reverse_graph_.resize(nd_);
#pragma omp parallel for shared(reverse_graph_, direct_graph_)
    for (unsigned i = 0; i < nd_; i++) {
      for (unsigned j = 0; j < direct_graph_[i].size(); j++) {
        std::lock_guard<std::mutex> lock(ms[direct_graph_[i][j]]);
        reverse_graph_[direct_graph_[i][j]].emplace_back(i);
      }
    }
    std::cout << "reverse graph done." << std::endl;
    for (unsigned i = 0; i < partition_number_; i++) {
      pmutex_.push_back(std::make_unique<std::mutex>());
    }
  }

  void cout_step() {
    if (!visual_) {
      return;
    }

#pragma omp atomic
    cur_++;
    if ((cur_ + 0) % cursize_ == 0) {
      std::cout << static_cast<double>(cur_ + 0) / static_cast<double>(nd_) * 100 << "%    \r";
      std::cout.flush();
    }
  }
  /**
   * load vamana graph index from disk
   * @param filename
   */
  void load_vamana(const char *filename) {
    std::cout << "Reading index file: " << filename << "... " << std::flush;
    std::ifstream in;
    in.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
      in.open(filename, std::ios::binary);
      size_t expected_file_size;
      size_t file_frozen_pts;
      in.read(reinterpret_cast<char *>(&expected_file_size), sizeof(size_t));
      in.read(reinterpret_cast<char *>(&width_), sizeof(uint32_t));
      in.read(reinterpret_cast<char *>(&ep_), sizeof(uint32_t));
      in.read(reinterpret_cast<char *>(&file_frozen_pts), sizeof(size_t));
      std::cout << "Loading vamana index " << filename << "..." << std::flush;

      size_t vamana_metadata_size =
          sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);
      size_t bytes_read = vamana_metadata_size;

      size_t cc = 0;
      unsigned nodes = 0;
      while (bytes_read != expected_file_size) {
        unsigned k;
        in.read(reinterpret_cast<char *>(&k), sizeof(unsigned));
        cc += k;
        ++nodes;
        std::vector<unsigned> tmp(k);
        in.read(reinterpret_cast<char *>(tmp.data()),
                static_cast<std::streamsize>(k * sizeof(unsigned)));
        full_graph_.emplace_back(tmp);
        bytes_read += sizeof(uint32_t) * (static_cast<uint32_t>(k) + 1);
        if (nodes % 10000000 == 0) std::cout << "." << std::flush;
      }
      assert(nd_ == full_graph_.size());
      partition_number_ = ROUND_UP(nd_, C_) / C_;
      reverse_graph_.resize(nd_);
      std::vector<std::mutex> ms(nd_);
#pragma omp parallel for shared(reverse_graph_, full_graph_)
      for (unsigned i = 0; i < nd_; i++) {
        for (unsigned j = 0; j < full_graph_[i].size(); j++) {
          std::lock_guard<std::mutex> lock(ms[full_graph_[i][j]]);
          reverse_graph_[full_graph_[i][j]].emplace_back(i);
        }
      }
      std::cout << "done. Index has " << nodes << " nodes and " << cc << " out-edges" << std::endl;
      for (unsigned i = 0; i < partition_number_; i++) {
        pmutex_.push_back(std::make_unique<std::mutex>());
      }
    } catch (const std::system_error &e) {
      throw std::runtime_error(
          std::string("Partitioner: system_error while loading vamana graph: ") + e.what());
    }
  }

  /**
   * save the partition result
   * @tparam T
   * @param filename
   * @param partition
   */
  void save_partition(const char *filename) {
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    std::cout << "writing bin: " << filename << std::endl;
    writer.write(reinterpret_cast<char *>(&C_), sizeof(_u64));
    writer.write(reinterpret_cast<char *>(&partition_number_), sizeof(_u64));
    writer.write(reinterpret_cast<char *>(&nd_), sizeof(_u64));
    std::cout << "_partition_num: " << partition_number_ << " C: " << C_ << " _nd: " << nd_
              << std::endl;
    for (unsigned i = 0; i < partition_number_; i++) {
      auto p = partition_[i];
      unsigned s = p.size();
      writer.write(reinterpret_cast<char *>(&s), sizeof(unsigned));
      writer.write(reinterpret_cast<char *>(p.data()),
                   static_cast<std::streamsize>(sizeof(unsigned) * s));
    }
    std::vector<unsigned> id2pidv(nd_);
    for (auto n : id2pid_) {
      id2pidv[n.first] = n.second;
    }
    writer.write(reinterpret_cast<char *>(id2pidv.data()),
                 static_cast<std::streamsize>(sizeof(unsigned) * nd_));
  }

  /**
   * load partition from disk
   * @param filename
   */
  void load_partition(const char *filename) {
    std::ifstream reader(filename, std::ios::binary);
    reader.read(reinterpret_cast<char *>(&C_), sizeof(_u64));
    reader.read(reinterpret_cast<char *>(&partition_number_), sizeof(_u64));
    reader.read(reinterpret_cast<char *>(&nd_), sizeof(_u64));
    std::cout << "load partition _partition_num: " << partition_number_ << ", C: " << C_
              << std::endl;
    partition_.clear();
    auto *tmp = new unsigned[C_];
    for (unsigned i = 0; i < partition_number_; i++) {
      unsigned c;
      reader.read(reinterpret_cast<char *>(&c), sizeof(unsigned));
      reader.read(reinterpret_cast<char *>(tmp),
                  static_cast<std::streamsize>(c * sizeof(unsigned)));
      std::vector<unsigned> tt;
      tt.reserve(C_);
      for (unsigned j = 0; j < c; j++) {
        tt.push_back(*(tmp + j));
      }
      partition_.push_back(tt);
    }
    delete[] tmp;
    re_id2pid();
  }
  void re_id2pid() {
    id2pid_.clear();
    for (unsigned i = 0; i < partition_number_; i++) {
      for (unsigned j = 0; j < partition_[i].size(); j++) {
        id2pid_[partition_[i][j]] = i;
      }
    }
  }
  /**
   * count the id overlap according to the graph partitioning
   */
  void partition_statistic() {
    std::vector<unsigned> overlap(nd_, 0);
    std::vector<unsigned> blk_neighbor_overlap(partition_number_, 0);
    double overlap_ratio = 0;

#pragma omp parallel for schedule(dynamic, 100) reduction(+ : overlap_ratio)
    for (size_t i = 0; i < partition_number_; i++) {
      std::unordered_set<unsigned> neighbors;
      unsigned blk_neighbor_num = 0;
      for (size_t j = 0; j < partition_[i].size(); j++) {
        blk_neighbor_num += full_graph_[partition_[i][j]].size();
        std::unordered_set<unsigned> ne;
        for (unsigned &x : full_graph_[partition_[i][j]]) {
          neighbors.insert(x);
          ne.insert(x);
        }
        blk_neighbor_overlap[i] = blk_neighbor_num - neighbors.size();
        for (size_t z = 0; z < partition_[i].size(); z++) {
          if (partition_[i][j] == partition_[i][z]) continue;
          if (ne.find(partition_[i][z]) != ne.end()) {
            overlap[partition_[i][j]]++;
          }
        }
        overlap_ratio +=
            (partition_[i].size() == 1 ? 0
                                       : (1.0 * overlap[partition_[i][j]] /
                                          static_cast<double>(partition_[i].size() - 1)));
      }
    }
    unsigned max_overlaps = 0;
    unsigned min_overlaps = std::numeric_limits<unsigned>::max();
    double ave_overlap_ratio = 0;
    std::map<unsigned, unsigned> overlap_count;
    for (size_t i = 0; i < nd_; i++) {
      if (overlap_count.count(overlap[i]) != 0U) {
        overlap_count[overlap[i]]++;
      } else {
        overlap_count[overlap[i]] = 1;
      }
      if (overlap[i] > max_overlaps) max_overlaps = overlap[i];
      if (overlap[i] < min_overlaps) min_overlaps = overlap[i];
    }
    ave_overlap_ratio = overlap_ratio / static_cast<double>(nd_);
    for (auto &it : overlap_count) {
      std::cout << "each id, overlap number " << it.first << ", count: " << it.second << std::endl;
    }
    std::cout << "each id, max overlaps: " << max_overlaps << std::endl;
    std::cout << "each id, min overlaps: " << min_overlaps << std::endl;
    std::cout << "each id, average overlap ratio: " << ave_overlap_ratio << std::endl;
  }

  unsigned select_partition(unsigned i) {
#pragma omp atomic
    select_nums_++;

    float maxn = 0.0;
    unsigned res = INF;
    std::unordered_map<unsigned, unsigned> pcount;
    unsigned tpid = 0;
    for (auto n : direct_graph_[i]) {
      unsigned pid = id2pid_[n];
      if (pid == INF) continue;
      pcount[pid] = pcount[pid] + 1;
      if (tpid < pid) {
        tpid = pid;
      }
    }
    for (auto n : reverse_graph_[i]) {
      unsigned pid = id2pid_[n];
      if (pid == INF) continue;
      pcount[pid] = pcount[pid] + 1;
      if (tpid < pid) {
        tpid = pid;
      }
    }
    for (auto c : pcount) {
      unsigned pid = c.first;
      auto cnt = static_cast<float>(c.second);
      std::lock_guard<std::mutex> lock(*pmutex_[pid]);
      auto s = static_cast<double>(partition_[pid].size());
      cnt *= (1 - static_cast<float>(s) / static_cast<float>(C_));
      if (cnt > maxn && partition_[pid].size() < C_) {
        res = pid;
        maxn = cnt;
      }
    }
    pcount.clear();
    if (res == INF) {
#pragma omp atomic
      select_free_++;
      res = get_unfilled();
    }
    return res;
  }

  unsigned get_unfilled() {
#pragma omp atomic
    getUnfilled_nums_++;
    unsigned res;
    do {
      free_q_.pop(res);
    } while (partition_[res].size() == C_);
    return res;
  }

  // graph partition
  void graph_partition(const char *filename, int k, int lock_nums = 0) {
    for (unsigned i = 0; i < nd_; i++) {
      id2pid_[i] = INF;
    }
    partition_.clear();
    partition_.resize(partition_number_);
    std::unordered_set<unsigned> vis;
    std::vector<unsigned> init_stream;
    init_stream.reserve(nd_);
    if (!freq_list_.empty()) {
      for (auto p : freq_list_) {
        init_stream.emplace_back(p.first);
      }
    } else {
      init_stream.resize(nd_);
      std::iota(init_stream.begin(), init_stream.end(), 0);
    }
    lock_nodes_.clear();
    lock_pids_.clear();
    lock_nodes_.resize(nd_, false);
    lock_pids_.resize(partition_number_, false);
    unsigned pid = 0;
    vis.clear();
    if (lock_nums != 0) {
      std::cout << "lock first " << lock_nums << " nodes at init stage." << std::endl;
    }
    for (auto i : init_stream) {
      if (vis.count(i) != 0U) {
        lock_nums--;
        continue;  // has insert into partition
      }
      if (partition_[pid].size() == C_) {
        ++pid;
      }
      vis.insert(i);
      partition_[pid].push_back(i);
      id2pid_[i] = pid;
      if (lock_nums > 0) {
        lock_pids_[pid] = true;
      }
      for (unsigned s : full_graph_[i]) {
        if (vis.count(s) != 0U) continue;
        if (partition_[pid].size() == C_) {
          ++pid;
          break;
        }
        partition_[pid].push_back(s);
        id2pid_[s] = pid;
        vis.insert(s);
      }
      if (lock_nums != 0) --lock_nums;
    }
    int s = 0;
    for (unsigned i = 0; i < partition_number_; i++) {
      if (!lock_pids_[i]) break;
      for (unsigned s : partition_[i]) {
        lock_nodes_[s] = true;
      }
      s++;
    }
    if (lock_pids_[0]) {
      std::cout << "finally, it locks partition nums: " << s << " locks nodes num: " << s * C_
                << std::endl;
    }

    std::cout << "init over." << std::endl;

    for (int i = 0; i < k; i++) {
      select_free_ = 0;
      graph_partition_LDG();
      std::cout << "select free: "
                << static_cast<double>(select_free_) / static_cast<double>(partition_number_)
                << std::endl;
      partition_statistic();
      auto ivf_file_name = std::string(filename) + std::string(".ivf") + std::to_string(i + 1);
      std::cout << "total ivf time: " << ivf_time_ << std::endl;
      save_partition(ivf_file_name.c_str());
    }
    save_partition(filename);
    std::cout << "select pid nums" << select_nums_
              << " get unfilled partition nums: " << getUnfilled_nums_ << std::endl;
    std::cout << "total ivf time: " << ivf_time_ << std::endl;
  }
  void graph_partition_LDG() {
    free_q_.clear();
#pragma omp parallel for
    for (unsigned i = 0; i < partition_number_; i++) {
      if (lock_pids_[i]) continue;
      partition_[i].clear();
      free_q_.push(i);
    }

    cur_ = 0;
    std::cout << "start" << std::endl;
    std::vector<unsigned> stream(nd_);
    std::iota(stream.begin(), stream.end(), 0);
    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(stream), std::end(stream), rng);
    auto start = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
    for (unsigned i = 0; i < nd_; i++) {
      size_t n = stream[i];
      if (lock_nodes_[n]) continue;
      sync(n);
      cout_step();
    }
    auto end = omp_get_wtime();
    std::cout << "ivf time: " << end - start << " round: " << round_ << std::endl;
    ivf_time_ += end - start;
    round_++;
  }
  unsigned sync(unsigned i) {
    unsigned pid = select_partition(i);
    pmutex_[pid]->lock();

    while (partition_[pid].size() == C_) {
      pmutex_[pid]->unlock();
      pid = select_partition(i);
      pmutex_[pid]->lock();
    }
    partition_[pid].emplace_back(i);
    id2pid_[i] = pid;
    unsigned s = partition_[pid].size();
    pmutex_[pid]->unlock();

    if (s != C_) {
      free_q_.push(pid);
    }

    return pid;
  }

  void copy_layout(std::vector<unsigned> &id2page, std::vector<std::vector<unsigned>> &gp_layout) {
    id2page.resize(nd_);
    for (unsigned i = 0; i < partition_number_; i++) {
      for (unsigned j = 0; j < partition_[i].size(); j++) {
        id2page[partition_[i][j]] = i;
      }
    }
    gp_layout.resize(partition_number_);
    for (unsigned i = 0; i < partition_number_; i++) {
      gp_layout[i].assign(partition_[i].begin(), partition_[i].end());
    }
  }

 private:
  size_t dim_;  // vector dimension
  _u64 nd_;     // vector number
  _u64 max_node_len_;
  unsigned width_;                                   // max out-degree
  unsigned ep_;                                      // seed vertex id
  std::vector<std::vector<unsigned>> direct_graph_;  // neighbor list
  std::vector<std::vector<unsigned>> full_graph_;
  unsigned select_free_;
  _u64 C_;                                                 // partition size threshold
  _u64 partition_number_ = 0;                              // the number of partitions
  std::vector<std::vector<unsigned>> partition_{1000000};  // each partition set
  std::vector<std::unique_ptr<std::mutex>> pmutex_;
  int cur_ = 0;
  std::vector<std::vector<unsigned>> reverse_graph_;
  std::vector<std::vector<unsigned>> undirect_graph_;
  std::unordered_map<unsigned, unsigned> id2pid_;
  std::unordered_map<unsigned, unsigned> id2ratio_;
  int round_ = 0;
  double ivf_time_ = 0.0;
  bool visual_ = false;
  unsigned cursize_ = 10000;
  uint64_t select_nums_ = 0;
  uint64_t getUnfilled_nums_ = 0;
  _u64 E_;
  std::uniform_real_distribution<> *dis_;
  std::mt19937 *gen_;
  std::random_device *rd_;
  concurrent_queue free_q_;

  std::vector<puu> freq_list_;
  vpu freq_nei_list_;
  std::vector<bool> lock_nodes_;
  std::vector<bool> lock_pids_;
};
}  // namespace gp
