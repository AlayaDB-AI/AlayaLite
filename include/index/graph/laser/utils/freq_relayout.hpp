// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <vector>
namespace gp {
using puu = std::pair<unsigned, unsigned>;
using vpu = std::vector<std::vector<puu>>;
using vvu = std::vector<std::vector<unsigned>>;
inline void read_freq(std::vector<puu> &freq_list,
                      vpu &freq_nei_list,
                      const std::string &freq_file) {
  std::ifstream reader(freq_file, std::ios::binary | std::ios::out);
  std::cout << "read visited neighbors information: " << freq_file << std::endl;
  unsigned num = 0;
  reader.read(reinterpret_cast<char *>(&num), 4);
  freq_list.clear();
  freq_list.reserve(num);
  freq_nei_list.clear();
  freq_nei_list.resize(num);
  unsigned n_size = 0;
  for (size_t i = 0; i < num; i++) {
    unsigned v_freq = 0;
    reader.read(reinterpret_cast<char *>(&v_freq), sizeof(unsigned));
    freq_list.emplace_back(i, v_freq);
  }
  std::sort(freq_list.begin(), freq_list.end(), [](puu &left, puu &right) -> bool {
    return left.second > right.second;
  });
  for (size_t i = 0; i < num; i++) {
    reader.read(reinterpret_cast<char *>(&n_size), sizeof(unsigned));
    freq_nei_list[i].reserve(n_size);
    for (size_t j = 0; j < n_size; j++) {
      unsigned nbr_id;
      unsigned visit_nbr_id_freq;
      reader.read(reinterpret_cast<char *>(&nbr_id), sizeof(unsigned));
      reader.read(reinterpret_cast<char *>(&visit_nbr_id_freq), sizeof(unsigned));
      freq_nei_list[i].emplace_back(nbr_id, visit_nbr_id_freq);
    }
  }
}

void relayout_adj(vpu &freq_nei_list, vvu &full_graph) {
  std::vector<unsigned> tmp_adj(100);
  std::unordered_set<unsigned> vis;
  const int64_t graph_size_signed = static_cast<int64_t>(full_graph.size());
#pragma omp parallel for schedule(dynamic, 1000) private(tmp_adj, vis)
  for (int64_t ii = 0; ii < graph_size_signed; ii++) {
    const unsigned i = static_cast<unsigned>(ii);
    std::sort(freq_nei_list[i].begin(), freq_nei_list[i].end(), [](puu &left, puu &right) -> bool {
      return left.second > right.second;
    });
    tmp_adj.clear();
    vis.clear();
    for (auto v : freq_nei_list[i]) {
      tmp_adj.emplace_back(v.first);
      vis.insert(v.first);
    }
    for (auto v : full_graph[i]) {
      if (vis.count(v) != 0U) continue;
      vis.insert(v);
      tmp_adj.emplace_back(v);
    }
    if (full_graph[i].size() != tmp_adj.size()) {
      throw std::runtime_error(
          "freq_relayout: freq file was generated from a different graph (adjacency size "
          "mismatch)");
    }
    full_graph[i].swap(tmp_adj);
  }
}
}  // namespace gp
