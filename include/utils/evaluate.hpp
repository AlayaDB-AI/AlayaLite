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

#include <sys/types.h>
#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>
#include "space/distance/dist_l2.hpp"
#include "utils/log.hpp"
namespace alaya {

template <typename DataType = float, typename DistanceType = float, typename IDType = uint32_t>
auto find_exact_gt(const std::vector<DataType> &queries,
                   const std::vector<DataType> &data_view,
                   uint32_t dim,
                   uint32_t topk,
                   std::unordered_set<IDType> *deleted = nullptr) -> std::vector<IDType> {
  if (queries.empty() || data_view.empty() || queries.size() % dim != 0 ||
      data_view.size() % dim != 0) {
    LOG_ERROR("The input data to find ground truth is invalid.");
    return {};
  }
  auto query_num = queries.size() / dim;

  std::vector<IDType> res(topk * query_num, 0);
  for (IDType i = 0; i < query_num; i++) {
    std::vector<std::pair<IDType, DistanceType>> dists;
    for (uint32_t j = 0; j < data_view.size() / dim; j++) {
      if (deleted && deleted->find(j) != deleted->end()) {
        continue;
      }
      float dist = l2_sqr<DataType, DistanceType>(queries.data() + (i * dim),
                                                  data_view.data() + (j * dim),
                                                  dim);
      dists.emplace_back(j, dist);
    }
    std::sort(dists.begin(), dists.end(), [](const auto &lhs, const auto &rhs) -> auto {
      return lhs.second < rhs.second;
    });
    for (uint32_t j = 0; j < topk; j++) {
      res[(i * topk) + j] = dists[j].first;
    }
  }
  return res;
}

template <typename IDType>
auto calc_recall(const IDType *res,
                 const IDType *gt,
                 uint32_t query_num,
                 uint32_t gt_dim,
                 uint32_t topk) -> float {
  uint32_t cnt = 0;
  for (uint32_t i = 0; i < query_num; i++) {
    for (uint32_t j = 0; j < topk; j++) {
      for (uint32_t k = 0; k < gt_dim; k++) {
        if (res[i * topk + j] == gt[i * gt_dim + k]) {
          cnt++;
          break;
        }
      }
    }
  }
  return static_cast<float>(cnt) / (query_num * topk);
}

template <typename IDType>
auto calc_recall(const std::vector<std::vector<IDType>> &res,
                 const IDType *gt,
                 uint32_t query_num,
                 uint32_t gt_dim,
                 uint32_t topk) -> float {
  uint32_t cnt = 0;
  for (uint32_t i = 0; i < query_num; i++) {
    for (uint32_t j = 0; j < topk; j++) {
      for (uint32_t k = 0; k < gt_dim; k++) {
        if (res[i][j] == gt[i * gt_dim + k]) {
          cnt++;
          break;
        }
      }
    }
  }
  return static_cast<float>(cnt) / (query_num * topk);
}

template <typename T>
auto horizontal_avg(const std::vector<std::vector<T>> &data) -> std::vector<T> {
  size_t rows = data.size();
  size_t cols = data[0].size();

  std::vector<T> avg(cols, 0);
  for (auto &row : data) {
    for (size_t j = 0; j < cols; ++j) {
      avg[j] += row[j];
    }
  }

  for (size_t j = 0; j < cols; ++j) {
    avg[j] /= rows;
  }

  return avg;
}

}  // namespace alaya
