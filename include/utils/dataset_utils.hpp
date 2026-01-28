/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <filesystem>  // NOLINT(build/c++17)
#include <string>
#include "utils/evaluate.hpp"
#include "utils/io_utils.hpp"
#include "utils/locks.hpp"

namespace alaya {

/**
 * @brief Loaded dataset containing vectors, queries and ground truth.
 *
 * Usage:
 *   auto ds = load_dataset(sift_small("/path/to/data"));
 *   space->fit(ds.data_.data(), ds.data_num_);
 */
struct Dataset {
  std::string name_;
  std::vector<float> data_;
  std::vector<float> queries_;
  std::vector<uint32_t> ground_truth_;
  uint32_t data_num_ = 0;
  uint32_t query_num_ = 0;
  uint32_t dim_ = 0;
  uint32_t gt_dim_ = 0;
};

/**
 * @brief Configuration for loading a dataset.
 */
struct DatasetConfig {
  std::string name_;
  std::filesystem::path dir_;
  std::filesystem::path data_file_;
  std::filesystem::path query_file_;
  std::filesystem::path gt_file_;
  std::string download_url_;
  std::string archive_name_ = "data.tar.gz";
  int strip_components_ = 1;
  uint32_t max_data_num_ = 0;   ///< Max vectors to load (0 = all)
  uint32_t max_query_num_ = 0;  ///< Max queries to load (0 = all)
};

/**
 * @brief Create config for SIFT small dataset (10K vectors, 128 dim).
 */
inline auto sift_small(const std::filesystem::path &data_dir) -> DatasetConfig {
  auto dir = data_dir / "siftsmall";
  return DatasetConfig{
      .name_ = "siftsmall",
      .dir_ = dir,
      .data_file_ = dir / "siftsmall_base.fvecs",
      .query_file_ = dir / "siftsmall_query.fvecs",
      .gt_file_ = dir / "siftsmall_groundtruth.ivecs",
      .download_url_ = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
  };
}

/**
 * @brief Create config for SIFT micro dataset (subset of siftsmall: 1K vectors, 128 dim).
 *
 * This is a smaller subset for fast CI testing. Uses the same files as siftsmall
 * but limits the number of vectors loaded.
 */
inline auto sift_micro(const std::filesystem::path &data_dir) -> DatasetConfig {
  auto dir = data_dir / "siftsmall";
  return DatasetConfig{
      .name_ = "siftmicro",
      .dir_ = dir,
      .data_file_ = dir / "siftsmall_base.fvecs",
      .query_file_ = dir / "siftsmall_query.fvecs",
      .gt_file_ = dir / "siftsmall_groundtruth.ivecs",
      .download_url_ = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
      .max_data_num_ = 1000,
      .max_query_num_ = 50,
  };
}

/**
 * @brief Create config for DEEP1M dataset (1M vectors, 96 dim).
 */
inline auto deep1m(const std::filesystem::path &data_dir) -> DatasetConfig {
  auto dir = data_dir / "deep1M";
  return DatasetConfig{
      .name_ = "deep1M",
      .dir_ = dir,
      .data_file_ = dir / "deep1M_base.fvecs",
      .query_file_ = dir / "deep1M_query.fvecs",
      .gt_file_ = dir / "deep1M_groundtruth.ivecs",
      .download_url_ = "http://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/deep1M.tar.gz",
      .archive_name_ = "deep1M.tar.gz",
  };
}

/**
 * @brief Load dataset from config. Downloads if needed.
 *
 * Uses file locking to prevent concurrent downloads when multiple tests run in parallel.
 *
 * Usage:
 *   auto ds = load_dataset(sift_small("/data"));
 *   // Use ds.data_, ds.queries_, ds.ground_truth_ directly
 */
inline auto load_dataset(const DatasetConfig &config) -> Dataset {
  // Use file lock to prevent concurrent downloads
  auto lock_file = config.dir_.parent_path() / (config.name_ + ".lock");
  FileLock lock(lock_file);

  // Download if files don't exist (check again after acquiring lock)
  bool files_exist = std::filesystem::exists(config.data_file_) &&
                     std::filesystem::exists(config.query_file_) &&
                     std::filesystem::exists(config.gt_file_);
  if (!files_exist) {
    if (!std::filesystem::exists(config.dir_)) {
      std::filesystem::create_directories(config.dir_);
    }
    auto archive_path = config.dir_ / config.archive_name_;
    auto download_cmd = "wget " + config.download_url_ + " -O " + archive_path.string();
    auto extract_cmd = "tar -zxvf " + archive_path.string() +
                       " --strip-components=" + std::to_string(config.strip_components_) + " -C " +
                       config.dir_.string();
    [[maybe_unused]] int ret1 = std::system(download_cmd.c_str());
    [[maybe_unused]] int ret2 = std::system(extract_cmd.c_str());
  }

  Dataset ds;
  ds.name_ = config.name_;

  uint32_t data_dim = 0;
  uint32_t query_dim = 0;
  load_fvecs(config.data_file_, ds.data_, ds.data_num_, data_dim);
  load_fvecs(config.query_file_, ds.queries_, ds.query_num_, query_dim);
  load_ivecs(config.gt_file_, ds.ground_truth_, ds.query_num_, ds.gt_dim_);

  if (data_dim != query_dim) {
    LOG_CRITICAL("Dimension mismatch: data_dim={}, query_dim={}", data_dim, query_dim);
    exit(-1);
  }
  ds.dim_ = data_dim;

  // Check if we need to truncate data
  bool data_truncated = config.max_data_num_ > 0 && ds.data_num_ > config.max_data_num_;
  bool query_truncated = config.max_query_num_ > 0 && ds.query_num_ > config.max_query_num_;

  // Apply data limit
  if (data_truncated) {
    ds.data_num_ = config.max_data_num_;
    ds.data_.resize(ds.data_num_ * ds.dim_);
  }

  // Apply query limit
  if (query_truncated) {
    ds.query_num_ = config.max_query_num_;
    ds.queries_.resize(ds.query_num_ * ds.dim_);
  }

  // Recompute ground truth if data was truncated (original GT IDs may be invalid)
  if (data_truncated) {
    ds.ground_truth_ = find_exact_gt(ds.queries_, ds.data_, ds.dim_, ds.gt_dim_);
  } else if (query_truncated) {
    // Only query truncated, just resize GT
    ds.ground_truth_.resize(ds.query_num_ * ds.gt_dim_);
  }

  return ds;
}

}  // namespace alaya
