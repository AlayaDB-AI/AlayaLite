// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/evaluate.hpp"
#include "utils/io_utils.hpp"
#include "utils/locks.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/metric_type.hpp"
#include "utils/scalar_data.hpp"

namespace alaya {

/**
 * @brief Loaded dataset containing vectors, queries and ground truth.
 *
 * Hybrid datasets may also populate scalar metadata and query filters.
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
  std::vector<ScalarData> scalar_data_;
  std::vector<MetadataFilter> query_filters_;
  std::vector<std::string> indexed_fields_;
  uint32_t data_num_ = 0;
  uint32_t query_num_ = 0;
  uint32_t dim_ = 0;
  uint32_t gt_dim_ = 0;
  MetricType metric_ = MetricType::L2;
};

/**
 * @brief Configuration for loading a vector-only dataset.
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
  MetricType metric_ = MetricType::L2;
};

/**
 * @brief Resolve the dataset root for tests and local runs.
 *
 * Priority:
 *   1. `ALAYALITE_DATA_DIR`
 *   2. nearest `data/` found by walking upward from the current directory
 *   3. `cwd/data`
 */
inline auto resolve_data_dir() -> std::filesystem::path {
  if (const char *env = std::getenv("ALAYALITE_DATA_DIR"); env != nullptr && *env != '\0') {
    return env;
  }

  auto current = std::filesystem::current_path();
  std::optional<std::filesystem::path> nearest_existing_data_dir;
  while (true) {
    auto candidate = current / "data";
    if (!nearest_existing_data_dir.has_value() && std::filesystem::exists(candidate)) {
      nearest_existing_data_dir = candidate;
    }

    if (std::filesystem::exists(current / ".git") && std::filesystem::exists(candidate)) {
      return candidate;
    }

    auto parent = current.parent_path();
    if (parent == current) {
      break;
    }
    current = parent;
  }

  if (nearest_existing_data_dir.has_value()) {
    return *nearest_existing_data_dir;
  }

  return std::filesystem::current_path() / "data";
}

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
 * @brief Create config for a tiny SIFT subset used by heavier graph tests.
 */
inline auto sift_tiny(const std::filesystem::path &data_dir) -> DatasetConfig {
  auto dir = data_dir / "siftsmall";
  return DatasetConfig{
      .name_ = "sifttiny",
      .dir_ = dir,
      .data_file_ = dir / "siftsmall_base.fvecs",
      .query_file_ = dir / "siftsmall_query.fvecs",
      .gt_file_ = dir / "siftsmall_groundtruth.ivecs",
      .download_url_ = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
      .max_data_num_ = 200,
      .max_query_num_ = 10,
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
 * @brief Create config for T2I-1M dataset (1M vectors, 200 dim, IP metric).
 * This dataset must be prepared locally; no automatic download is available.
 */
inline auto t2i1m(const std::filesystem::path &data_dir) -> DatasetConfig {
  auto dir = data_dir / "t2i-1m";
  return DatasetConfig{
      .name_ = "t2i-1m",
      .dir_ = dir,
      .data_file_ = dir / "base.fvecs",
      .query_file_ = dir / "query.fvecs",
      .gt_file_ = dir / "groundtruth.ivecs",
      .metric_ = MetricType::IP,
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
  auto lock_dir = config.dir_.parent_path();
  if (!std::filesystem::exists(lock_dir)) {
    std::filesystem::create_directories(lock_dir);
  }

  auto lock_file = lock_dir / (config.dir_.filename().string() + ".lock");
  FileLock lock(lock_file);

  bool files_exist = std::filesystem::exists(config.data_file_) &&
                     std::filesystem::exists(config.query_file_) &&
                     std::filesystem::exists(config.gt_file_);
  if (!files_exist) {
    if (config.download_url_.empty()) {
      throw std::runtime_error("Dataset '" + config.name_ + "' not found at " +
                               config.dir_.string() + " and no download URL provided.");
    }
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
  ds.metric_ = config.metric_;

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

  bool data_truncated = config.max_data_num_ > 0 && ds.data_num_ > config.max_data_num_;
  bool query_truncated = config.max_query_num_ > 0 && ds.query_num_ > config.max_query_num_;

  if (data_truncated) {
    ds.data_num_ = config.max_data_num_;
    ds.data_.resize(ds.data_num_ * ds.dim_);
  }

  if (query_truncated) {
    ds.query_num_ = config.max_query_num_;
    ds.queries_.resize(ds.query_num_ * ds.dim_);
  }

  if (data_truncated) {
    ds.ground_truth_ = find_exact_gt(ds.queries_, ds.data_, ds.dim_, ds.gt_dim_);
  } else if (query_truncated) {
    ds.ground_truth_.resize(ds.query_num_ * ds.gt_dim_);
  }

  return ds;
}

}  // namespace alaya
