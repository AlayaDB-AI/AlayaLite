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

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
  #include <process.h>
#else
  #include <sys/wait.h>
  #include <unistd.h>
#endif

#include "utils/dataset_utils.hpp"
#include "utils/parser.hpp"

namespace alaya {

/**
 * @brief Configuration for hybrid datasets with vectors, scalar payloads and filters.
 */
struct HybridDatasetConfig {
  std::string name_;
  std::filesystem::path dir_;
  std::filesystem::path data_file_;
  std::filesystem::path scalar_file_;
  std::filesystem::path query_file_;
  std::string download_url_;
  std::string archive_name_ = "data.tar.gz";
  int strip_components_ = 0;
  uint32_t max_data_num_ = 0;   ///< Max vectors/payloads to load (0 = all)
  uint32_t max_query_num_ = 0;  ///< Max filtered queries to load (0 = all)
  MetricType metric_ = MetricType::COS;
  std::vector<std::string> indexed_fields_;
};

/**
 * @brief Create config for Qdrant filtered ANN datasets.
 *
 * The archive is expected to contain vectors.npy, payloads.jsonl and tests.jsonl.
 * Geo filters are intentionally not configured here because MetadataFilter does
 * not currently model geo-radius predicates.
 */
inline auto qdrant_filtered_dataset(const std::filesystem::path &data_dir,
                                    const std::string &name,
                                    const std::string &archive_name,
                                    const std::string &download_url) -> HybridDatasetConfig {
  auto dir = data_dir / name;
  return HybridDatasetConfig{
      .name_ = name,
      .dir_ = dir,
      .data_file_ = dir / "vectors.npy",
      .scalar_file_ = dir / "payloads.jsonl",
      .query_file_ = dir / "tests.jsonl",
      .download_url_ = download_url,
      .archive_name_ = archive_name,
      .metric_ = MetricType::COS,
  };
}

inline auto qdrant_arxiv(const std::filesystem::path &data_dir) -> HybridDatasetConfig {
  return qdrant_filtered_dataset(data_dir,
                                 "qdrant-arxiv",
                                 "arxiv.tar.gz",
                                 "https://storage.googleapis.com/ann-filtered-benchmark/"
                                 "datasets/arxiv.tar.gz");
}

inline auto qdrant_hnm(const std::filesystem::path &data_dir) -> HybridDatasetConfig {
  return qdrant_filtered_dataset(data_dir,
                                 "qdrant-hnm",
                                 "hnm.tgz",
                                 "https://storage.googleapis.com/ann-filtered-benchmark/"
                                 "datasets/hnm.tgz");
}

inline auto qdrant_laion_small_clip(const std::filesystem::path &data_dir) -> HybridDatasetConfig {
  return qdrant_filtered_dataset(data_dir,
                                 "qdrant-laion-small-clip",
                                 "laion-small-clip.tgz",
                                 "https://storage.googleapis.com/ann-filtered-benchmark/"
                                 "datasets/laion-small-clip.tgz");
}

inline auto qdrant_random_keywords_1m(const std::filesystem::path &data_dir)
    -> HybridDatasetConfig {
  return qdrant_filtered_dataset(data_dir,
                                 "qdrant-random-keywords-1m",
                                 "random_keywords_1m.tgz",
                                 "https://storage.googleapis.com/ann-filtered-benchmark/"
                                 "datasets/random_keywords_1m.tgz");
}

inline auto qdrant_random_ints_1m(const std::filesystem::path &data_dir) -> HybridDatasetConfig {
  return qdrant_filtered_dataset(data_dir,
                                 "qdrant-random-ints-1m",
                                 "random_ints_1m.tgz",
                                 "https://storage.googleapis.com/ann-filtered-benchmark/"
                                 "datasets/random_ints_1m.tgz");
}

inline auto qdrant_random_float_1m(const std::filesystem::path &data_dir) -> HybridDatasetConfig {
  return qdrant_filtered_dataset(data_dir,
                                 "qdrant-random-float-1m",
                                 "random_float_1m.tgz",
                                 "https://storage.googleapis.com/ann-filtered-benchmark/"
                                 "datasets/random_float_1m.tgz");
}

inline auto qdrant_random_keywords_100k(const std::filesystem::path &data_dir)
    -> HybridDatasetConfig {
  return qdrant_filtered_dataset(data_dir,
                                 "qdrant-random-keywords-100k",
                                 "random_keywords_100k.tgz",
                                 "https://storage.googleapis.com/ann-filtered-benchmark/"
                                 "datasets/random_keywords_100k.tgz");
}

inline auto qdrant_random_ints_100k(const std::filesystem::path &data_dir) -> HybridDatasetConfig {
  return qdrant_filtered_dataset(data_dir,
                                 "qdrant-random-ints-100k",
                                 "random_ints_100k.tgz",
                                 "https://storage.googleapis.com/ann-filtered-benchmark/"
                                 "datasets/random_ints_100k.tgz");
}

inline auto qdrant_random_float_100k(const std::filesystem::path &data_dir) -> HybridDatasetConfig {
  return qdrant_filtered_dataset(data_dir,
                                 "qdrant-random-float-100k",
                                 "random_float_100k.tgz",
                                 "https://storage.googleapis.com/ann-filtered-benchmark/"
                                 "datasets/random_float_100k.tgz");
}

namespace detail {

inline void add_match_condition(MetadataFilter &filter,
                                const std::string &field,
                                const JsonValue &match_value) {
  if (const auto *match_object = as_object(match_value); match_object != nullptr) {
    if (auto *value = find_any(*match_object, {"value", "text"}); value != nullptr) {
      if (auto metadata_value = json_to_metadata_value(*value); metadata_value.has_value()) {
        filter.add_eq(field, *metadata_value);
      }
      return;
    }
    if (auto *values = find_any(*match_object, {"any", "values"}); values != nullptr) {
      if (const auto *array = as_array(*values); array != nullptr) {
        std::vector<MetadataValue> metadata_values;
        metadata_values.reserve(array->size());
        for (const auto &item : *array) {
          if (auto metadata_value = json_to_metadata_value(item); metadata_value.has_value()) {
            metadata_values.push_back(*metadata_value);
          }
        }
        if (!metadata_values.empty()) {
          filter.add_in(field, std::move(metadata_values));
        }
      }
      return;
    }
  }

  if (auto metadata_value = json_to_metadata_value(match_value); metadata_value.has_value()) {
    filter.add_eq(field, *metadata_value);
  }
}

inline void add_range_conditions(MetadataFilter &filter,
                                 const std::string &field,
                                 const JsonValue &range_value) {
  const auto *range_object = as_object(range_value);
  if (range_object == nullptr) {
    return;
  }
  for (const auto &[op, raw_value] : *range_object) {
    auto metadata_value = json_to_metadata_value(raw_value);
    if (!metadata_value.has_value()) {
      continue;
    }
    if (op == "gt" || op == "$gt") {
      filter.add_gt(field, *metadata_value);
    } else if (op == "gte" || op == "ge" || op == "$gte" || op == "$ge") {
      filter.add_ge(field, *metadata_value);
    } else if (op == "lt" || op == "$lt") {
      filter.add_lt(field, *metadata_value);
    } else if (op == "lte" || op == "le" || op == "$lte" || op == "$le") {
      filter.add_le(field, *metadata_value);
    }
  }
}

inline void add_field_condition(MetadataFilter &filter,
                                const std::string &field,
                                const JsonValue &condition_value) {
  const auto *condition_object = as_object(condition_value);
  if (condition_object == nullptr) {
    add_match_condition(filter, field, condition_value);
    return;
  }

  bool handled = false;
  if (auto it = condition_object->find("match"); it != condition_object->end()) {
    add_match_condition(filter, field, it->second);
    handled = true;
  }
  if (auto it = condition_object->find("range"); it != condition_object->end()) {
    add_range_conditions(filter, field, it->second);
    handled = true;
  }
  if (auto it = condition_object->find("$eq"); it != condition_object->end()) {
    add_match_condition(filter, field, it->second);
    handled = true;
  }
  if (auto it = condition_object->find("$in"); it != condition_object->end()) {
    add_match_condition(filter, field, JsonValue{JsonValue::Object{{"any", it->second}}});
    handled = true;
  }
  if (!handled) {
    add_match_condition(filter, field, condition_value);
  }
}

inline auto parse_qdrant_filter(const JsonValue &value) -> MetadataFilter;

inline void add_filter_array(MetadataFilter &filter, const JsonValue &value) {
  if (const auto *array = as_array(value); array != nullptr) {
    for (const auto &item : *array) {
      filter.add_sub_filter(parse_qdrant_filter(item));
    }
    return;
  }
  filter.add_sub_filter(parse_qdrant_filter(value));
}

inline auto parse_qdrant_filter(const JsonValue &value) -> MetadataFilter {
  MetadataFilter filter;
  const auto *object = as_object(value);
  if (object == nullptr) {
    return filter;
  }

  if (auto key_it = object->find("key"); key_it != object->end()) {
    const auto *field = as_string(key_it->second);
    if (field != nullptr) {
      if (auto it = object->find("match"); it != object->end()) {
        add_match_condition(filter, *field, it->second);
      }
      if (auto it = object->find("range"); it != object->end()) {
        add_range_conditions(filter, *field, it->second);
      }
    }
    return filter;
  }

  bool has_logical = false;
  if (auto it = find_any(*object, {"must", "and", "$and"}); it != nullptr) {
    filter.logic_op = LogicOp::AND;
    add_filter_array(filter, *it);
    has_logical = true;
  }
  if (auto it = find_any(*object, {"should", "or", "$or"}); it != nullptr) {
    MetadataFilter sub_filter;
    sub_filter.logic_op = LogicOp::OR;
    add_filter_array(sub_filter, *it);
    filter.add_sub_filter(std::move(sub_filter));
    has_logical = true;
  }
  if (auto it = find_any(*object, {"must_not", "not", "$not"}); it != nullptr) {
    MetadataFilter sub_filter;
    sub_filter.logic_op = LogicOp::AND;
    add_filter_array(sub_filter, *it);

    MetadataFilter not_filter;
    not_filter.logic_op = LogicOp::NOT;
    not_filter.add_sub_filter(std::move(sub_filter));
    filter.add_sub_filter(std::move(not_filter));
    has_logical = true;
  }
  if (has_logical) {
    return filter;
  }

  for (const auto &[field, condition] : *object) {
    add_field_condition(filter, field, condition);
  }
  return filter;
}

inline auto load_qdrant_payloads(const std::filesystem::path &filepath, uint32_t max_rows)
    -> std::vector<ScalarData> {
  std::ifstream reader(filepath);
  if (!reader.is_open()) {
    throw std::runtime_error("Open payload jsonl file error: " + filepath.string());
  }

  std::vector<ScalarData> scalar_data;
  std::string line;
  uint32_t row_id = 0;
  while (std::getline(reader, line)) {
    if (line.empty()) {
      continue;
    }
    if (max_rows > 0 && row_id >= max_rows) {
      break;
    }
    auto root = parse_json_line(line);
    const auto *object = as_object(root);
    if (object == nullptr) {
      throw std::runtime_error("payload jsonl row is not an object");
    }

    MetadataMap metadata;
    for (const auto &[key, value] : *object) {
      if (auto metadata_value = json_to_metadata_value(value); metadata_value.has_value()) {
        metadata.emplace(key, *metadata_value);
      }
    }
    scalar_data.emplace_back(std::to_string(row_id), "", std::move(metadata));
    ++row_id;
  }
  return scalar_data;
}

inline void load_qdrant_tests(const std::filesystem::path &filepath,
                              uint32_t max_query_num,
                              Dataset &dataset) {
  std::ifstream reader(filepath);
  if (!reader.is_open()) {
    throw std::runtime_error("Open tests jsonl file error: " + filepath.string());
  }

  std::string line;
  while (std::getline(reader, line)) {
    if (line.empty()) {
      continue;
    }
    if (max_query_num > 0 && dataset.query_num_ >= max_query_num) {
      break;
    }

    auto root = parse_json_line(line);
    const auto *object = as_object(root);
    if (object == nullptr) {
      throw std::runtime_error("test jsonl row is not an object");
    }

    auto *query_value = find_any(*object, {"vector", "query", "query_vector"});
    if (query_value == nullptr) {
      throw std::runtime_error("test jsonl row has no query vector");
    }
    const auto *query_array = as_array(*query_value);
    if (query_array == nullptr || query_array->empty()) {
      throw std::runtime_error("test jsonl query vector is not a non-empty array");
    }
    if (dataset.dim_ == 0) {
      dataset.dim_ = static_cast<uint32_t>(query_array->size());
    } else if (dataset.dim_ != query_array->size()) {
      throw std::runtime_error("test jsonl query dimension mismatch");
    }
    for (const auto &value : *query_array) {
      dataset.queries_.push_back(json_to_float(value));
    }

    auto *filter_value = find_any(*object, {"conditions", "filter", "payload_filter"});
    dataset.query_filters_.push_back(filter_value == nullptr ? MetadataFilter::empty()
                                                             : parse_qdrant_filter(*filter_value));

    auto *gt_value = find_any(*object, {"closest_ids", "ground_truth", "neighbors", "nearest_ids"});
    if (gt_value != nullptr) {
      const auto *gt_array = as_array(*gt_value);
      if (gt_array == nullptr) {
        throw std::runtime_error("test jsonl ground truth is not an array");
      }
      if (dataset.gt_dim_ == 0) {
        dataset.gt_dim_ = static_cast<uint32_t>(gt_array->size());
      } else if (dataset.gt_dim_ != gt_array->size()) {
        throw std::runtime_error("test jsonl ground truth dimension mismatch");
      }
      for (const auto &id : *gt_array) {
        dataset.ground_truth_.push_back(json_to_u32(id));
      }
    }
    ++dataset.query_num_;
  }
}

inline void collect_filter_fields(const MetadataFilter &filter,
                                  std::unordered_set<std::string> &fields) {
  for (const auto &condition : filter.conditions) {
    fields.insert(condition.field);
  }
  for (const auto &sub_filter : filter.sub_filters) {
    collect_filter_fields(*sub_filter, fields);
  }
}

inline auto infer_indexed_fields(const std::vector<ScalarData> &scalar_data,
                                 const std::vector<MetadataFilter> &query_filters)
    -> std::vector<std::string> {
  std::unordered_set<std::string> fields;
  for (const auto &filter : query_filters) {
    collect_filter_fields(filter, fields);
  }
  if (fields.empty()) {
    for (const auto &scalar : scalar_data) {
      for (const auto &[field, value] : scalar.metadata) {
        (void)value;
        fields.insert(field);
      }
    }
  }
  std::vector<std::string> result(fields.begin(), fields.end());
  std::ranges::sort(result);
  return result;
}

inline auto metric_score(const float *query, const float *data, uint32_t dim, MetricType metric)
    -> float {
  if (metric == MetricType::L2) {
    return -simd::l2_sqr<float, float>(query, data, dim);
  }

  float dot = 0.0F;
  float query_norm = 0.0F;
  float data_norm = 0.0F;
  for (uint32_t i = 0; i < dim; ++i) {
    dot += query[i] * data[i];
    query_norm += query[i] * query[i];
    data_norm += data[i] * data[i];
  }
  if (metric == MetricType::COS) {
    if (query_norm == 0.0F || data_norm == 0.0F) {
      return -std::numeric_limits<float>::infinity();
    }
    return dot / std::sqrt(query_norm * data_norm);
  }
  return dot;
}

inline auto score_greater(const std::pair<uint32_t, float> &lhs,
                          const std::pair<uint32_t, float> &rhs) -> bool {
  return lhs.second > rhs.second;
}

inline auto find_exact_hybrid_gt(const Dataset &dataset, uint32_t topk) -> std::vector<uint32_t> {
  std::vector<uint32_t> result(static_cast<size_t>(dataset.query_num_) * topk,
                               std::numeric_limits<uint32_t>::max());
  for (uint32_t query_id = 0; query_id < dataset.query_num_; ++query_id) {
    std::vector<std::pair<uint32_t, float>> scores;
    scores.reserve(dataset.data_num_);
    const auto &filter = dataset.query_filters_[query_id];
    const auto *query = dataset.queries_.data() + static_cast<size_t>(query_id) * dataset.dim_;
    for (uint32_t data_id = 0; data_id < dataset.data_num_; ++data_id) {
      if (!filter.evaluate(dataset.scalar_data_[data_id].metadata)) {
        continue;
      }
      const auto *data = dataset.data_.data() + static_cast<size_t>(data_id) * dataset.dim_;
      scores.emplace_back(data_id, metric_score(query, data, dataset.dim_, dataset.metric_));
    }
    auto count = std::min<size_t>(static_cast<size_t>(topk), scores.size());
    if (count > 0) {
      std::ranges::partial_sort(scores.begin(),
                                scores.begin() + count,
                                scores.end(),
                                score_greater);
    }
    for (size_t i = 0; i < count; ++i) {
      result[static_cast<size_t>(query_id) * topk + i] = scores[i].first;
    }
  }
  return result;
}

inline auto run_process(const std::vector<std::string> &args) -> int {
  if (args.empty()) {
    throw std::runtime_error("cannot run empty command");
  }

#if defined(_WIN32)
  std::vector<const char *> argv;
  argv.reserve(args.size() + 1);
  for (const auto &arg : args) {
    argv.push_back(arg.c_str());
  }
  argv.push_back(nullptr);
  return static_cast<int>(_spawnvp(_P_WAIT, args.front().c_str(), argv.data()));
#else
  std::vector<char *> argv;
  argv.reserve(args.size() + 1);
  for (const auto &arg : args) {
    argv.push_back(const_cast<char *>(arg.c_str()));
  }
  argv.push_back(nullptr);

  auto pid = ::fork();
  if (pid < 0) {
    return -1;
  }
  if (pid == 0) {
    ::execvp(args.front().c_str(), argv.data());
    ::_exit(127);
  }

  int status = 0;
  while (::waitpid(pid, &status, 0) < 0) {
    if (errno == EINTR) {
      continue;
    }
    return -1;
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    return 128 + WTERMSIG(status);
  }
  return -1;
#endif
}

inline void ensure_hybrid_dataset_files(const HybridDatasetConfig &config) {
  auto lock_dir = config.dir_.parent_path();
  if (!std::filesystem::exists(lock_dir)) {
    std::filesystem::create_directories(lock_dir);
  }

  auto lock_file = lock_dir / (config.dir_.filename().string() + ".lock");
  FileLock lock(lock_file);

  bool files_exist = std::filesystem::exists(config.data_file_) &&
                     std::filesystem::exists(config.scalar_file_) &&
                     std::filesystem::exists(config.query_file_);
  if (files_exist) {
    return;
  }

  if (config.download_url_.empty()) {
    throw std::runtime_error("Hybrid dataset '" + config.name_ + "' not found at " +
                             config.dir_.string() + " and no download URL provided.");
  }
  if (!std::filesystem::exists(config.dir_)) {
    std::filesystem::create_directories(config.dir_);
  }

  auto archive_path = config.dir_ / config.archive_name_;
  if (config.strip_components_ < 0) {
    throw std::runtime_error("Hybrid dataset strip-components must be non-negative");
  }

  std::vector<std::string> download_args = {"wget",
                                            "-O",
                                            archive_path.string(),
                                            "--",
                                            config.download_url_};
  std::vector<std::string> extract_args = {"tar",
                                           "-zxf",
                                           archive_path.string(),
                                           "--strip-components=" +
                                               std::to_string(config.strip_components_),
                                           "-C",
                                           config.dir_.string()};
  if (run_process(download_args) != 0) {
    throw std::runtime_error("Failed to download hybrid dataset: " + config.download_url_);
  }
  if (run_process(extract_args) != 0) {
    throw std::runtime_error("Failed to extract hybrid dataset archive: " + archive_path.string());
  }
}

}  // namespace detail

/**
 * @brief Load a hybrid dataset with vectors, scalar payloads and filtered queries.
 *
 * Currently supports Qdrant filtered ANN benchmark archives containing:
 *   - vectors.npy: dense float32 matrix
 *   - payloads.jsonl: one flat scalar metadata object per vector
 *   - tests.jsonl: query vector, filter conditions and closest_ids per query
 */
inline auto load_hybrid_dataset(const HybridDatasetConfig &config) -> Dataset {
  detail::ensure_hybrid_dataset_files(config);

  Dataset ds;
  ds.name_ = config.name_;
  ds.metric_ = config.metric_;

  auto vectors = load_npy_float_matrix(config.data_file_, config.max_data_num_);
  ds.data_ = std::move(vectors.data_);
  ds.data_num_ = vectors.rows_;
  ds.dim_ = vectors.cols_;

  ds.scalar_data_ = detail::load_qdrant_payloads(config.scalar_file_, config.max_data_num_);
  if (ds.scalar_data_.size() != ds.data_num_) {
    throw std::runtime_error("Hybrid dataset vector/payload count mismatch for " + config.name_);
  }

  detail::load_qdrant_tests(config.query_file_, config.max_query_num_, ds);
  if (ds.query_num_ == 0) {
    throw std::runtime_error("Hybrid dataset has no queries: " + config.name_);
  }
  if (ds.dim_ != vectors.cols_) {
    throw std::runtime_error("Hybrid dataset data/query dimension mismatch for " + config.name_);
  }
  if (ds.query_filters_.size() != ds.query_num_) {
    throw std::runtime_error("Hybrid dataset query/filter count mismatch for " + config.name_);
  }
  if (!ds.ground_truth_.empty() && ds.ground_truth_.size() != ds.query_num_ * ds.gt_dim_) {
    throw std::runtime_error("Hybrid dataset query/ground-truth count mismatch for " +
                             config.name_);
  }

  ds.indexed_fields_ = config.indexed_fields_.empty()
                           ? detail::infer_indexed_fields(ds.scalar_data_, ds.query_filters_)
                           : config.indexed_fields_;

  if (config.max_data_num_ > 0 && ds.gt_dim_ > 0) {
    ds.ground_truth_ = detail::find_exact_hybrid_gt(ds, ds.gt_dim_);
  }

  return ds;
}

}  // namespace alaya
