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

#include <filesystem>
#include <string>
#include "utils/io_utils.hpp"
namespace alaya {
class TestDatasetBase {
 public:
  TestDatasetBase() = default;

  // Virtual destructor for proper cleanup of derived classes
  virtual ~TestDatasetBase() = default;

  // Check if dataset exists and is complete
  bool ensure_dataset() {
    bool files_exist = std::filesystem::exists(dataset_dir_) &&
                       std::filesystem::exists(data_file_) &&
                       std::filesystem::exists(query_file_) && std::filesystem::exists(gt_file_);

    if (!files_exist) {
      if (!std::filesystem::exists(dataset_dir_)) {
        std::filesystem::create_directories(dataset_dir_);
      }
      [[maybe_unused]] int ret1 = std::system(get_download_command().c_str());
      [[maybe_unused]] int ret2 = std::system(get_extract_command().c_str());
    }

    uint32_t data_dim;
    uint32_t query_dim;
    alaya::load_fvecs(data_file_, data_, data_num_, data_dim);
    alaya::load_fvecs(query_file_, queries_, query_num_, query_dim);
    alaya::load_ivecs(gt_file_, answers_, ans_num_, ans_dim_);
    if (data_dim != query_dim || query_num_ != ans_num_) {
      LOG_CRITICAL(
          "The dimension of data, query and ground truth is not the same. data_dim: {}, query_dim: "
          "{}, query_num: {}, ans_num: {}",
          data_dim,
          query_dim,
          query_num_,
          ans_num_);
      exit(-1);
    }
    dim_ = data_dim;

    return files_exist;
  }

  const std::filesystem::path &get_data_file() const noexcept { return data_file_; }
  const std::filesystem::path &get_query_file() const noexcept { return query_file_; }
  const std::filesystem::path &get_gt_file() const noexcept { return gt_file_; }
  const std::filesystem::path &get_dataset_dir() const noexcept { return dataset_dir_; }
  const std::string &get_dataset_name() const noexcept { return dataset_name_; }

  std::vector<float> &get_data() noexcept { return data_; }
  std::vector<float> &get_queries() noexcept { return queries_; }
  std::vector<uint32_t> &get_answers() noexcept { return answers_; }
  uint32_t get_data_num() const noexcept { return data_num_; }
  uint32_t get_query_num() const noexcept { return query_num_; }
  uint32_t get_ans_num() const noexcept { return ans_num_; }
  uint32_t get_dim() const noexcept { return dim_; }
  uint32_t get_ans_dim() const noexcept { return ans_dim_; }

 protected:
  virtual std::string get_download_command() const = 0;
  virtual std::string get_extract_command() const = 0;

 protected:
  std::string dataset_name_;

  std::filesystem::path dataset_dir_;
  std::filesystem::path data_file_;
  std::filesystem::path query_file_;
  std::filesystem::path gt_file_;

  std::vector<float> data_;
  std::vector<float> queries_;
  std::vector<uint32_t> answers_;
  uint32_t data_num_;
  uint32_t query_num_;
  uint32_t ans_num_;
  uint32_t dim_;
  uint32_t ans_dim_;
};

class SIFTTestData : public TestDatasetBase {
 public:
  SIFTTestData(std::string data_dir) {
    dataset_name_ = "siftsmall";
    dataset_dir_ = std::filesystem::path(data_dir) / "siftsmall";
    data_file_ = dataset_dir_ / "siftsmall_base.fvecs";
    query_file_ = dataset_dir_ / "siftsmall_query.fvecs";
    gt_file_ = dataset_dir_ / "siftsmall_groundtruth.ivecs";
  }

 private:
  std::string get_download_command() const override {
    return "wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz -O " +
           dataset_dir_.string() + "/data.tar.gz";
  }

  std::string get_extract_command() const override {
    return "tar -zxvf " + dataset_dir_.string() + "/data.tar.gz" + " --strip-components=1 -C " +
           dataset_dir_.string();
  }
};

}  // namespace alaya
