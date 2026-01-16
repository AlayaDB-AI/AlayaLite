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

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include "log.hpp"
namespace alaya {

/**
 * @brief Load the vector data from fvecs file.
 *
 * @param filepath The filename.
 * @param data     The data position that will be loaded to.
 * @param num      The number of data.
 * @param dim      The dimension of vector.
 * File format: [num] [dim] [vector1] [vector2] ...
 */
template <typename T>
inline void load_vecs(const std::filesystem::path &filepath,
                      std::vector<T> &data,
                      uint32_t &num,
                      uint32_t &dim) {
  std::ifstream reader(filepath, std::ios::binary);

  if (!reader.is_open()) {
    LOG_CRITICAL("Open fvecs file error {}.", filepath.string());
    exit(-1);
  }

  reader.read(reinterpret_cast<char *>(&num), 4);
  reader.read(reinterpret_cast<char *>(&dim), 4);
  data.reserve(num * dim);

  LOG_INFO("Read {} , data number = {} , data dimension = {}.", filepath.string(), num, dim);

  for (size_t i = 0; i < num; i++) {
    reader.read(reinterpret_cast<char *>(data.data() + (i * dim)), dim * sizeof(T));
  }
  reader.close();
}

/**
 * @brief Save the vector data to ivecs file, usually used for groudture file.
 *
 * @param filename The filename.
 * @param data     The data position.
 * @param num      The number of data.
 * @param dim      The dimension of vector.
 * File format: [num] [dim] [vector1] [vector2] ...
 */
template <typename T>
inline void save_ivecs(const std::filesystem::path &filepath, T *data, uint32_t num, uint32_t dim) {
  std::ofstream writer(filepath, std::ios::binary);
  if (!writer.is_open()) {
    LOG_CRITICAL("Open fvecs file error for writing . {}", filepath.string());
    exit(-1);
  }

  writer.write(reinterpret_cast<char *>(&num), 4);

  for (unsigned i = 0; i < num; ++i) {
    writer.write(reinterpret_cast<char *>(&dim), 4);
    writer.write(reinterpret_cast<char *>(data + (i * dim)), dim * sizeof(T));
  }

  writer.close();
}

/**
 * @brief Load the ground truth ids from the file
 *
 * @param filename The filename.
 * @param data     The data position that will be loaded to.
 * @param num      The number of query.
 * @param gt_topk  Each query has gt_topk ground truth ids.
 * File format: [num] [gt_topk] [id1] [id2] ...
 */
template <typename T>
inline void load_gt(const std::filesystem::path &filepath,
                    std::vector<T> &data,
                    uint32_t &num,
                    uint32_t &gt_topk) {
  std::ifstream reader(filepath, std::ios::binary);
  if (!reader.is_open()) {
    LOG_CRITICAL("Open ivecs file error {}.", filepath.string());
    exit(-1);
  }

  reader.read(reinterpret_cast<char *>(&num), 4);
  reader.read(reinterpret_cast<char *>(&gt_topk), 4);

  data.reserve(num * gt_topk);

  for (size_t i = 0; i < num; i++) {
    reader.read(reinterpret_cast<char *>(data.data() + (i * gt_topk)), gt_topk * sizeof(T));
  }
  reader.close();
}

/**
 * @brief Load the vector data from fvecs file.
 *
 * @param filepath The filename.
 * @param data  The data position that will be loaded to.
 * @param num The number of data.
 * @param dim The dimension of each vector.
 *
 */
template <typename T>
inline void load_fvecs(const std::filesystem::path &filepath,
                       std::vector<T> &data,
                       uint32_t &num,
                       uint32_t &dim) {
  std::ifstream reader(filepath, std::ios::binary);

  if (!reader.is_open()) {
    LOG_CRITICAL("Open fvecs file error {}.", filepath.string());
    exit(-1);
  }

  num = 0;
  data.clear();

  while (!reader.eof()) {
    reader.read(reinterpret_cast<char *>(&dim), 4);
    if (reader.eof()) {
      break;
    }
    if (dim == 0) {
      LOG_CRITICAL("file {} is empty.", filepath.string());
      exit(-1);
    }
    std::vector<T> vec(dim);
    reader.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(T));
    if (reader.gcount() != dim * static_cast<int>(sizeof(T))) {
      LOG_CRITICAL("file {} is not valid.", filepath.string());
      exit(-1);
    }
    data.insert(data.end(), vec.begin(), vec.end());
    num++;
  }

  reader.close();
}

/**
 * @brief Load the vector data from bvecs file.
 *
 * @param filepath The filename.
 * @param data     The data position that will be loaded to.
 * @param num      The number of data.
 * @param dim      The dimension of each vector.
 *
 */
template <typename T>
inline void load_bvecs(const std::filesystem::path &filepath,
                       std::vector<T> &data,
                       uint32_t &num,
                       uint32_t &dim) {
  std::ifstream reader(filepath, std::ios::binary);

  if (!reader.is_open()) {
    LOG_CRITICAL("Open fvecs file error {}.", filepath.string());
    exit(-1);
  }

  // get byte size of file
  reader.read(reinterpret_cast<char *>(&dim), 4);
  reader.seekg(0, std::ios::end);
  size_t total_file_size = reader.tellg();

  reader.seekg(0, std::ios::beg);
  num = total_file_size / (4 + dim * sizeof(T));

  data.reserve(num * dim);

  LOG_INFO("Read {} , data number = {} , data dimension = {}.", filepath.string(), num, dim);

  for (size_t i = 0; i < num; i++) {
    reader.seekg(4, std::ios::cur);
    for (int j = 0; j < dim; j++) {
      reader.read(reinterpret_cast<char *>(data.data() + (i * dim + j)), sizeof(T));
    }
  }
  reader.close();
}

/**
 * @brief Load the vector data from ivecs file.
 *
 * @param filepath The filename.
 * @param data     The data position that will be loaded to.
 * @param num      The number of data.
 * @param dim      The dimension of each vector.
 *
 */
template <typename T>
inline void load_ivecs(const std::filesystem::path &filepath,
                       std::vector<T> &data,
                       uint32_t &num,
                       uint32_t &dim) {
  std::ifstream file(filepath, std::ios::binary);

  if (!file.is_open()) {
    LOG_CRITICAL("Open fvecs file error {}.", filepath.string());
    exit(-1);
  }

  file.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
  if (file.fail()) {
    std::cerr << "Failed to read dimension from file: " << filepath.string() << '\n';
  }

  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  // num = file_size / (sizeof(uint32_t) + dim * sizeof(float));
  num = file_size / (sizeof(uint32_t) + dim * sizeof(T));
  data.resize(num * dim);

  for (uint32_t i = 0; i < num; ++i) {
    file.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
    // file.read(reinterpret_cast<char *>(data.data() + i * dim), dim * sizeof(float));
    file.read(reinterpret_cast<char *>(data.data() + (i * dim)), dim * sizeof(T));
    if (file.fail()) {
      throw std::runtime_error("Failed to read data from file: " + filepath.string());
    }
  }
}

}  // namespace alaya
