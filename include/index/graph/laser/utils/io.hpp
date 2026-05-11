// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file io.hpp
 * @brief File I/O utilities for loading vector datasets.
 *
 * Supports two common vector file formats:
 * - .fvecs/.ivecs: Per-row dimension prefix format (dim + data for each row)
 * - .fbin/.ibin: Header-based format (rows, cols header followed by data)
 */

#pragma once

#include <sys/stat.h>

#include <cassert>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace alaya::laser {

/** @brief Returns file size in bytes, or -1 on error. */
inline size_t get_filesize(const char *filename) {
  struct stat64 stat_buf;
  int tmp = stat64(filename, &stat_buf);
  return tmp == 0 ? stat_buf.st_size : -1;
}

inline bool file_exists(const char *filename) { return std::filesystem::exists(filename); }

/** @brief Loads vectors from .fvecs/.ivecs format (dim prefix per row). */
template <typename T, class M>
void load_vecs(const char *filename, M &row_mat) {
  if (!file_exists(filename)) {
    throw std::runtime_error("load_vecs: file not found: " + std::string(filename));
  }

  assert((std::is_same_v<T *, std::decay_t<decltype(row_mat.data())>> == true));

  uint32_t tmp;
  size_t file_size = get_filesize(filename);
  std::ifstream input(filename, std::ios::binary);

  input.read(reinterpret_cast<char *>(&tmp), sizeof(uint32_t));

  size_t cols = tmp;
  size_t rows = file_size / (cols * sizeof(T) + sizeof(uint32_t));
  row_mat = M(rows, cols);

  input.seekg(0, std::ifstream::beg);

  for (size_t i = 0; i < rows; i++) {
    input.read(reinterpret_cast<char *>(&tmp), sizeof(uint32_t));
    input.read(reinterpret_cast<char *>(&row_mat(i, 0)), sizeof(T) * cols);
  }

  std::cout << "File " << filename << " loaded\n";
  std::cout << "Rows " << rows << " Cols " << cols << '\n' << std::flush;
  input.close();
}

/** @brief Loads vectors from .fbin/.ibin format (rows, cols header). */
template <typename T, class M>
void load_bin(const char *filename, M &row_mat) {
  if (!file_exists(filename)) {
    throw std::runtime_error("load_bin: file not found: " + std::string(filename));
  }

  assert((std::is_same_v<T *, std::decay_t<decltype(row_mat.data())>> == true));

  uint32_t rows;
  uint32_t cols;
  std::ifstream input(filename, std::ios::binary);

  input.read(reinterpret_cast<char *>(&rows), sizeof(uint32_t));
  input.read(reinterpret_cast<char *>(&cols), sizeof(uint32_t));

  row_mat = M(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    input.read(reinterpret_cast<char *>(&row_mat(i, 0)), sizeof(T) * cols);
  }

  std::cout << "File " << filename << " loaded\n";
  std::cout << "Rows " << rows << " Cols " << cols << '\n' << std::flush;
  input.close();
}
}  // namespace alaya::laser
