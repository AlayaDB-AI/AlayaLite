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
#include <fstream>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace alaya {
/**
 * @brief Scalar Quantization with 8-bit precision.
 *
 * This class implements a simple 8-bit scalar quantizer that maps input data
 * to an 8-bit representation based on the min/max values observed in the dataset.
 * The quantization is performed per dimension.
 *
 * @tparam DataType The data type of input values (e.g., float, double).
 */
template <typename DataType>
struct SQ8Quantizer {
  uint32_t dim_;                      ///< Dimensionality of input data
  std::vector<DataType> min_vector_;  ///< Per-dimension minimum values
  std::vector<DataType> max_vector_;  ///< Per-dimension maximum values

  SQ8Quantizer() = default;
  ~SQ8Quantizer() = default;

  SQ8Quantizer(const SQ8Quantizer &other)
      : dim_(other.dim_), min_vector_(other.min_vector_), max_vector_(other.max_vector_) {}

  SQ8Quantizer(SQ8Quantizer &&other) noexcept
      : dim_(other.dim_),
        min_vector_(std::move(other.min_vector_)),
        max_vector_(std::move(other.max_vector_)) {
    other.dim_ = 0;
  }

  auto operator=(const SQ8Quantizer &other) -> SQ8Quantizer & {
    if (this != &other) {
      dim_ = other.dim_;
      min_vector_ = other.min_vector_;
      max_vector_ = other.max_vector_;
    }
    return *this;
  }

  auto operator=(SQ8Quantizer &&other) noexcept -> SQ8Quantizer & {
    if (this != &other) {
      dim_ = other.dim_;
      min_vector_ = std::move(other.min_vector_);
      max_vector_ = std::move(other.max_vector_);
      other.dim_ = 0;
    }
    return *this;
  }

  /**
   * @brief Constructor initializing quantizer for a given dimension.
   * @param dim The dimensionality of the input data.
   */
  explicit SQ8Quantizer(const uint32_t &dim) : dim_(dim) {
    auto get_min_max_value = []() -> auto {
      if constexpr (std::is_integral_v<DataType>) {
        return std::pair<DataType, DataType>{std::numeric_limits<DataType>::min(),
                                             std::numeric_limits<DataType>::max()};
      }
      return std::pair<DataType, DataType>{std::numeric_limits<DataType>::lowest(),
                                           std::numeric_limits<DataType>::max()};
    };

    auto [min_value, max_value] = get_min_max_value();

    min_vector_ = std::vector<DataType>(dim, max_value);
    max_vector_ = std::vector<DataType>(dim, min_value);
  }

  /**
   * @brief Fit the quantizer by updating min/max vectors based on input data.
   * @param data Pointer to the input data array.
   * @param item_cnt Number of data items in the input array.
   */
  void fit(const DataType *data, size_t item_cnt) {
    for (size_t vector_idx = 0; vector_idx < item_cnt; vector_idx++) {
      for (uint32_t dim_idx = 0U; dim_idx < dim_; dim_idx++) {
        auto value = *(data + vector_idx * dim_ + dim_idx);
        if (value < min_vector_[dim_idx]) {
          min_vector_[dim_idx] = value;
        }
        if (value > max_vector_[dim_idx]) {
          max_vector_[dim_idx] = value;
        }
      }
    }
  }

  /**
   * @brief Quantize a single value to 8-bit representation within [min, max] range.
   * @param value Input value to be quantized.
   * @param min Minimum value of the range.
   * @param max Maximum value of the range.
   * @return 8-bit quantized value (0-255).
   */
  auto quantize(DataType value, DataType min, DataType max) const -> uint8_t {
    if (max == min) {
      return 0x00;
    }
    if (value >= max) {
      return 0xFF;
    }
    if (value <= min) {
      return 0x00;
    }
    auto scaled = (static_cast<float>(value) - min) / (max - min);
    return static_cast<uint8_t>(scaled * 255);
  }

  /**
   * @brief Encode a vector into 8-bit quantized representation.
   * @param raw_data Pointer to input raw data array.
   * @param encoded_data Reference to output encoded data pointer.
   */
  void encode(const DataType *raw_data, uint8_t *const encoded_data) const {
    for (uint32_t i = 0; i < dim_; i++) {
      encoded_data[i] = quantize(raw_data[i], min_vector_[i], max_vector_[i]);
    }
  }

  /**
   * @brief Get the minimum values of each dimension.
   * @return DataType* Pointer to the min values.
   */
  auto get_min() const -> const DataType * { return min_vector_.data(); }

  /**
   * @brief Get the maximum values of each dimension.
   * @return DataType* Pointer to the max values.
   */
  auto get_max() const -> const DataType * { return max_vector_.data(); }

  /**
   * @brief Load the quantizer parameters from a binary file.
   * @param reader Input file stream.
   */
  auto load(std::ifstream &reader) -> void {
    reader.read(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    min_vector_.resize(dim_);
    max_vector_.resize(dim_);
    reader.read(reinterpret_cast<char *>(min_vector_.data()), dim_ * sizeof(DataType));
    reader.read(reinterpret_cast<char *>(max_vector_.data()), dim_ * sizeof(DataType));
  }

  /**
   * @brief Save the quantizer parameters to a binary file.
   * @param writer Output file stream.
   */
  auto save(std::ofstream &writer) const -> void {
    writer.write(reinterpret_cast<const char *>(&dim_), sizeof(dim_));
    writer.write(reinterpret_cast<const char *>(min_vector_.data()), dim_ * sizeof(DataType));
    writer.write(reinterpret_cast<const char *>(max_vector_.data()), dim_ * sizeof(DataType));
  }
};
}  // namespace alaya
