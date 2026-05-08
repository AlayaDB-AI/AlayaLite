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

/**
 * @file rotator.hpp
 * @brief Fast Hadamard Transform (FHT) based vector rotation.
 *
 * Applies randomized FHT rotation to vectors before quantization.
 * This rotation normalizes the distribution of vector components,
 * improving the accuracy of binary quantization in RaBitQ.
 *
 * The rotation consists of:
 * 1. Element-wise multiplication with random signs
 * 2. Fast Hadamard Transform (O(d log d) complexity)
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include "index/graph/laser/utils/array.hpp"
#include "index/graph/laser/utils/memory.hpp"
#include "third_party/ffht/fht_avx.hpp"

namespace alaya::laser {

/**
 * @brief Applies randomized Fast Hadamard Transform rotation to vectors.
 *
 * Used during both indexing (to rotate neighbor vectors) and search
 * (to rotate query vectors) with the same random signs for consistency.
 */
class FHTRotator {
  using data_type = data::Array<float, std::vector<size_t>, memory::AlignedAllocator<float>>;

 private:
  std::function<void(float *)> fht_float_ = helper_float_6;
  size_t iter_ = 0;
  size_t remain_ = 0;
  size_t dimension_ = 0;
  size_t paded_dim_ = 0;
  data_type mat_;

 public:
  FHTRotator() = default;

  explicit FHTRotator(size_t dim, uint64_t seed = 0)
      : dimension_(dim), paded_dim_(1 << ceil_log2(dim)), mat_(std::vector<size_t>{1, paded_dim_}) {
    size_t log_b = ceil_log2(dim);

    // seed == 0 preserves upstream Laser's pre-change `std::random_device`
    // path so the single-arg call FHTRotator(dim) stays bit-identical to
    // upstream origin/main. Non-zero seeds pin the Bernoulli draw for the
    // Tier A byte-equality gate (align-laser-with-upstream design.md D2).
    std::uniform_int_distribution<int> bernoulli(0, 1);
    std::mt19937_64 gen =
        (seed == 0) ? std::mt19937_64(std::random_device()()) : std::mt19937_64(seed);
    for (size_t i = 0; i < paded_dim_; ++i) {
      mat_[i] =
          static_cast<float>((2 * bernoulli(gen)) - 1) / std::sqrt(static_cast<float>(paded_dim_));
    }
#if defined(__AVX512F__)
    remain_ = dimension_ & 0b1111;
    iter_ = dimension_ - remain_;
#elif defined(__AVX2__)
    remain_ = dimension_ & 0b111;
    iter_ = dimension_ - remain_;
#else
    remain_ = dimension_ & 0b11;
    iter_ = dimension_ - remain_;
#endif
    switch (log_b) {
      case 6:
        this->fht_float_ = helper_float_6;
        break;
      case 7:
        this->fht_float_ = helper_float_7;
        break;
      case 8:
        this->fht_float_ = helper_float_8;
        break;
      case 9:
        this->fht_float_ = helper_float_9;
        break;
      case 10:
        this->fht_float_ = helper_float_10;
        break;
      case 11:
        this->fht_float_ = helper_float_11;
        break;
      default:
        throw std::invalid_argument(
            "FHTRotator: vector dimension larger than supported FHT tables (max 2^11)");
    }
  }

  ~FHTRotator() = default;

  /**
   * @brief       rotate the scr vector by FHTRotator
   *
   * @param src   raw query vector, length dimension_
   * @param dst   rotated query vector, length B, must be aligned to 64 bytes
   */
  void rotate(const float *__restrict__ src, float *__restrict__ dst) const {
    size_t idx = 0;
#if defined(__AVX512F__)
    for (; idx < iter_; idx += 16) {
      __m512 ss = _mm512_loadu_ps(&src[idx]);
      __m512 mm = _mm512_load_ps(&mat_.at(idx));  // notice alignment requirement
      ss = _mm512_mul_ps(ss, mm);
      _mm512_store_ps(&dst[idx], ss);
    }
#elif defined(__AVX2__)
    for (; idx < iter_; idx += 8) {
      __m256 ss = _mm256_loadu_ps(&src[idx]);
      __m256 mm = _mm256_load_ps(&mat_.at(idx));
      ss = _mm256_mul_ps(ss, mm);
      _mm256_store_ps(&dst[idx], ss);
    }
#else
    for (idx = 0; idx < iter_; ++idx) {
      dst[idx] = src[idx] * mat_.at(idx);
    }
#endif
    for (; idx < dimension_; ++idx) {
      dst[idx] = src[idx] * mat_.at(idx);
    }
    std::fill(dst + dimension_, dst + paded_dim_, 0.0F);
    fht_float_(dst);
  }

  void load(std::ifstream &input) { mat_.load(input); }

  void save(std::ofstream &output) const { mat_.save(output); }

  /**
   * @brief Serialise the sign-scaled `mat_` vector in the Tier A dump format.
   *
   * Layout: `uint64 paded_dim` header immediately followed by
   * `float32[paded_dim]` values (host endianness; x86_64 is little-endian).
   * This is the canonical format compared SHA-256 by the Tier A alignment
   * harness between port and upstream runs. The single-vector shape
   * reflects the single FHT rotation the class applies (`mat_` carries the
   * already-scaled Bernoulli vector, divided by `sqrt(paded_dim_)`).
   */
  void dump_signs(const std::string &path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("FHTRotator::dump_signs: cannot open " + path);
    }
    const uint64_t paded = paded_dim_;
    out.write(reinterpret_cast<const char *>(&paded), sizeof(uint64_t));
    for (size_t i = 0; i < paded_dim_; ++i) {
      const float v = mat_.at(i);
      out.write(reinterpret_cast<const char *>(&v), sizeof(float));
    }
    if (!out) {
      throw std::runtime_error("FHTRotator::dump_signs: write failed on " + path);
    }
  }
};
}  // namespace alaya::laser
