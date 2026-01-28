/*
 * Copyright 2025 VectorDB.NTU
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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "./defines.hpp"
#include "./fastscan.hpp"

namespace alaya {

template <typename T>
inline void data_range(const T *ALAYA_RESTRICT vec0, size_t dim, T &lo, T &hi) {
  ConstRowMajorArrayMap<T> v0(vec0, 1, dim);
  lo = v0.minCoeff();
  hi = v0.maxCoeff();
}

template <typename T>
inline void scalar_quantize_normal(
    uint8_t *ALAYA_RESTRICT result,
    const T *ALAYA_RESTRICT vec0,
    size_t dim,
    T lo,
    T delta) {  // normal implementation when AVX512F is not available
  T one_over_delta = 1.0F / delta;

  // vec0:lut_float, result:lut_
  ConstRowMajorArrayMap<T> v0(vec0, 1, static_cast<long>(dim));  // NOLINT
  RowMajorArrayMap<uint8_t> res(result, 1, dim);

  // round to nearest integer, then cast to integer
  res = ((v0 - lo) * one_over_delta).round().template cast<uint8_t>();
}

template <typename T>
inline void scalar_quantize_optimized(uint8_t *ALAYA_RESTRICT result,
                                      const T *ALAYA_RESTRICT vec0,
                                      size_t dim,
                                      T lo,
                                      T delta) {
#if defined(__AVX512F__)
  size_t mul16 = dim - (dim & 0b1111);
  size_t i = 0;
  T one_over_delta = 1 / delta;
  auto lo512 = _mm512_set1_ps(lo);
  auto od512 = _mm512_set1_ps(one_over_delta);
  for (; i < mul16; i += 16) {
    auto cur = _mm512_loadu_ps(&vec0[i]);
    cur = _mm512_mul_ps(_mm512_sub_ps(cur, lo512), od512);  // NOLINT
    auto i8 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(cur));
    _mm_storeu_epi8(&result[i], i8);
  }
  for (; i < dim; ++i) {
    result[i] = static_cast<uint8_t>(std::round((vec0[i] - lo) * one_over_delta));
  }
#else
  scalar_quantize_normal(result, vec0, dim, lo, delta);
#endif
}

template <typename T>
class Lut {
  // split [vl_lut_f,vr_lut_f] into 2^(kNumBits) parts and use the boundaries to represent the
  // lookup results
  static constexpr size_t kNumBits = 8;

 private:
  size_t table_length_ = 0;
  std::vector<uint8_t> lut_;
  T delta_;
  T sum_vl_lut_f_;

 public:
  explicit Lut() = default;
  explicit Lut(const T *rotated_query, size_t padded_dim)
      : table_length_(padded_dim << 2), lut_(table_length_) {
    if constexpr (!std::is_floating_point_v<T>) {
      throw std::invalid_argument("Data type must be a floating point type!");
    }

    // quantize float lut
    std::vector<T> lut_float(
        table_length_);  // padded_dim/4 batch * 16 combination/batch => length = padded_dim*4
    fastscan::pack_lut(padded_dim, rotated_query, lut_float.data());
    T vl_lut_f;  // min val of lut_float
    T vr_lut_f;  // max val of lut_float
    data_range(lut_float.data(), table_length_, vl_lut_f, vr_lut_f);

    delta_ = (vr_lut_f - vl_lut_f) / ((1 << kNumBits) - 1);
    // Here, 'the inner product (float) of every 4 dimensions from <x_b, P^(-1)·qr>' is quantized
    // into a knumBits-bit integer nth_segment. And it can be recovered to the nearest quantization
    // boundary using: vl_lut_f + nth_segment * delta
    scalar_quantize_optimized(lut_.data(), lut_float.data(), table_length_, vl_lut_f, delta_);

    size_t num_table = table_length_ / 16;  // = padded_dim/4, the number of nth_segment
    sum_vl_lut_f_ = vl_lut_f * static_cast<T>(num_table);
    // for quick calculation for <x_b,P^(-1)·qr>, get nth_segment via LUT lookup and return :
    // sum_vl_lut_f_ + sum(nth_segment) * delta_
  }

  auto operator=(Lut &&other) noexcept -> Lut & {
    lut_ = std::move(other.lut_);
    delta_ = other.delta_;
    sum_vl_lut_f_ = other.sum_vl_lut_f_;
    return *this;
  }

  [[nodiscard]] auto lut() const -> const uint8_t * { return lut_.data(); };
  [[nodiscard]] auto delta() const -> T { return delta_; };
  [[nodiscard]] auto sum_vl() const -> T { return sum_vl_lut_f_; };
};
}  // namespace alaya
