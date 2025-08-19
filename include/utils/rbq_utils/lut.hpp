#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "./defines.hpp"
#include "./fastscan.hpp"

namespace alaya {

template <typename T>
inline void data_range(const T *__restrict__ vec0, size_t dim, T &lo, T &hi) { 
  ConstRowMajorArrayMap<T> v0(vec0, 1, dim);// 1行dim列
  lo = v0.minCoeff();
  hi = v0.maxCoeff();
}

template <typename T>
inline void scalar_quantize_normal(T *__restrict__ result, const float *__restrict__ vec0, size_t dim,
                            float lo, float delta) { // normal implementation when AVX512F is not available
  float one_over_delta = 1.0F / delta;

  // vec0:lut_float, result:lut_
  ConstRowMajorArrayMap<float> v0(vec0, 1, static_cast<long>(dim));  // NOLINT
  RowMajorArrayMap<T> res(result, 1, dim);

  // round to nearest integer, then cast to integer
  res = ((v0 - lo) * one_over_delta).round().template cast<T>();
}

inline void scalar_quantize_optimized(
    uint8_t* __restrict__ result,
    const float* __restrict__ vec0,
    size_t dim,
    float lo,
    float delta
) {
#if defined(__AVX512F__)
    size_t mul16 = dim - (dim & 0b1111);
    size_t i = 0;
    float one_over_delta = 1 / delta;
    auto lo512 = _mm512_set1_ps(lo);
    auto od512 = _mm512_set1_ps(one_over_delta);
    for (; i < mul16; i += 16) {
        auto cur = _mm512_loadu_ps(&vec0[i]);
        cur = _mm512_mul_ps(_mm512_sub_ps(cur, lo512), od512); // NOLINT
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
  static constexpr size_t kNumBits = 8;  // B_q
  static_assert(std::is_floating_point_v<T>, "T must be an floating type in Lut");

 private:
  size_t table_length_ = 0;
  std::vector<uint8_t> lut_;
  T delta_;
  T sum_vl_lut_;

 public:
  explicit Lut() = default;
  explicit Lut(const T *rotated_query, size_t padded_dim)
      : table_length_(padded_dim << 2), lut_(table_length_) {
    // quantize float lut
    std::vector<float> lut_float(
        table_length_);  // padded_dim/4 batch * 16 combination/batch => length = padded_dim*4
    fastscan::pack_lut(padded_dim, rotated_query, lut_float.data());
    T vl_lut; // min val of lut
    T vr_lut; // max val of lut
    data_range(lut_float.data(), table_length_, vl_lut, vr_lut);

    delta_ = (vr_lut - vl_lut) / ((1 << kNumBits) - 1);
    // 此处把<x_u(possible val),P^(-1)·qr>的值中每四维的内积加和（float）用knumBits位(m)表示(通过vl_lut+m*delta可还原为归约到的最近边界)
    scalar_quantize_optimized(lut_.data(), lut_float.data(), table_length_, vl_lut, delta_);

    size_t num_table = table_length_ / 16;  // = padded_dim/4, the number of batch
    sum_vl_lut_ = vl_lut * static_cast<float>(num_table);
    // for quick calculation for <x_u,P^(-1)·qr>, get val_vec via LUT lookup and return : sum_vl_lut_+sum(val_vec)*delta_
  }

  auto operator=(Lut &&other) noexcept -> Lut & {
    lut_ = std::move(other.lut_);
    delta_ = other.delta_;
    sum_vl_lut_ = other.sum_vl_lut_;
    return *this;
  }

  [[nodiscard]] auto lut() const -> const uint8_t * { return lut_.data(); };
  [[nodiscard]] auto delta() const -> T { return delta_; };
  [[nodiscard]] auto sum_vl() const -> T { return sum_vl_lut_; };
};
}  // namespace alaya