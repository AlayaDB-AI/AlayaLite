// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "simd/cpu_features.hpp"
#include "utils/log.hpp"
#include "utils/platform.hpp"

namespace alaya::laser::simd {
namespace detail {}

enum class LaserSimdLevel : std::uint8_t { kGeneric, kAvx512, kAvx2 };

using AccumulateFn = void (*)(size_t, const uint8_t *, const uint8_t *, uint16_t *);
using AppDistFn = void (*)(size_t,
                           float,
                           float,
                           float,
                           float,
                           const float *,
                           const float *,
                           const float *,
                           const float *,
                           float *);
using ConvertAccumFn = void (*)(size_t, const uint16_t *, int32_t, float *);
using RotateLoopFn = size_t (*)(const float *, const float *, size_t, float *);
using DataRangeFn = void (*)(const float *, size_t, float &, float &);
using L2SqrSingleFn = float (*)(const float *, size_t);

inline auto get_laser_simd_name(LaserSimdLevel level) -> const char * {
  switch (level) {
    case LaserSimdLevel::kGeneric:
      return "generic";
    case LaserSimdLevel::kAvx512:
      return "avx512";
    case LaserSimdLevel::kAvx2:
      return "avx2";
  }
  throw std::runtime_error("unknown LASER SIMD level");
}

inline auto detect_laser_simd_level() -> LaserSimdLevel {
#ifdef ALAYA_ARCH_X86
  const auto &features = ::alaya::simd::get_cpu_features();
  if (features.avx512f_ && features.avx512bw_) {
    return LaserSimdLevel::kAvx512;
  }
  if (features.avx2_ && features.fma_) {
    return LaserSimdLevel::kAvx2;
  }
  // x86 build_system adds -mavx2 -mfma; silently dropping to scalar would mask
  // a misconfigured runtime. Refuse to fall back here — generic is only for
  // non-x86 targets where the AVX symbols don't exist.
  throw std::runtime_error("LASER on x86 requires AVX2+FMA at minimum");
#else
  return LaserSimdLevel::kGeneric;
#endif
}

inline auto get_laser_simd_level() -> LaserSimdLevel {
  static const LaserSimdLevel kLevel = [] {
    const LaserSimdLevel level = detect_laser_simd_level();
    LOG_INFO("laser_simd={}", get_laser_simd_name(level));
    return level;
  }();
  return kLevel;
}

inline auto get_laser_simd_name() -> const char * {
  return get_laser_simd_name(get_laser_simd_level());
}

template <typename Fn>
inline auto select_laser_simd(Fn generic_fn, Fn avx512_fn, Fn avx2_fn) -> Fn {
  switch (get_laser_simd_level()) {
    case LaserSimdLevel::kGeneric:
      return generic_fn;
    case LaserSimdLevel::kAvx512:
      return avx512_fn;
    case LaserSimdLevel::kAvx2:
      return avx2_fn;
  }
  throw std::runtime_error("unknown LASER SIMD level");
}

namespace detail {

constexpr std::array<int, 16> kPackedLaneOrder =
    {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};

inline void accumulate_impl_generic(size_t dim,
                                    const uint8_t *ALAYA_RESTRICT codes,
                                    const uint8_t *ALAYA_RESTRICT LUT,
                                    uint16_t *ALAYA_RESTRICT result) {
  std::fill(result, result + 32, static_cast<uint16_t>(0));
  const size_t num_codebook = dim >> 2;
  const uint8_t *packed = codes;
  for (size_t codebook = 0; codebook < num_codebook; codebook += 2) {
    const uint8_t *lut0 = LUT + codebook * 16;
    const uint8_t *lut1 = lut0 + 16;
    for (size_t lane = 0; lane < kPackedLaneOrder.size(); ++lane) {
      const int low_id = kPackedLaneOrder[lane];
      const int high_id = low_id + 16;

      const uint8_t packed0 = packed[lane];
      result[low_id] = static_cast<uint16_t>(result[low_id] + lut0[packed0 & 0x0FU]);
      result[high_id] = static_cast<uint16_t>(result[high_id] + lut0[packed0 >> 4U]);

      const uint8_t packed1 = packed[lane + 16];
      result[low_id] = static_cast<uint16_t>(result[low_id] + lut1[packed1 & 0x0FU]);
      result[high_id] = static_cast<uint16_t>(result[high_id] + lut1[packed1 >> 4U]);
    }
    packed += 32;
  }
}

inline void appro_dist_impl_generic(size_t num_points,
                                    float sqr_y,
                                    float width,
                                    float vl,
                                    float sqr_qr,
                                    const float *ALAYA_RESTRICT result,
                                    const float *ALAYA_RESTRICT triple_x,
                                    const float *ALAYA_RESTRICT fac_dq,
                                    const float *ALAYA_RESTRICT fac_vq,
                                    float *ALAYA_RESTRICT appro_dist) {
  for (size_t i = 0; i < num_points; ++i) {
    appro_dist[i] =
        sqr_y + triple_x[i] + (fac_dq[i] * width * result[i]) + (fac_vq[i] * vl) + sqr_qr;
  }
}

inline void convert_accum_to_float_generic(size_t count,
                                           const uint16_t *ALAYA_RESTRICT result,
                                           int32_t sumq,
                                           float *ALAYA_RESTRICT result_float) {
  for (size_t i = 0; i < count; ++i) {
    result_float[i] = static_cast<float>((static_cast<int16_t>(result[i]) << 1) - sumq);
  }
}

inline auto rotate_loop_generic(const float *ALAYA_RESTRICT src,
                                const float *ALAYA_RESTRICT mat,
                                size_t dim,
                                float *ALAYA_RESTRICT dst) -> size_t {
  for (size_t i = 0; i < dim; ++i) {
    dst[i] = src[i] * mat[i];
  }
  return dim;
}

inline void data_range_generic(const float *ALAYA_RESTRICT vec, size_t dim, float &lo, float &hi) {
  if (dim == 0) {
    lo = 0.0F;
    hi = 0.0F;
    return;
  }
  lo = FLT_MAX;
  hi = -FLT_MAX;
  for (size_t i = 0; i < dim; ++i) {
    const float value = vec[i];
    lo = value < lo ? value : lo;
    hi = value > hi ? value : hi;
  }
}

inline float l2_sqr_single_generic(const float *ALAYA_RESTRICT vec0, size_t dim) {
  float result = 0.0F;
  for (size_t i = 0; i < dim; ++i) {
    const float value = vec0[i];
    result += value * value;
  }
  return result;
}

#ifdef ALAYA_ARCH_X86

ALAYA_TARGET_AVX512_BW
inline void accumulate_impl_avx512(size_t dim,
                                   const uint8_t *ALAYA_RESTRICT codes,
                                   const uint8_t *ALAYA_RESTRICT LUT,
                                   uint16_t *ALAYA_RESTRICT result) {
  size_t code_length = dim << 2;
  __m512i c;
  __m512i lo;
  __m512i hi;
  __m512i lut;
  __m512i res_lo;
  __m512i res_hi;

  const __m512i lo_mask = _mm512_set1_epi8(0x0f);
  __m512i accu0 = _mm512_setzero_si512();
  __m512i accu1 = _mm512_setzero_si512();
  __m512i accu2 = _mm512_setzero_si512();
  __m512i accu3 = _mm512_setzero_si512();

  for (size_t i = 0; i < code_length; i += 64) {
    c = _mm512_loadu_si512(&codes[i]);
    lut = _mm512_loadu_si512(&LUT[i]);
    lo = _mm512_and_si512(c, lo_mask);
    hi = _mm512_and_si512(_mm512_srli_epi16(c, 4), lo_mask);

    res_lo = _mm512_shuffle_epi8(lut, lo);
    res_hi = _mm512_shuffle_epi8(lut, hi);

    accu0 = _mm512_add_epi16(accu0, res_lo);
    accu1 = _mm512_add_epi16(accu1, _mm512_srli_epi16(res_lo, 8));
    accu2 = _mm512_add_epi16(accu2, res_hi);
    accu3 = _mm512_add_epi16(accu3, _mm512_srli_epi16(res_hi, 8));
  }
  accu0 = _mm512_sub_epi16(accu0, _mm512_slli_epi16(accu1, 8));
  accu2 = _mm512_sub_epi16(accu2, _mm512_slli_epi16(accu3, 8));

  __m512i ret1 = _mm512_add_epi16(_mm512_mask_blend_epi64(0b11110000, accu0, accu1),
                                  _mm512_shuffle_i64x2(accu0, accu1, 0b01001110));
  __m512i ret2 = _mm512_add_epi16(_mm512_mask_blend_epi64(0b11110000, accu2, accu3),
                                  _mm512_shuffle_i64x2(accu2, accu3, 0b01001110));
  __m512i ret = _mm512_setzero_si512();

  ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b10001000));
  ret = _mm512_add_epi16(ret, _mm512_shuffle_i64x2(ret1, ret2, 0b11011101));

  _mm512_storeu_si512(result, ret);
}

ALAYA_TARGET_AVX2
inline void accumulate_impl_avx2(size_t dim,
                                 const uint8_t *ALAYA_RESTRICT codes,
                                 const uint8_t *ALAYA_RESTRICT LUT,
                                 uint16_t *ALAYA_RESTRICT result) {
  size_t code_length = dim << 2;
  __m256i c, lo, hi, lut, res_lo, res_hi;

  __m256i low_mask = _mm256_set1_epi8(0xf);
  __m256i accu0 = _mm256_setzero_si256();
  __m256i accu1 = _mm256_setzero_si256();
  __m256i accu2 = _mm256_setzero_si256();
  __m256i accu3 = _mm256_setzero_si256();

  for (size_t i = 0; i < code_length; i += 64) {
    c = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&codes[i]));
    lut = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&LUT[i]));
    lo = _mm256_and_si256(c, low_mask);
    hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

    res_lo = _mm256_shuffle_epi8(lut, lo);
    res_hi = _mm256_shuffle_epi8(lut, hi);

    accu0 = _mm256_add_epi16(accu0, res_lo);
    accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
    accu2 = _mm256_add_epi16(accu2, res_hi);
    accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

    c = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&codes[i + 32]));
    lut = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&LUT[i + 32]));
    lo = _mm256_and_si256(c, low_mask);
    hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

    res_lo = _mm256_shuffle_epi8(lut, lo);
    res_hi = _mm256_shuffle_epi8(lut, hi);

    accu0 = _mm256_add_epi16(accu0, res_lo);
    accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
    accu2 = _mm256_add_epi16(accu2, res_hi);
    accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
  }

  accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
  __m256i dis0 = _mm256_add_epi16(_mm256_permute2f128_si256(accu0, accu1, 0x21),
                                  _mm256_blend_epi32(accu0, accu1, 0xF0));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(result), dis0);

  accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
  __m256i dis1 = _mm256_add_epi16(_mm256_permute2f128_si256(accu2, accu3, 0x21),
                                  _mm256_blend_epi32(accu2, accu3, 0xF0));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(&result[16]), dis1);
}

ALAYA_TARGET_AVX512_BW
inline void appro_dist_impl_avx512(size_t num_points,
                                   float sqr_y,
                                   float width,
                                   float vl,
                                   float sqr_qr,
                                   const float *ALAYA_RESTRICT result,
                                   const float *ALAYA_RESTRICT triple_x,
                                   const float *ALAYA_RESTRICT fac_dq,
                                   const float *ALAYA_RESTRICT fac_vq,
                                   float *ALAYA_RESTRICT appro_dist) {
  // sqr_qr is ||q_r||^2 (query residual norm); triple_x already carries ||x_r||^2.
  const __m512 sqr_y_simd = _mm512_set1_ps(sqr_y);
  const __m512 width_simd = _mm512_set1_ps(width);
  const __m512 vl_simd = _mm512_set1_ps(vl);
  const __m512 sqr_qr_simd = _mm512_set1_ps(sqr_qr);

  size_t i = 0;
  const size_t simd_count = num_points - (num_points % 16);
  for (; i < simd_count; i += 16) {
    __m512 result_simd = _mm512_loadu_ps(&result[i]);
    __m512 triple_x_simd = _mm512_loadu_ps(&triple_x[i]);
    __m512 fac_dq_simd = _mm512_loadu_ps(&fac_dq[i]);
    __m512 fac_vq_simd = _mm512_loadu_ps(&fac_vq[i]);

    triple_x_simd = _mm512_add_ps(triple_x_simd, sqr_y_simd);
    fac_dq_simd = _mm512_mul_ps(_mm512_mul_ps(fac_dq_simd, width_simd), result_simd);
    fac_vq_simd = _mm512_fmadd_ps(fac_vq_simd, vl_simd, triple_x_simd);
    triple_x_simd = _mm512_add_ps(fac_dq_simd, fac_vq_simd);
    triple_x_simd = _mm512_add_ps(triple_x_simd, sqr_qr_simd);
    _mm512_storeu_ps(&appro_dist[i], triple_x_simd);
  }
  for (; i < num_points; ++i) {
    appro_dist[i] =
        sqr_y + triple_x[i] + (fac_dq[i] * width * result[i]) + (fac_vq[i] * vl) + sqr_qr;
  }
}

ALAYA_TARGET_AVX2
inline void appro_dist_impl_avx2(size_t num_points,
                                 float sqr_y,
                                 float width,
                                 float vl,
                                 float sqr_qr,
                                 const float *ALAYA_RESTRICT result,
                                 const float *ALAYA_RESTRICT triple_x,
                                 const float *ALAYA_RESTRICT fac_dq,
                                 const float *ALAYA_RESTRICT fac_vq,
                                 float *ALAYA_RESTRICT appro_dist) {
  // sqr_qr is ||q_r||^2 (query residual norm); triple_x already carries ||x_r||^2.
  const __m256 sqr_y_simd = _mm256_set1_ps(sqr_y);
  const __m256 width_simd = _mm256_set1_ps(width);
  const __m256 vl_simd = _mm256_set1_ps(vl);
  const __m256 sqr_qr_simd = _mm256_set1_ps(sqr_qr);

  size_t i = 0;
  const size_t simd_count = num_points - (num_points % 8);
  for (; i < simd_count; i += 8) {
    __m256 result_simd = _mm256_loadu_ps(&result[i]);
    __m256 triple_x_simd = _mm256_loadu_ps(&triple_x[i]);
    __m256 fac_dq_simd = _mm256_loadu_ps(&fac_dq[i]);
    __m256 fac_vq_simd = _mm256_loadu_ps(&fac_vq[i]);

    triple_x_simd = _mm256_add_ps(triple_x_simd, sqr_y_simd);
    fac_dq_simd = _mm256_mul_ps(_mm256_mul_ps(fac_dq_simd, width_simd), result_simd);
    fac_vq_simd = _mm256_mul_ps(fac_vq_simd, vl_simd);
    triple_x_simd = _mm256_add_ps(_mm256_add_ps(triple_x_simd, fac_dq_simd), fac_vq_simd);
    triple_x_simd = _mm256_add_ps(triple_x_simd, sqr_qr_simd);
    _mm256_storeu_ps(&appro_dist[i], triple_x_simd);
  }
  for (; i < num_points; ++i) {
    appro_dist[i] =
        sqr_y + triple_x[i] + (fac_dq[i] * width * result[i]) + (fac_vq[i] * vl) + sqr_qr;
  }
}

ALAYA_TARGET_AVX512_BW
inline void convert_accum_to_float_avx512(size_t count,
                                          const uint16_t *ALAYA_RESTRICT result,
                                          int32_t sumq,
                                          float *ALAYA_RESTRICT result_float) {
  // FastScan accumulators are stored as uint16 but carry int16 bit patterns
  // (the _mm*_sub_epi16 in accumulate_impl_* can produce negatives). The SIMD
  // path uses _mm*_cvtepi16_epi32 to sign-extend; the scalar tail mirrors this
  // via static_cast<int16_t> -- changing it to int32_t would zero-extend and
  // diverge from the SIMD output.
  const __m512i qq = _mm512_set1_epi32(sumq);
  size_t i = 0;
  const size_t simd_count = count - (count % 32);
  for (; i < simd_count; i += 32) {
    __m256i i16a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&result[i]));
    __m256i i16b = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&result[i + 16]));
    __m512i i32a = _mm512_cvtepi16_epi32(i16a);
    __m512i i32b = _mm512_cvtepi16_epi32(i16b);

    i32a = _mm512_sub_epi32(_mm512_slli_epi32(i32a, 1), qq);
    i32b = _mm512_sub_epi32(_mm512_slli_epi32(i32b, 1), qq);
    _mm512_storeu_ps(&result_float[i], _mm512_cvtepi32_ps(i32a));
    _mm512_storeu_ps(&result_float[i + 16], _mm512_cvtepi32_ps(i32b));
  }
  for (; i < count; ++i) {
    result_float[i] = static_cast<float>((static_cast<int16_t>(result[i]) << 1) - sumq);
  }
}

ALAYA_TARGET_AVX2
inline void convert_accum_to_float_avx2(size_t count,
                                        const uint16_t *ALAYA_RESTRICT result,
                                        int32_t sumq,
                                        float *ALAYA_RESTRICT result_float) {
  const __m256i qq = _mm256_set1_epi32(sumq);
  size_t i = 0;
  const size_t simd_count = count - (count % 8);
  for (; i < simd_count; i += 8) {
    __m128i i16 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&result[i]));
    __m256i i32 = _mm256_cvtepi16_epi32(i16);
    i32 = _mm256_sub_epi32(_mm256_slli_epi32(i32, 1), qq);
    _mm256_storeu_ps(&result_float[i], _mm256_cvtepi32_ps(i32));
  }
  for (; i < count; ++i) {
    result_float[i] = static_cast<float>((static_cast<int16_t>(result[i]) << 1) - sumq);
  }
}

ALAYA_TARGET_AVX512_BW
inline auto rotate_loop_avx512(const float *ALAYA_RESTRICT src,
                               const float *ALAYA_RESTRICT mat,
                               size_t dim,
                               float *ALAYA_RESTRICT dst) -> size_t {
  size_t idx = 0;
  for (; idx + 16 <= dim; idx += 16) {
    __m512 ss = _mm512_loadu_ps(&src[idx]);
    __m512 mm = _mm512_loadu_ps(&mat[idx]);
    ss = _mm512_mul_ps(ss, mm);
    _mm512_storeu_ps(&dst[idx], ss);
  }
  return idx;
}

ALAYA_TARGET_AVX2
inline auto rotate_loop_avx2(const float *ALAYA_RESTRICT src,
                             const float *ALAYA_RESTRICT mat,
                             size_t dim,
                             float *ALAYA_RESTRICT dst) -> size_t {
  size_t idx = 0;
  for (; idx + 8 <= dim; idx += 8) {
    __m256 ss = _mm256_loadu_ps(&src[idx]);
    __m256 mm = _mm256_loadu_ps(&mat[idx]);
    ss = _mm256_mul_ps(ss, mm);
    _mm256_storeu_ps(&dst[idx], ss);
  }
  return idx;
}

ALAYA_TARGET_AVX512_BW
inline void data_range_avx512(const float *ALAYA_RESTRICT vec, size_t dim, float &lo, float &hi) {
  if (dim == 0) {
    lo = 0.0F;
    hi = 0.0F;
    return;
  }
  __m512 max_q = _mm512_set1_ps(-FLT_MAX);
  __m512 min_q = _mm512_set1_ps(FLT_MAX);
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    __m512 y1 = _mm512_loadu_ps(&vec[i]);
    max_q = _mm512_max_ps(y1, max_q);
    min_q = _mm512_min_ps(y1, min_q);
  }
  hi = (i == 0) ? -FLT_MAX : _mm512_reduce_max_ps(max_q);
  lo = (i == 0) ? FLT_MAX : _mm512_reduce_min_ps(min_q);
  for (; i < dim; ++i) {
    const float tmp = vec[i];
    lo = tmp < lo ? tmp : lo;
    hi = tmp > hi ? tmp : hi;
  }
}

ALAYA_TARGET_AVX2
inline void data_range_avx2(const float *ALAYA_RESTRICT vec, size_t dim, float &lo, float &hi) {
  if (dim == 0) {
    lo = 0.0F;
    hi = 0.0F;
    return;
  }
  __m256 max_q = _mm256_set1_ps(-FLT_MAX);
  __m256 min_q = _mm256_set1_ps(FLT_MAX);
  size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m256 y1 = _mm256_loadu_ps(&vec[i]);
    max_q = _mm256_max_ps(y1, max_q);
    min_q = _mm256_min_ps(y1, min_q);
  }
  if (i == 0) {
    hi = -FLT_MAX;
    lo = FLT_MAX;
  } else {
    alignas(32) std::array<float, 8> max_values{};
    alignas(32) std::array<float, 8> min_values{};
    _mm256_store_ps(max_values.data(), max_q);
    _mm256_store_ps(min_values.data(), min_q);
    hi = max_values[0];
    lo = min_values[0];
    for (size_t lane = 1; lane < max_values.size(); ++lane) {
      hi = max_values[lane] > hi ? max_values[lane] : hi;
      lo = min_values[lane] < lo ? min_values[lane] : lo;
    }
  }
  for (; i < dim; ++i) {
    const float tmp = vec[i];
    lo = tmp < lo ? tmp : lo;
    hi = tmp > hi ? tmp : hi;
  }
}

inline float reduce_add_m256(__m256 x) {
  auto sumh = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
  auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

ALAYA_TARGET_AVX512_BW
inline float l2_sqr_single_avx512(const float *ALAYA_RESTRICT vec0, size_t dim) {
  float result = 0;
  size_t mul16 = dim - (dim & 0b1111);
  auto sum = _mm512_setzero_ps();
  size_t i = 0;
  for (; i < mul16; i += 16) {
    auto xxx = _mm512_loadu_ps(&vec0[i]);
    sum = _mm512_fmadd_ps(xxx, xxx, sum);
  }
  result = _mm512_reduce_add_ps(sum);
  for (; i < dim; ++i) {
    float tmp = vec0[i];
    result += tmp * tmp;
  }
  return result;
}

ALAYA_TARGET_AVX2
inline float l2_sqr_single_avx2(const float *ALAYA_RESTRICT vec0, size_t dim) {
  size_t mul8 = dim - (dim & 0b111);
  __m256 sum = _mm256_setzero_ps();
  size_t i = 0;
  for (; i < mul8; i += 8) {
    __m256 xx = _mm256_loadu_ps(&vec0[i]);
    sum = _mm256_fmadd_ps(xx, xx, sum);
  }
  float result = reduce_add_m256(sum);
  for (; i < dim; ++i) {
    float tmp = vec0[i];
    result += tmp * tmp;
  }
  return result;
}

#endif  // ALAYA_ARCH_X86

}  // namespace detail

inline auto get_accumulate_func() -> AccumulateFn {
#ifdef ALAYA_ARCH_X86
  static const AccumulateFn kFunc = select_laser_simd<AccumulateFn>(detail::accumulate_impl_generic,
                                                                    detail::accumulate_impl_avx512,
                                                                    detail::accumulate_impl_avx2);
#else
  static const AccumulateFn kFunc = detail::accumulate_impl_generic;
#endif
  return kFunc;
}

inline auto get_appro_dist_func() -> AppDistFn {
#ifdef ALAYA_ARCH_X86
  static const AppDistFn kFunc = select_laser_simd<AppDistFn>(detail::appro_dist_impl_generic,
                                                              detail::appro_dist_impl_avx512,
                                                              detail::appro_dist_impl_avx2);
#else
  static const AppDistFn kFunc = detail::appro_dist_impl_generic;
#endif
  return kFunc;
}

inline auto get_convert_func() -> ConvertAccumFn {
#ifdef ALAYA_ARCH_X86
  static const ConvertAccumFn kFunc =
      select_laser_simd<ConvertAccumFn>(detail::convert_accum_to_float_generic,
                                        detail::convert_accum_to_float_avx512,
                                        detail::convert_accum_to_float_avx2);
#else
  static const ConvertAccumFn kFunc = detail::convert_accum_to_float_generic;
#endif
  return kFunc;
}

inline auto get_rotate_loop_func() -> RotateLoopFn {
#ifdef ALAYA_ARCH_X86
  static const RotateLoopFn kFunc = select_laser_simd<RotateLoopFn>(detail::rotate_loop_generic,
                                                                    detail::rotate_loop_avx512,
                                                                    detail::rotate_loop_avx2);
#else
  static const RotateLoopFn kFunc = detail::rotate_loop_generic;
#endif
  return kFunc;
}

inline auto get_data_range_func() -> DataRangeFn {
#ifdef ALAYA_ARCH_X86
  static const DataRangeFn kFunc = select_laser_simd<DataRangeFn>(detail::data_range_generic,
                                                                  detail::data_range_avx512,
                                                                  detail::data_range_avx2);
#else
  static const DataRangeFn kFunc = detail::data_range_generic;
#endif
  return kFunc;
}

inline auto get_l2_sqr_single_func() -> L2SqrSingleFn {
#ifdef ALAYA_ARCH_X86
  static const L2SqrSingleFn kFunc = select_laser_simd<L2SqrSingleFn>(detail::l2_sqr_single_generic,
                                                                      detail::l2_sqr_single_avx512,
                                                                      detail::l2_sqr_single_avx2);
#else
  static const L2SqrSingleFn kFunc = detail::l2_sqr_single_generic;
#endif
  return kFunc;
}

}  // namespace alaya::laser::simd
