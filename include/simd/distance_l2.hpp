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

#include <cstddef>
#include <type_traits>
#include "cpu_features.hpp"
#include "platform.hpp"

namespace alaya::simd {

// Type Definitions
using L2SqrFunc = float (*)(const float *__restrict, const float *__restrict, size_t);
using L2SqrSq8Func = float (*)(const uint8_t *__restrict,
                               const uint8_t *__restrict,
                               const float *,
                               const float *,
                               size_t);
using L2SqrSq4Func = float (*)(const uint8_t *__restrict,
                               const uint8_t *__restrict,
                               const float *,
                               const float *,
                               size_t);

auto l2_sqr_generic(const float *__restrict x, const float *__restrict y, size_t dim) -> float;
#ifdef ALAYA_X86
auto l2_sqr_avx2(const float *__restrict x, const float *__restrict y, size_t dim) -> float;
auto l2_sqr_avx512(const float *__restrict x, const float *__restrict y, size_t dim) -> float;
#endif

auto l2_sqr_sq8_generic(const uint8_t *__restrict x,
                        const uint8_t *__restrict y,
                        const float *min,
                        const float *max,
                        size_t dim) -> float;

#ifdef ALAYA_X86
auto l2_sqr_sq8_avx2(const uint8_t *__restrict x,
                     const uint8_t *__restrict y,
                     const float *min,
                     const float *max,
                     size_t dim) -> float;
auto l2_sqr_sq8_avx512(const uint8_t *__restrict x,
                       const uint8_t *__restrict y,
                       const float *min,
                       const float *max,
                       size_t dim) -> float;
#endif

auto l2_sqr_sq4_generic(const uint8_t *__restrict x,
                        const uint8_t *__restrict y,
                        const float *min,
                        const float *max,
                        size_t dim) -> float;

#ifdef ALAYA_X86
auto l2_sqr_sq4_avx2(const uint8_t *__restrict x,
                     const uint8_t *__restrict y,
                     const float *min,
                     const float *max,
                     size_t dim) -> float;
auto l2_sqr_sq4_avx512(const uint8_t *__restrict x,
                       const uint8_t *__restrict y,
                       const float *min,
                       const float *max,
                       size_t dim) -> float;
#endif

// Dispatch
auto get_l2_sqr_func() -> L2SqrFunc;
auto get_l2_sqr_sq8_func() -> L2SqrSq8Func;
auto get_l2_sqr_sq4_func() -> L2SqrSq4Func;

// Public API
template <typename DataType = float, typename DistanceType = float>
auto l2_sqr(const DataType *__restrict x, const DataType *__restrict y, size_t dim) -> DistanceType;

template <typename DataType = float, typename DistanceType = float>
auto l2_sqr_sq8(const uint8_t *__restrict x,
                const uint8_t *__restrict y,
                const DataType *min,
                const DataType *max,
                size_t dim) -> DistanceType;

template <typename DataType = float, typename DistanceType = float>
auto l2_sqr_sq4(const uint8_t *__restrict x,
                const uint8_t *__restrict y,
                const DataType *min,
                const DataType *max,
                size_t dim) -> DistanceType;

}  // namespace alaya::simd

// Implementation
#include "distance_l2.ipp"
