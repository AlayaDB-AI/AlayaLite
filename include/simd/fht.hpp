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

#include <cstddef>
#include <type_traits>
#include "cpu_features.hpp"
#include "platform.hpp"

namespace alaya::simd {

using FHT_Helper_Func = void (*)(float *a);

template <size_t N>
auto fwht_generic_template(float *buf) -> void;

#ifdef ALAYA_X86
auto helper_float_6_avx2(float *buf) -> void;
auto helper_float_7_avx2(float *buf) -> void;
auto helper_float_8_avx2(float *buf) -> void;
auto helper_float_9_avx2(float *buf) -> void;
auto helper_float_10_avx2(float *buf) -> void;
auto helper_float_11_avx2(float *buf) -> void;

auto helper_float_6_avx512(float *buf) -> void;
auto helper_float_7_avx512(float *buf) -> void;
auto helper_float_8_avx512_recursive(float *buf, int depth) -> void;
auto helper_float_8_avx512(float *buf) -> void;
auto helper_float_9_avx512(float *buf) -> void;
auto helper_float_10_avx512(float *buf) -> void;
auto helper_float_11_avx512(float *buf) -> void;
#endif

// dispatch and public api
auto helper_float_6(float *buf) -> void;
auto helper_float_7(float *buf) -> void;
auto helper_float_8(float *buf) -> void;
auto helper_float_9(float *buf) -> void;
auto helper_float_10(float *buf) -> void;
auto helper_float_11(float *buf) -> void;
auto fht_float(float *buf, int log_n) -> int;
}  // namespace alaya::simd
#include "fht.ipp"
