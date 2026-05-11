// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <type_traits>
#include "cpu_features.hpp"

namespace alaya::simd {

using FHT_Helper_Func = void (*)(float *a);

template <size_t N>
auto fwht_generic_template(float *buf) -> void;

#if defined(ALAYA_ARCH_X86) && !defined(_MSC_VER)
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
