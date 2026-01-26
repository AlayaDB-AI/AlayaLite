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

// ============================================================================
// Platform Detection
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #define ALAYA_X86
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
  #define ALAYA_ARM64
#endif

// ============================================================================
// Compiler-specific Target Attributes
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)
  #define ALAYA_TARGET_AVX512 __attribute__((target("avx512f,avx512bw,avx512dq")))
  #define ALAYA_TARGET_AVX2 __attribute__((target("avx2,fma")))
  #define ALAYA_TARGET_SSE4 __attribute__((target("sse4.1")))
  #define ALAYA_TARGET_SSE2 __attribute__((target("sse2")))  // Baseline for x86-64
  #define ALAYA_NOINLINE __attribute__((noinline))
  #define ALAYA_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
  #define ALAYA_TARGET_AVX512
  #define ALAYA_TARGET_AVX2
  #define ALAYA_TARGET_SSE4
  #define ALAYA_TARGET_SSE2
  #define ALAYA_NOINLINE __declspec(noinline)
  #define ALAYA_ALWAYS_INLINE __forceinline
#else
  #define ALAYA_TARGET_AVX512
  #define ALAYA_TARGET_AVX2
  #define ALAYA_TARGET_SSE4
  #define ALAYA_TARGET_SSE2
  #define ALAYA_NOINLINE
  #define ALAYA_ALWAYS_INLINE inline
#endif

// ============================================================================
// SIMD Headers
// ============================================================================

#ifdef ALAYA_X86
  #if defined(__GNUC__) || defined(__clang__)
    #include <cpuid.h>
  #elif defined(_MSC_VER)
    #include <intrin.h>
  #endif
  #include <immintrin.h>
#endif

#ifdef ALAYA_ARM64
  #include <arm_neon.h>
#endif

// ============================================================================
// Optimization Pragmas
// ============================================================================
#if defined(__GNUC__) && !defined(__clang__)
  #define FAST_BEGIN \
    _Pragma("GCC push_options") _Pragma("GCC optimize (\"unroll-loops,fast-math\")")
  #define FAST_END _Pragma("GCC pop_options")
#else
  // Clang / MSVC / others: no-op
  #define FAST_BEGIN
  #define FAST_END
#endif
