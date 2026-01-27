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
// 1. Operating System Detection
// ============================================================================

#include <cstdlib>
#if defined(_WIN32)
  #define ALAYA_OS_WINDOWS
  #include <malloc.h>  // _aligned_malloc required header
#elif defined(__linux__)
  #define ALAYA_OS_LINUX
  #include <sys/mman.h>  // madvise
#elif defined(__APPLE__)
  #define ALAYA_OS_MACOS
#else
  #define ALAYA_OS_UNKNOWN
#endif

// ============================================================================
// 2. Architecture Detection
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #define ALAYA_ARCH_X86
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
  #define ALAYA_ARCH_ARM64
#endif

// ============================================================================
// 3. Compiler-specific Attributes & Helpers
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)
  #define ALAYA_TARGET_AVX512 __attribute__((target("avx512f,avx512bw,avx512dq")))
  #define ALAYA_TARGET_AVX2 __attribute__((target("avx2,fma")))
  #define ALAYA_TARGET_SSE4 __attribute__((target("sse4.1")))
  #define ALAYA_TARGET_SSE2 __attribute__((target("sse2")))  // Baseline for x86-64

  #define ALAYA_NOINLINE __attribute__((noinline))
  #define ALAYA_ALWAYS_INLINE __attribute__((always_inline)) inline

  #define ALAYA_LIKELY(x) __builtin_expect(!!(x), 1)  // Branch Prediction
  #define ALAYA_UNLIKELY(x) __builtin_expect(!!(x), 0)

  #define ALAYA_RESTRICT __restrict__  // Memory Alignment Hint
  #define ALAYA_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
  #define ALAYA_TARGET_AVX512
  #define ALAYA_TARGET_AVX2
  #define ALAYA_TARGET_SSE4
  #define ALAYA_TARGET_SSE2
  #define ALAYA_NOINLINE __declspec(noinline)
  #define ALAYA_ALWAYS_INLINE __forceinline

  #define ALAYA_LIKELY(x) (x)
  #define ALAYA_UNLIKELY(x) (x)

  #define ALAYA_RESTRICT __restrict
  #define ALAYA_UNREACHABLE __assume(0)
#else
  #define ALAYA_TARGET_AVX512
  #define ALAYA_TARGET_AVX2
  #define ALAYA_TARGET_SSE4
  #define ALAYA_TARGET_SSE2
  #define ALAYA_NOINLINE
  #define ALAYA_ALWAYS_INLINE inline
  #define ALAYA_LIKELY(x) (x)
  #define ALAYA_UNLIKELY(x) (x)
  #define ALAYA_RESTRICT
  #define ALAYA_UNREACHABLE
#endif

// ============================================================================
// 4. Memory Allocation Abstraction
// ============================================================================

inline auto alaya_aligned_alloc_impl(size_t size, size_t alignment) -> void * {
#ifdef ALAYA_OS_WINDOWS
  return _aligned_malloc(size, alignment);
#else
  // Notice: C++17 std::aligned_alloc requires size to be a multiple of alignment
  //  size % alignment == 0
  return std::aligned_alloc(alignment, size);
#endif
}

inline void alaya_aligned_free_impl(void *ptr) {
  // Handle nullptr gracefully
  if (ptr == nullptr) {
    return;
  }
#ifdef ALAYA_OS_WINDOWS
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

// ============================================================================
// 5. SIMD Headers
// ============================================================================

#ifdef ALAYA_ARCH_X86
  #if defined(__GNUC__) || defined(__clang__)
    #include <cpuid.h>
  #elif defined(_MSC_VER)
    #include <intrin.h>
  #endif
  #include <immintrin.h>
#endif

#ifdef ALAYA_ARCH_ARM64
  #include <arm_neon.h>
#endif

// ============================================================================
// 6. Optimization Pragmas
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
