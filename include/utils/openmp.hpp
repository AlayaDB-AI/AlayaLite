// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include "utils/log.hpp"

#ifdef _OPENMP
  #define ALAYA_OMP_PARALLEL_FOR_DYNAMIC _Pragma("omp parallel for schedule(dynamic)")
#else
  #define ALAYA_OMP_PARALLEL_FOR_DYNAMIC
#endif

namespace alaya::platform {

inline auto openmp_enabled() -> bool {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

inline auto log_openmp_fallback_once() -> void {
#ifndef _OPENMP
  LOG_INFO_ONCE("openmp fallback: OpenMP is unavailable, using serial execution path");
#endif
}

inline auto set_openmp_thread_count(std::size_t num_threads) -> void {
#ifdef _OPENMP
  omp_set_num_threads(static_cast<int>(num_threads));
#else
  (void)num_threads;
  log_openmp_fallback_once();
#endif
}

inline auto openmp_thread_num() -> int {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  log_openmp_fallback_once();
  return 0;
#endif
}

}  // namespace alaya::platform
