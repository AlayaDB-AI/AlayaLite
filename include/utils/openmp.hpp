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
