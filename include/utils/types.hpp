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
#include <cstdint>
#include <stdexcept>

using WorkerID = uint32_t;
using CpuID = uint32_t;
namespace alaya {
inline auto do_align(size_t size, size_t align_num) -> size_t {
  // Ensure align_num is not zero (to avoid division by zero)
  if (align_num == 0) {
    throw std::invalid_argument("align_num cannot be zero");
  }

  // Compute the aligned dimension
  return ((size + align_num - 1) / align_num) * align_num;
}
}  // namespace alaya
