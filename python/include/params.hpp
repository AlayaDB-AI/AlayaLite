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
#include <pybind11/numpy.h>
#include <sys/types.h>
#include <cstdint>

#include "index/index_type.hpp"
#include "utils/metric_type.hpp"
#include "utils/quantization_type.hpp"

namespace py = pybind11;
namespace alaya {

struct IndexParams {
  IndexType index_type_ = IndexType::HNSW;
  py::dtype data_type_ = py::dtype::of<float>();
  py::dtype id_type_ = py::dtype::of<uint32_t>();
  QuantizationType quantization_type_ = QuantizationType::NONE;
  MetricType metric_ = MetricType::L2;
  uint32_t capacity_ = 100000;
  uint32_t max_nbrs_ = 32;

  IndexParams(IndexType index_type = IndexType::HNSW,  // NOLINT
              py::dtype data_type = py::dtype::of<float>(),
              py::dtype id_type = py::dtype::of<uint32_t>(),
              QuantizationType quantization_type = QuantizationType::NONE,
              MetricType metric = MetricType::L2, uint32_t capacity = 100000,
              uint32_t max_nbrs = 32)
      : index_type_(index_type),
        data_type_(std::move(data_type)),
        id_type_(std::move(id_type)),
        quantization_type_(quantization_type),
        metric_(metric),
        capacity_(capacity),
        max_nbrs_(max_nbrs) {}
};
}  // namespace alaya