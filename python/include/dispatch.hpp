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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#pragma once

namespace alaya {

namespace py = pybind11;

#define DISPATCH_DATA_TYPE(...)                                    \
  do {                                                             \
    if (params_.data_type_.is(py::dtype::of<float>())) {           \
      using DataType = float;                                      \
      __VA_ARGS__                                                  \
    } else if (params_.data_type_.is(py::dtype::of<uint8_t>())) {  \
      using DataType = uint8_t;                                    \
      __VA_ARGS__                                                  \
    } else if (params_.data_type_.is(py::dtype::of<int8_t>())) {   \
      using DataType = int8_t;                                     \
      __VA_ARGS__                                                  \
    } else if (params_.data_type_.is(py::dtype::of<int32_t>())) {  \
      using DataType = int32_t;                                    \
      __VA_ARGS__                                                  \
    } else if (params_.data_type_.is(py::dtype::of<uint32_t>())) { \
      using DataType = uint32_t;                                   \
      __VA_ARGS__                                                  \
    } else if (params_.data_type_.is(py::dtype::of<double>())) {   \
      using DataType = double;                                     \
      __VA_ARGS__                                                  \
    } else {                                                       \
      throw std::runtime_error("Unsupported data type");           \
    }                                                              \
  } while (0);

#define DISPATCH_DATA_TYPE_WITH_ARR(NTYPED_ARR, TYPED_ARR, ...)    \
  do {                                                             \
    if (params_.data_type_.is(py::dtype::of<float>())) {           \
      using DataType = float;                                      \
      auto TYPED_ARR = NTYPED_ARR.cast<py::array_t<DataType>>();   \
      __VA_ARGS__                                                  \
    } else if (params_.data_type_.is(py::dtype::of<uint8_t>())) {  \
      using DataType = uint8_t;                                    \
      auto TYPED_ARR = NTYPED_ARR.cast<py::array_t<DataType>>();   \
      __VA_ARGS__                                                  \
    } else if (params_.data_type_.is(py::dtype::of<int8_t>())) {   \
      using DataType = int8_t;                                     \
      auto TYPED_ARR = NTYPED_ARR.cast<py::array_t<DataType>>();   \
      __VA_ARGS__                                                  \
    } else if (params_.data_type_.is(py::dtype::of<int32_t>())) {  \
      using DataType = int32_t;                                    \
      auto TYPED_ARR = NTYPED_ARR.cast<py::array_t<DataType>>();   \
      __VA_ARGS__                                                  \
    } else if (params_.data_type_.is(py::dtype::of<uint32_t>())) { \
      using DataType = uint32_t;                                   \
      auto TYPED_ARR = NTYPED_ARR.cast<py::array_t<DataType>>();   \
      __VA_ARGS__                                                  \
    } else if (params_.data_type_.is(py::dtype::of<double>())) {   \
      using DataType = double;                                     \
      auto TYPED_ARR = NTYPED_ARR.cast<py::array_t<DataType>>();   \
      __VA_ARGS__                                                  \
    } else {                                                       \
      throw std::runtime_error("Unsupported data type");           \
    }                                                              \
  } while (0);

#define DISPATCH_ID_TYPE(...)                                    \
  do {                                                           \
    if (params_.id_type_.is(py::dtype::of<uint32_t>())) {        \
      using IDType = uint32_t;                                   \
      __VA_ARGS__                                                \
    } else if (params_.id_type_.is(py::dtype::of<uint64_t>())) { \
      using IDType = uint64_t;                                   \
      __VA_ARGS__                                                \
    } else {                                                     \
      throw std::runtime_error("Unsupported id type");           \
    }                                                            \
  } while (0);

#define DISPATCH_DISTANCE_TYPE(...) \
  do {                              \
    using DistanceType = float;     \
    __VA_ARGS__                     \
  } while (0);

#define DISPATCH_BUILD_SPACE_TYPE(...)                               \
  do {                                                               \
    using BuildSpaceType = RawSpace<DataType, DistanceType, IDType>; \
    __VA_ARGS__                                                      \
  } while (0);

#define DISPATCH_BUILDER_TYPE(...)                                             \
  do {                                                                         \
    if (params_.index_type_ == IndexType::HNSW) {                              \
      using GraphBuilderType = HNSWBuilder<BuildSpaceType>;                    \
      __VA_ARGS__                                                              \
    } else if (params_.index_type_ == IndexType::NSG) {                        \
      using GraphBuilderType = NSGBuilder<BuildSpaceType>;                     \
      __VA_ARGS__                                                              \
    } else if (params_.index_type_ == IndexType::FUSION) {                     \
      using GraphBuilderType = FusionGraphBuilder<BuildSpaceType,              \
                                                  HNSWBuilder<BuildSpaceType>, \
                                                  NSGBuilder<BuildSpaceType>>; \
      __VA_ARGS__                                                              \
    } else {                                                                   \
      throw std::runtime_error("Unsupported index type");                      \
    }                                                                          \
  } while (0);

#define DISPATCH_SEARCH_SPACE_TYPE(...)                                    \
  do {                                                                     \
    if (params_.quantization_type_ == QuantizationType::NONE) {            \
      using SearchSpaceType = RawSpace<DataType, DistanceType, IDType>;    \
      __VA_ARGS__                                                          \
    } else if (params_.quantization_type_ == QuantizationType::SQ8) {      \
      using SearchSpaceType = SQ8Space<DataType, DistanceType, IDType>;    \
      __VA_ARGS__                                                          \
    } else if (params_.quantization_type_ == QuantizationType::SQ4) {      \
      using SearchSpaceType = SQ4Space<DataType, DistanceType, IDType>;    \
      __VA_ARGS__                                                          \
    } else if (params_.quantization_type_ == QuantizationType::RABITQ) {   \
      using SearchSpaceType = RaBitQSpace<DataType, DistanceType, IDType>; \
      __VA_ARGS__                                                          \
    } else {                                                               \
      throw std::runtime_error("Unsupported quantization type");           \
    }                                                                      \
  } while (0);

#define CAST_INDEX(INDEX, ...)                                                             \
  do {                                                                                     \
    auto INDEX =                                                                           \
        std::reinterpret_pointer_cast<PyIndex<GraphBuilderType, SearchSpaceType>>(index_); \
    __VA_ARGS__                                                                            \
  } while (0);

#define CREATE_INDEX(PARAMS, ...)                                                  \
  do {                                                                             \
    index_ = std::make_shared<PyIndex<GraphBuilderType, SearchSpaceType>>(PARAMS); \
    __VA_ARGS__                                                                    \
  } while (0);

#define DISPATCH_AND_CAST_WITH_ARR(NTYPED_ARR, TYPED_ARR, INDEX, ...)                              \
  do {                                                                                             \
    DISPATCH_DATA_TYPE_WITH_ARR(NTYPED_ARR,                                                        \
                                TYPED_ARR,                                                         \
                                DISPATCH_DISTANCE_TYPE(DISPATCH_ID_TYPE(DISPATCH_BUILD_SPACE_TYPE( \
                                    DISPATCH_BUILDER_TYPE(DISPATCH_SEARCH_SPACE_TYPE(              \
                                        CAST_INDEX(INDEX, __VA_ARGS__)))))));                      \
  } while (0);

#define DISPATCH_AND_CAST(INDEX, ...)                                                          \
  do {                                                                                         \
    DISPATCH_DATA_TYPE(DISPATCH_DISTANCE_TYPE(DISPATCH_ID_TYPE(DISPATCH_BUILD_SPACE_TYPE(      \
        DISPATCH_BUILDER_TYPE(DISPATCH_SEARCH_SPACE_TYPE(CAST_INDEX(INDEX, __VA_ARGS__))))))); \
  } while (0);

#define DISPATCH_AND_CREATE(PARAMS, ...)                                                          \
  do {                                                                                            \
    DISPATCH_DATA_TYPE(DISPATCH_DISTANCE_TYPE(DISPATCH_ID_TYPE(DISPATCH_BUILD_SPACE_TYPE(         \
        DISPATCH_BUILDER_TYPE(DISPATCH_SEARCH_SPACE_TYPE(CREATE_INDEX(PARAMS, __VA_ARGS__))))))); \
  } while (0);
}  // namespace alaya
