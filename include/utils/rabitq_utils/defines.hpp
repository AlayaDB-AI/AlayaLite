/*
 * Copyright 2025 VectorDB.NTU
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

#include <Eigen/Dense>

#define LOWBIT(x) ((x) & (-(x)))

#if defined(_MSC_VER)
#define ALAYA_UNREACHABLE __assume(0)
#elif defined(__GNUC__) || defined(__clang__)
#define ALAYA_UNREACHABLE __builtin_unreachable()
#else
#define ALAYA_UNREACHABLE
#endif

namespace alaya {
// Eigen::Matrix<T, Rows, Cols, Options>
// A dense linear algebra matrix supporting mathematical operations (e.g., multiplication,
// inversion). Designed for linear algebra — operators like * mean matrix multiplication. Use it
// when you need standard matrix math (e.g., A * B, A.inverse(), A.transpose()).

// Eigen::Array<T, Rows, Cols, Options>
// A general-purpose multidimensional array supporting coefficient-wise operations (e.g., +, *, sin,
// <). Designed for element-wise math — operators like * mean element-wise multiplication. Use for
// signal processing, per-pixel operations, or when you need "array-style" math, not linear algebra.

// Eigen::Map<MatrixType>
// Zero-copy wrapper that maps existing raw memory (e.g., T*) to an Eigen Matrix or Array interface.
// Allows external buffers (from C arrays, std::vector, etc.) to be used as Eigen objects without
// copying. Ideal for interoperability and performance-critical code. Modifying the Map modifies the
// original memory.
template <typename T>
using RowMajorMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using RowMajorMatrixMap = Eigen::Map<RowMajorMatrix<T>>;

template <typename T>
using ConstRowMajorMatrixMap = Eigen::Map<const RowMajorMatrix<T>>;

template <typename T>
using RowMajorArray = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using RowMajorArrayMap = Eigen::Map<RowMajorArray<T>>;

template <typename T>
using ConstRowMajorArrayMap = Eigen::Map<const RowMajorArray<T>>;

template <typename T>
using VectorMap = Eigen::Map<Vector<T>>;

template <typename T>
using ConstVectorMap = Eigen::Map<const Vector<T>>;

template <typename T>
auto dot_product(const T *__restrict__ vec0, const T *__restrict__ vec1, size_t dim) -> T {
  ConstVectorMap<T> v0(vec0, dim);
  ConstVectorMap<T> v1(vec1, dim);
  return v0.dot(v1);
}

template <typename T>
inline auto l2_sqr(const T *__restrict__ vec0, size_t dim) -> T {
  ConstVectorMap<T> v0(vec0, dim);
  return v0.dot(v0);
}

template <typename T>
inline auto l2_sqr(const T *__restrict__ vec0, const T *__restrict__ vec1, size_t dim) -> T {
  ConstVectorMap<T> v0(vec0, dim);
  ConstVectorMap<T> v1(vec1, dim);
  return (v0 - v1).dot(v0 - v1);
}
}  // namespace alaya
