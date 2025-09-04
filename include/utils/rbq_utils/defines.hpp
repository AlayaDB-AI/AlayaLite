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

#include <Eigen/Dense>
#include <cstdint>

#define BIT_ID(x) (__builtin_popcount((x) - 1))
#define LOWBIT(x) ((x) & (-(x)))

namespace alaya {
using PID = uint32_t;

constexpr uint32_t kPidMax = 0xFFFFFFFF;

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

}  // namespace alaya
