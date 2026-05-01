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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "index/disk/disk_collection.hpp"
#include "index/disk/types.hpp"
#include "utils/metric_type.hpp"

namespace alaya::disk::pybindings {

namespace py = pybind11;

inline auto index_type_from_string_strict(const std::string &s) -> DiskIndexType {
  if (s == "disk_flat") {
    return DiskIndexType::Flat;
  }
  if (s == "disk_vamana") {
    return DiskIndexType::Vamana;
  }
  if (s == "disk_laser") {
    throw py::value_error("DiskCollection: index_type \"disk_laser\" not implemented in v1");
  }
  throw py::value_error("DiskCollection: unknown index_type \"" + s +
                        "\"; supported values are \"disk_flat\" and \"disk_vamana\"");
}

inline auto metric_name(MetricType metric) -> std::string {
  switch (metric) {
    case MetricType::L2:
      return "L2";
    case MetricType::IP:
      return "IP";
    case MetricType::COS:
      return "COS";
    case MetricType::NONE:
      return "NONE";
  }
  return "unknown";
}

inline auto is_finite_f64(double value) -> bool {
  uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(value));
  return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

inline auto is_finite_f32(float value) -> bool {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(value));
  return (bits & 0x7F800000U) != 0x7F800000U;
}

inline auto resolve_vamana_num_threads(int64_t num_threads) -> int64_t {
  if (num_threads != 0) {
    return num_threads;
  }
  const char *env = std::getenv("OMP_NUM_THREADS");
  if (env == nullptr || *env == '\0') {
    return 0;
  }
  try {
    size_t pos = 0;
    const auto value = std::stoll(env, &pos);
    if (pos == std::strlen(env) && value > 0 &&
        value <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return value;
    }
  } catch (const std::exception &) {
    // Invalid environment values keep the C++ adapter default.
  }
  return 0;
}

inline auto validate_vamana_params(int64_t r,
                                   int64_t l,
                                   double alpha,
                                   int64_t seed,
                                   int64_t num_threads) -> VamanaSegmentBuildParams {
  if (r <= 0 || r > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    throw py::value_error("DiskCollection: vamana_R must be in [1, 2**32 - 1]");
  }
  if (l <= 0 || l > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    throw py::value_error("DiskCollection: vamana_L must be in [1, 2**32 - 1]");
  }
  if (l < r) {
    throw py::value_error("DiskCollection: vamana_L must be >= vamana_R");
  }
  if (!is_finite_f64(alpha) || alpha < 1.0) {
    throw py::value_error("DiskCollection: vamana_alpha must be finite and >= 1.0");
  }
  if (alpha > static_cast<double>(std::numeric_limits<float>::max())) {
    throw py::value_error("DiskCollection: vamana_alpha must be representable as finite float32");
  }
  if (seed < 0 || seed > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    throw py::value_error("DiskCollection: vamana_seed must be in [0, 2**32 - 1]");
  }
  const auto resolved_threads = resolve_vamana_num_threads(num_threads);
  if (resolved_threads < 0 ||
      resolved_threads > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    throw py::value_error("DiskCollection: vamana_num_threads must be in [0, 2**32 - 1]");
  }
  VamanaSegmentBuildParams params;
  params.R = static_cast<uint32_t>(r);
  params.L = static_cast<uint32_t>(l);
  params.alpha = static_cast<float>(alpha);
  if (!is_finite_f32(params.alpha)) {
    throw py::value_error("DiskCollection: vamana_alpha must be representable as finite float32");
  }
  params.seed = static_cast<uint32_t>(seed);
  params.num_threads = static_cast<uint32_t>(resolved_threads);
  return params;
}

inline auto require_dtype(const py::array &array,
                          const py::dtype &dtype,
                          const std::string &message) -> void {
  if (!array.dtype().is(dtype)) {
    throw py::type_error(message);
  }
}

class PyDiskCollection {
 public:
  PyDiskCollection(const std::string &path,
                   uint32_t dim,
                   MetricType metric,
                   const std::string &index_type,
                   size_t max_pending_bytes = DiskCollection::kDefaultMaxPendingBytes,
                   int64_t vamana_R = 64,
                   int64_t vamana_L = 100,
                   double vamana_alpha = 1.2,
                   int64_t vamana_seed = 1234,
                   int64_t vamana_num_threads = 0) {
    const auto parsed_index_type = index_type_from_string_strict(index_type);
    VamanaSegmentBuildParams vamana_params;
    if (parsed_index_type == DiskIndexType::Vamana) {
      if (metric != MetricType::L2) {
        throw py::value_error("DiskCollection: disk_vamana metric " + metric_name(metric) +
                              " is not supported; Vamana v1 supports L2 only");
      }
      vamana_params = validate_vamana_params(
          vamana_R, vamana_L, vamana_alpha, vamana_seed, vamana_num_threads);
    }
    impl_ = std::make_unique<DiskCollection>(path,
                                             dim,
                                             metric,
                                             parsed_index_type,
                                             max_pending_bytes,
                                             vamana_params);
  }

  static auto open(const std::string &path) -> std::shared_ptr<PyDiskCollection> {
    const auto manifest =
        CollectionManifest::load(std::filesystem::path(path) / "collection_manifest.txt");
    if (manifest.index_type == DiskIndexType::Laser) {
      throw py::value_error("DiskCollection.open: index_type \"disk_laser\" not implemented in v1");
    }
    auto inner = DiskCollection::open(path);
    return std::shared_ptr<PyDiskCollection>(new PyDiskCollection(std::move(inner)));
  }

  void add(py::array vectors, py::array ids) {
    require_dtype(vectors,
                  py::dtype::of<float>(),
                  "DiskCollection.add: vectors.dtype must be float32");
    require_dtype(ids, py::dtype::of<uint64_t>(), "DiskCollection.add: ids.dtype must be uint64");
    if (vectors.ndim() != 2) {
      throw py::value_error("DiskCollection.add: vectors must be 2D (got ndim=" +
                            std::to_string(vectors.ndim()) + ")");
    }
    if (ids.ndim() != 1) {
      throw py::value_error(
          "DiskCollection.add: ids must be 1D (got ndim=" + std::to_string(ids.ndim()) + ")");
    }
    const auto n_rows = static_cast<uint64_t>(vectors.shape(0));
    const auto v_dim = static_cast<uint64_t>(vectors.shape(1));
    const auto n_ids = static_cast<uint64_t>(ids.shape(0));
    if (v_dim != impl_->dim()) {
      throw py::value_error("DiskCollection.add: vectors.shape[1]=" + std::to_string(v_dim) +
                            " does not match collection dim=" + std::to_string(impl_->dim()));
    }
    if (n_ids != n_rows) {
      throw py::value_error("DiskCollection.add: ids.shape[0]=" + std::to_string(n_ids) +
                            " does not match vectors.shape[0]=" + std::to_string(n_rows));
    }
    if ((vectors.flags() & py::array::c_style) == 0) {
      throw py::type_error("DiskCollection.add: vectors must be C-contiguous");
    }
    if ((ids.flags() & py::array::c_style) == 0) {
      throw py::type_error("DiskCollection.add: ids must be C-contiguous");
    }

    const auto *vector_data = static_cast<const float *>(vectors.data());
    const auto *id_data = static_cast<const uint64_t *>(ids.data());
    {
      py::gil_scoped_release release;
      std::unique_lock<std::shared_mutex> lock(mutex_);
      impl_->add_batch(vector_data, id_data, n_rows);
    }
  }

  void flush() {
    py::gil_scoped_release release;
    std::unique_lock<std::shared_mutex> lock(mutex_);
    impl_->flush();
  }

  auto search(py::array query, int k, int ef) -> std::vector<std::tuple<uint64_t, float>> {
    if (k <= 0) {
      throw py::value_error("DiskCollection.search: k must be > 0 (got " + std::to_string(k) + ")");
    }
    if (ef <= 0) {
      throw py::value_error("DiskCollection.search: ef must be > 0 (got " + std::to_string(ef) +
                            ")");
    }
    require_dtype(query,
                  py::dtype::of<float>(),
                  "DiskCollection.search: query.dtype must be float32");
    if (query.ndim() != 1) {
      throw py::value_error("DiskCollection.search: query must be 1D (got ndim=" +
                            std::to_string(query.ndim()) + ")");
    }
    if (static_cast<uint64_t>(query.shape(0)) != impl_->dim()) {
      throw py::value_error(
          "DiskCollection.search: query.shape[0]=" + std::to_string(query.shape(0)) +
          " does not match collection dim=" + std::to_string(impl_->dim()));
    }
    if ((query.flags() & py::array::c_style) == 0) {
      throw py::type_error("DiskCollection.search: query must be C-contiguous");
    }

    DiskSearchOptions opts;
    opts.top_k = static_cast<uint32_t>(k);
    opts.ef = static_cast<uint32_t>(ef);
    const auto *query_data = static_cast<const float *>(query.data());

    std::vector<DiskSearchHit> hits;
    {
      py::gil_scoped_release release;
      std::shared_lock<std::shared_mutex> lock(mutex_);
      hits = impl_->search(query_data, opts);
    }

    std::vector<std::tuple<uint64_t, float>> out;
    out.reserve(hits.size());
    for (const auto &h : hits) {
      out.emplace_back(h.label, h.distance);
    }
    return out;
  }

  auto size() const -> uint64_t {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return impl_->size();
  }
  auto dim() const -> uint32_t {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return impl_->dim();
  }

 private:
  explicit PyDiskCollection(DiskCollection &&inner)
      : impl_(std::make_unique<DiskCollection>(std::move(inner))) {}

  std::unique_ptr<DiskCollection> impl_;
  mutable std::shared_mutex mutex_;
};

inline void register_disk_collection(py::module_ &m) {
  py::class_<PyDiskCollection, std::shared_ptr<PyDiskCollection>>(m, "DiskCollection")
      .def(py::init<const std::string &,
                    uint32_t,
                    MetricType,
                    const std::string &,
                    size_t,
                    int64_t,
                    int64_t,
                    double,
                    int64_t,
                    int64_t>(),
           py::arg("path"),
           py::arg("dim"),
           py::arg("metric"),
           py::arg("index_type"),
           py::kw_only(),
           py::arg("max_pending_bytes") = DiskCollection::kDefaultMaxPendingBytes,
           py::arg("vamana_R") = 64,
           py::arg("vamana_L") = 100,
           py::arg("vamana_alpha") = 1.2,
           py::arg("vamana_seed") = 1234,
           py::arg("vamana_num_threads") = 0)
      .def_static("open", &PyDiskCollection::open, py::arg("path"))
      .def("add", &PyDiskCollection::add, py::arg("vectors"), py::arg("ids"))
      .def("flush", &PyDiskCollection::flush)
      .def("search",
           &PyDiskCollection::search,
           py::arg("query"),
           py::arg("k") = 10,
           py::arg("ef") = 100,
           // The metric contract docstring is enforced by spec scenario
           // `test_disk_collection_cos_distance_docstring`; the literal phrases
           // below are required.
           "Return the top-k nearest neighbors as a list of (label, distance) tuples.\n\n"
           "Distance semantics (smaller is better):\n"
           "  L2: squared distance (Σ(qi - vi)^2)\n"
           "  IP: negative inner product (-Σ(qi * vi))\n"
           "  COS: negative cosine similarity after L2-normalizing stored vectors and query\n\n"
           "Argument k must be > 0; k > total_count returns total_count results.\n"
           "Argument ef must be > 0; Vamana uses ef as the greedy-search beam size.")
      .def("size", &PyDiskCollection::size)
      .def("dim", &PyDiskCollection::dim);
}

}  // namespace alaya::disk::pybindings
