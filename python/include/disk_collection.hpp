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
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
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
  // v1 only accepts "disk_flat" at the Python boundary. The C++ format-level
  // round-trip helpers in disk-types accept disk_vamana / disk_laser too, but
  // those are reserved future names and this binding rejects them. The error
  // message embeds the dual literal contract shared with the C++
  // `DiskSegmentFactory` ("disk_<engine>" + "not implemented in v1") so the
  // Python and C++ rejection messages can be matched against the same regex.
  if (s == "disk_flat") {
    return DiskIndexType::Flat;
  }
  throw py::value_error("DiskCollection: index_type \"" + s +
                        "\" not implemented in v1 (must be \"disk_flat\"); " +
                        "disk_vamana and disk_laser are reserved future names");
}

class PyDiskCollection {
 public:
  PyDiskCollection(const std::string &path, uint32_t dim, MetricType metric,
                   const std::string &index_type)
      : impl_(std::make_unique<DiskCollection>(path, dim, metric,
                                               index_type_from_string_strict(index_type))) {}

  static auto open(const std::string &path) -> std::shared_ptr<PyDiskCollection> {
    const auto manifest = CollectionManifest::load(std::filesystem::path(path) /
                                                   "collection_manifest.txt");
    if (manifest.index_type != DiskIndexType::Flat) {
      const std::string engine(index_type_to_string(manifest.index_type));
      throw py::value_error("DiskCollection.open: index_type \"" + engine +
                            "\" not implemented in v1 (must be \"disk_flat\"); " +
                            "disk_vamana exposure is reserved for a follow-up change");
    }
    auto inner = DiskCollection::open(path);
    return std::shared_ptr<PyDiskCollection>(new PyDiskCollection(std::move(inner)));
  }

  void add(py::array vectors, py::array ids) {
    if (!py::isinstance<py::array_t<float>>(vectors)) {
      throw py::type_error("DiskCollection.add: vectors.dtype must be float32");
    }
    if (!py::isinstance<py::array_t<uint64_t>>(ids)) {
      throw py::type_error("DiskCollection.add: ids.dtype must be uint64");
    }
    auto v = py::cast<py::array_t<float>>(vectors);
    auto i = py::cast<py::array_t<uint64_t>>(ids);

    if (v.ndim() != 2) {
      throw py::value_error("DiskCollection.add: vectors must be 2D (got ndim=" +
                            std::to_string(v.ndim()) + ")");
    }
    if (i.ndim() != 1) {
      throw py::value_error("DiskCollection.add: ids must be 1D (got ndim=" +
                            std::to_string(i.ndim()) + ")");
    }
    const auto n_rows = static_cast<uint64_t>(v.shape(0));
    const auto v_dim = static_cast<uint64_t>(v.shape(1));
    const auto n_ids = static_cast<uint64_t>(i.shape(0));
    if (v_dim != impl_->dim()) {
      throw py::value_error("DiskCollection.add: vectors.shape[1]=" + std::to_string(v_dim) +
                            " does not match collection dim=" + std::to_string(impl_->dim()));
    }
    if (n_ids != n_rows) {
      throw py::value_error("DiskCollection.add: ids.shape[0]=" + std::to_string(n_ids) +
                            " does not match vectors.shape[0]=" + std::to_string(n_rows));
    }
    if ((v.flags() & py::array::c_style) == 0) {
      throw py::type_error("DiskCollection.add: vectors must be C-contiguous");
    }
    if ((i.flags() & py::array::c_style) == 0) {
      throw py::type_error("DiskCollection.add: ids must be C-contiguous");
    }

    impl_->add_batch(v.data(), i.data(), n_rows);
  }

  void flush() {
    py::gil_scoped_release release;
    impl_->flush();
  }

  auto search(py::array query, int k) -> std::vector<std::tuple<uint64_t, float>> {
    if (k <= 0) {
      throw py::value_error("DiskCollection.search: k must be > 0 (got " + std::to_string(k) + ")");
    }
    if (!py::isinstance<py::array_t<float>>(query)) {
      throw py::type_error("DiskCollection.search: query.dtype must be float32");
    }
    auto q = py::cast<py::array_t<float>>(query);
    if (q.ndim() != 1) {
      throw py::value_error("DiskCollection.search: query must be 1D (got ndim=" +
                            std::to_string(q.ndim()) + ")");
    }
    if (static_cast<uint64_t>(q.shape(0)) != impl_->dim()) {
      throw py::value_error("DiskCollection.search: query.shape[0]=" +
                            std::to_string(q.shape(0)) +
                            " does not match collection dim=" + std::to_string(impl_->dim()));
    }
    if ((q.flags() & py::array::c_style) == 0) {
      throw py::type_error("DiskCollection.search: query must be C-contiguous");
    }

    DiskSearchOptions opts;
    opts.top_k = static_cast<uint32_t>(k);

    std::vector<DiskSearchHit> hits;
    {
      py::gil_scoped_release release;
      hits = impl_->search(q.data(), opts);
    }

    std::vector<std::tuple<uint64_t, float>> out;
    out.reserve(hits.size());
    for (const auto &h : hits) {
      out.emplace_back(h.label, h.distance);
    }
    return out;
  }

  auto size() const -> uint64_t { return impl_->size(); }
  auto dim() const -> uint32_t { return impl_->dim(); }

 private:
  explicit PyDiskCollection(DiskCollection &&inner)
      : impl_(std::make_unique<DiskCollection>(std::move(inner))) {}

  std::unique_ptr<DiskCollection> impl_;
};

inline void register_disk_collection(py::module_ &m) {
  py::class_<PyDiskCollection, std::shared_ptr<PyDiskCollection>>(m, "DiskCollection")
      .def(py::init<const std::string &, uint32_t, MetricType, const std::string &>(),
           py::arg("path"), py::arg("dim"), py::arg("metric"), py::arg("index_type"))
      .def_static("open", &PyDiskCollection::open, py::arg("path"))
      .def("add", &PyDiskCollection::add, py::arg("vectors"), py::arg("ids"))
      .def("flush", &PyDiskCollection::flush)
      .def("search", &PyDiskCollection::search, py::arg("query"), py::arg("k") = 10,
           // The metric contract docstring is enforced by spec scenario
           // `test_disk_collection_cos_distance_docstring`; the literal phrases
           // below are required.
           "Return the top-k nearest neighbors as a list of (label, distance) tuples.\n\n"
           "Distance semantics (smaller is better):\n"
           "  L2: squared distance (Σ(qi - vi)^2)\n"
           "  IP: negative inner product (-Σ(qi * vi))\n"
           "  COS: negative inner product of L2-normalized pair (cosine similarity\n"
           "       contract — note the negation, so distance ∈ [-1, 1])\n\n"
           "Argument k must be > 0; k > total_count returns total_count results.")
      .def("size", &PyDiskCollection::size)
      .def("dim", &PyDiskCollection::dim);
}

}  // namespace alaya::disk::pybindings
