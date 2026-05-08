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

// Laser (alaya::laser) pybind11 registration.
//
// Exposes a single entry point `register_laser_module(py::module_&)` that
// attaches the `Index` class to the given pybind11 module. Called from
// python/src/pybind.cpp to register the Laser submodule of _alayalitepy.

#pragma once

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"

namespace alaya::laser::bindings {

namespace py = pybind11;
using py_float_array = py::array_t<float, py::array::c_style | py::array::forcecast>;
using py_uint_array = py::array_t<uint32_t, py::array::c_style | py::array::forcecast>;

namespace detail {
inline void get_arr_shape(const py::buffer_info &buffer, size_t &rows, size_t &cols) {
  if (buffer.ndim != 2 && buffer.ndim != 1) {
    throw py::value_error("Input data must be a 1D or 2D array, got ndim=" +
                          std::to_string(buffer.ndim));
  }
  if (buffer.ndim == 2) {
    rows = buffer.shape[0];
    cols = buffer.shape[1];
  } else {
    rows = 1;
    cols = buffer.shape[0];
  }
}
}  // namespace detail

struct Index {
  std::unique_ptr<alaya::laser::QuantizedGraph> index = nullptr;

  explicit Index(const std::string &index_type,
                 const std::string &metric,
                 size_t num_points,
                 size_t main_dim,
                 size_t dim,
                 size_t degree,
                 uint64_t rotator_seed = 0,
                 const std::string &rotator_dump_path = "") {
    if (metric != "l2") {
      throw py::value_error("Laser Index: only metric='l2' is supported, got '" + metric + "'");
    }
    if (degree < 32 || degree % 32 != 0) {
      throw py::value_error("Laser Index: degree_bound must be a positive multiple of 32, got " +
                            std::to_string(degree));
    }
    if (index_type != "QG") {
      throw py::value_error("Laser Index: only index_type='QG' is supported, got '" + index_type +
                            "'");
    }
    index = std::make_unique<alaya::laser::QuantizedGraph>(num_points,
                                                           degree,
                                                           main_dim,
                                                           dim,
                                                           rotator_seed,
                                                           rotator_dump_path);
  }

  void load(const std::string &filename, float search_DRAM_budget) const {
    index->load_disk_index(filename.c_str(), search_DRAM_budget);
  }

  void set_params(size_t ef_search, size_t num_threads, int beam_width) const {
    index->set_params(ef_search, num_threads, beam_width);
  }

  void build_index(const std::string &vamana_file,
                   const std::string &data_file,
                   size_t ef_indexing = 200,
                   size_t num_iter = 3,
                   size_t num_threads = UINT_MAX) const {
    std::cout << "vamana_file: " << vamana_file << ", data_file: " << data_file << std::endl;
    alaya::laser::QGBuilder builder(*index, ef_indexing, num_threads);
    builder.build(vamana_file.c_str(), data_file.c_str());
    std::cout << "\tQuantizedGraph created\n";
  }

  auto search(py_float_array &query, uint32_t knn) const {
    py_uint_array result(knn);
    auto *result_ptr = static_cast<uint32_t *>(result.request().ptr);
    index->search(query.data(0), knn, result_ptr);
    return result;
  }

  auto batch_search(py::object &data, uint32_t knn) const {
    py::array_t<float, py::array::c_style | py::array::forcecast> queries(data);
    auto buffer = queries.request();
    size_t num = 0;
    size_t dim = 0;
    detail::get_arr_shape(buffer, num, dim);
    py_uint_array result({num, static_cast<size_t>(knn)});
    auto *result_ptr = static_cast<uint32_t *>(result.request().ptr);
    index->batch_search(queries.data(), knn, result_ptr, num);
    return result;
  }
};

// Attach the Laser `Index` class to the given module. Call this from
// PYBIND11_MODULE(...) to expose the Laser API under any chosen namespace —
// a standalone module or a submodule of the main pybind binding.
inline void register_laser_module(py::module_ &m) {
  m.doc() =
      R"pbdoc(Laser on-disk Quantized Graph index (ported from symqg into alaya::laser).)pbdoc";

  py::class_<Index>(m, "Index")
      .def(py::init<const std::string &,
                    const std::string &,
                    size_t,
                    size_t,
                    size_t,
                    size_t,
                    uint64_t,
                    const std::string &>(),
           py::arg("index_type"),
           py::arg("metric"),
           py::arg("num_elements"),
           py::arg("main_dimension"),
           py::arg("dimension"),
           py::arg("degree_bound") = 32,
           py::arg("rotator_seed") = 0,
           py::arg("rotator_dump_path") = "")
      .def("load", &Index::load, py::arg("filename"), py::arg("search_DRAM_budget"))
      .def("set_params",
           &Index::set_params,
           py::arg("ef_search") = 200,
           py::arg("num_threads") = 48,
           py::arg("beam_width") = 16)
      .def("build_index",
           &Index::build_index,
           py::arg("vamana_file"),
           py::arg("data_file"),
           py::arg("EF") = 200,
           py::arg("num_iter") = 3,
           py::arg("num_thread") = UINT_MAX)
      .def("search", &Index::search, py::arg("query"), py::arg("k"))
      .def("batch_search", &Index::batch_search, py::arg("queries"), py::arg("k"));
}

}  // namespace alaya::laser::bindings
