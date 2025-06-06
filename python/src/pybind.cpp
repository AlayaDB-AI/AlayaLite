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

#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "index/graph/hnsw/hnsw_builder.hpp"
#include "index/index_type.hpp"
// #include "reg.hpp"
#include "params.hpp"
#include "space/raw_space.hpp"
#include "space/sq8_space.hpp"
#include "utils/metric_type.hpp"

#include "client.hpp"
#include "index.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_alayalitepy, m) {
  m.doc() = "AlayaLite";

  // define version info
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  // enumeral types
  py::enum_<alaya::IndexType>(m, "IndexType")
      .value("FLAT", alaya::IndexType::FLAT)
      .value("HNSW", alaya::IndexType::HNSW)
      .value("NSG", alaya::IndexType::NSG)
      .value("FUSION", alaya::IndexType::FUSION)
      .export_values();

  py::enum_<alaya::MetricType>(m, "MetricType")
      .value("L2", alaya::MetricType::L2)
      .value("IP", alaya::MetricType::IP)
      .value("COS", alaya::MetricType::COS)
      .export_values();

  py::enum_<alaya::QuantizationType>(m, "QuantizationType")
      .value("NONE", alaya::QuantizationType::NONE)
      .value("SQ8", alaya::QuantizationType::SQ8)
      .value("SQ4", alaya::QuantizationType::SQ4)
      .export_values();

  py::class_<alaya::IndexParams>(m, "IndexParams")
      .def(py::init<>())
      .def(py::init<alaya::IndexType, py::dtype, py::dtype, alaya::QuantizationType,
                    alaya::MetricType, uint32_t>(),
           py::arg("index_type_") = alaya::IndexType::HNSW,
           py::arg("data_type_") = py::dtype::of<float>(),
           py::arg("id_type_") = py::dtype::of<uint32_t>(),
           py::arg("quantization_type_") = alaya::QuantizationType::NONE,
           py::arg("metric_") = alaya::MetricType::L2,
           py::arg("capacity_") = py::dtype::of<uint32_t>())
      .def_readwrite("index_type_", &alaya::IndexParams::index_type_)
      .def_readwrite("data_type_", &alaya::IndexParams::data_type_)
      .def_readwrite("id_type_", &alaya::IndexParams::id_type_)
      .def_readwrite("quantization_type_", &alaya::IndexParams::quantization_type_)
      .def_readwrite("metric_", &alaya::IndexParams::metric_)
      .def_readwrite("capacity_", &alaya::IndexParams::capacity_);
  ;

  alaya::IndexParams default_param;

  py::class_<alaya::Client>(m, "Client")
      .def(py::init<>())
      .def("create_index", &alaya::Client::create_index,  //
           py::arg("name"),                               //
           py::arg("param"))
      .def("load_index",                          //
           &alaya::Client::load_index,            //
           py::arg("name"),                       //
           py::arg("param"),                      //
           py::arg("index_path"),                 //
           py::arg("data_path") = std::string(),  //
           py::arg("quant_path") = std::string());

  py::class_<alaya::PyIndexInterface, std::shared_ptr<alaya::PyIndexInterface>>(m,
                                                                                "PyIndexInterface")
      .def(py::init<alaya::IndexParams>(), py::arg("params"))
      .def("to_string", &alaya::PyIndexInterface::to_string)
      .def("fit", &alaya::PyIndexInterface::fit,  //
           py::arg("vectors"),                    //
           py::arg("ef_construction"),            //
           py::arg("num_threads"))
      .def("search", &alaya::PyIndexInterface::search,  //
           py::arg("query"),                            //
           py::arg("topk"),                             //
           py::arg("ef"))
      .def("get_data_by_id", &alaya::PyIndexInterface::get_data_by_id, py::arg("id"))
      .def("insert", &alaya::PyIndexInterface::insert,  //
           py::arg("insert_data"),                      //
           py::arg("ef"))
      .def("remove", &alaya::PyIndexInterface::remove, py::arg("id"))
      .def("batch_search", &alaya::PyIndexInterface::batch_search,                              //
           py::arg("queries"),                                                                  //
           py::arg("topk"),                                                                     //
           py::arg("ef"),                                                                       //
           py::arg("num_threads"))                                                              //
      .def("batch_search_with_distance", &alaya::PyIndexInterface::batch_search_with_distance,  //
           py::arg("queries"),                                                                  //
           py::arg("topk"),                                                                     //
           py::arg("ef"),                                                                       //
           py::arg("num_threads"))                                                              //
      .def("save",                                                                              //
           &alaya::PyIndexInterface::save,                                                      //
           py::arg("index_path"),                                                               //
           py::arg("data_path"),                                                                //
           py::arg("quant_path") = std::string())
      .def("load",                          //
           &alaya::PyIndexInterface::load,  //
           py::arg("index_path"),           //
           py::arg("data_path"),            //
           py::arg("quant_path") = std::string())
      .def("get_data_dim", &alaya::PyIndexInterface::get_data_dim);
}
