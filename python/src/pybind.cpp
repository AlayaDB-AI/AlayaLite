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
#include "utils/metadata_filter.hpp"
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
      .value("RABITQ", alaya::QuantizationType::RABITQ)
      .export_values();

  // Filter enums and classes for hybrid search
  py::enum_<alaya::FilterOp>(m, "FilterOp")
      .value("EQ", alaya::FilterOp::EQ)
      .value("NE", alaya::FilterOp::NE)
      .value("GT", alaya::FilterOp::GT)
      .value("GE", alaya::FilterOp::GE)
      .value("LT", alaya::FilterOp::LT)
      .value("LE", alaya::FilterOp::LE)
      .value("IN", alaya::FilterOp::IN)
      .value("NOT_IN", alaya::FilterOp::NOT_IN)
      .value("CONTAINS", alaya::FilterOp::CONTAINS)
      .export_values();

  py::enum_<alaya::LogicOp>(m, "LogicOp")
      .value("AND", alaya::LogicOp::AND)
      .value("OR", alaya::LogicOp::OR)
      .value("NOT", alaya::LogicOp::NOT)
      .export_values();

  py::class_<alaya::FilterCondition>(m, "FilterCondition")
      .def(py::init<>())
      .def_readwrite("field", &alaya::FilterCondition::field)
      .def_readwrite("op", &alaya::FilterCondition::op)
      .def_readwrite("value", &alaya::FilterCondition::value)
      .def_readwrite("values", &alaya::FilterCondition::values);

  py::class_<alaya::MetadataFilter>(m, "MetadataFilter")
      .def(py::init<>())
      .def_readwrite("logic_op", &alaya::MetadataFilter::logic_op)
      .def_readwrite("conditions", &alaya::MetadataFilter::conditions)
      .def("is_empty", &alaya::MetadataFilter::is_empty)
      .def("add_eq", &alaya::MetadataFilter::add_eq, py::arg("field"), py::arg("value"))
      .def("add_gt", &alaya::MetadataFilter::add_gt, py::arg("field"), py::arg("value"))
      .def("add_lt", &alaya::MetadataFilter::add_lt, py::arg("field"), py::arg("value"))
      .def("add_in", &alaya::MetadataFilter::add_in, py::arg("field"), py::arg("values"))
      .def("add_sub_filter", &alaya::MetadataFilter::add_sub_filter, py::arg("sub_filter"));

  py::class_<alaya::IndexParams>(m, "IndexParams")
      .def(py::init<>())
      .def(py::init<alaya::IndexType,
                    py::dtype,
                    py::dtype,
                    alaya::QuantizationType,
                    alaya::MetricType,
                    uint32_t,
                    uint32_t,
                    std::string,
                    bool>(),
           py::arg("index_type_") = alaya::IndexType::HNSW,
           py::arg("data_type_") = py::dtype::of<float>(),
           py::arg("id_type_") = py::dtype::of<uint32_t>(),
           py::arg("quantization_type_") = alaya::QuantizationType::NONE,
           py::arg("metric_") = alaya::MetricType::L2,
           py::arg("capacity_") = py::dtype::of<uint32_t>(),
           py::arg("max_nbrs_") = 32,
           py::arg("rocksdb_path_") = "",
           py::arg("has_scalar_data_") = false)
      .def_readwrite("index_type_", &alaya::IndexParams::index_type_)
      .def_readwrite("data_type_", &alaya::IndexParams::data_type_)
      .def_readwrite("id_type_", &alaya::IndexParams::id_type_)
      .def_readwrite("quantization_type_", &alaya::IndexParams::quantization_type_)
      .def_readwrite("metric_", &alaya::IndexParams::metric_)
      .def_readwrite("capacity_", &alaya::IndexParams::capacity_)
      .def_readwrite("rocksdb_path_", &alaya::IndexParams::rocksdb_path_)
      .def_readwrite("has_scalar_data_", &alaya::IndexParams::has_scalar_data_);

  alaya::IndexParams default_param;

  py::class_<alaya::Client>(m, "Client")
      .def(py::init<>())
      .def("create_index",
           &alaya::Client::create_index,  //
           py::arg("name"),               //
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
      .def("fit",
           &alaya::PyIndexInterface::fit,
           py::arg("vectors"),
           py::arg("ef_construction"),
           py::arg("num_threads"),
           py::arg("item_ids") = py::none(),
           py::arg("documents") = py::none(),
           py::arg("metadata_list") = py::none())
      .def("search",
           &alaya::PyIndexInterface::search,  //
           py::arg("query"),                  //
           py::arg("topk"),                   //
           py::arg("ef"))
      .def("get_data_by_id", &alaya::PyIndexInterface::get_data_by_id, py::arg("id"))
      .def("get_data_num", &alaya::PyIndexInterface::get_data_num)
      .def("insert",
           &alaya::PyIndexInterface::insert,
           py::arg("insert_data"),
           py::arg("ef"),
           py::arg("item_id") = py::none(),
           py::arg("document") = "",
           py::arg("metadata") = py::dict())
      .def("remove", &alaya::PyIndexInterface::remove, py::arg("id"))
      .def("remove_by_item_id", &alaya::PyIndexInterface::remove_by_item_id, py::arg("item_id"))
      .def("contains", &alaya::PyIndexInterface::contains, py::arg("item_id"))
      .def("get_scalar_data_by_item_id",
           &alaya::PyIndexInterface::get_scalar_data_by_item_id,
           py::arg("item_id"))
      .def("get_scalar_data_by_internal_id",
           &alaya::PyIndexInterface::get_scalar_data_by_internal_id,
           py::arg("internal_id"))
      .def("filter_query",
           &alaya::PyIndexInterface::filter_query,
           py::arg("filter"),
           py::arg("limit"),
           "Query records by metadata filter without vector search")
      .def("batch_search",
           &alaya::PyIndexInterface::batch_search,  //
           py::arg("queries"),                      //
           py::arg("topk"),                         //
           py::arg("ef"),                           //
           py::arg("num_threads"))                  //
      .def("batch_search_with_distance",
           &alaya::PyIndexInterface::batch_search_with_distance,  //
           py::arg("queries"),                                    //
           py::arg("topk"),                                       //
           py::arg("ef"),                                         //
           py::arg("num_threads"))                                //
      .def("save",                                                //
           &alaya::PyIndexInterface::save,                        //
           py::arg("index_path"),                                 //
           py::arg("data_path"),                                  //
           py::arg("quant_path") = std::string())
      .def("load",                          //
           &alaya::PyIndexInterface::load,  //
           py::arg("index_path"),           //
           py::arg("data_path"),            //
           py::arg("quant_path") = std::string())
      .def("get_data_dim", &alaya::PyIndexInterface::get_data_dim)
      .def("hybrid_search",
           &alaya::PyIndexInterface::hybrid_search,
           py::arg("query"),
           py::arg("topk"),
           py::arg("ef"),
           py::arg("filter"))
      .def("batch_hybrid_search",
           &alaya::PyIndexInterface::batch_hybrid_search,
           py::arg("queries"),
           py::arg("topk"),
           py::arg("ef"),
           py::arg("filter"),
           py::arg("num_threads"))
      .def("close_db", &alaya::PyIndexInterface::close_db, "Close and release RocksDB resources");
}
