// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "bindings/pybind_index_methods.hpp"
#include "pybind_modules.hpp"

#include "client.hpp"
#include "index/index_type.hpp"
#include "params.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/metric_type.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace alaya::pybindings {

void register_index(py::module_ &m) {
  // enumeral types
  py::enum_<IndexType>(m, "IndexType")
      .value("FLAT", IndexType::FLAT)
      .value("HNSW", IndexType::HNSW)
      .value("NSG", IndexType::NSG)
      .value("FUSION", IndexType::FUSION)
      .export_values();

  py::enum_<MetricType>(m, "MetricType")
      .value("L2", MetricType::L2)
      .value("IP", MetricType::IP)
      .value("COS", MetricType::COS)
      .export_values();

  py::enum_<QuantizationType>(m, "QuantizationType")
      .value("NONE", QuantizationType::NONE)
      .value("SQ8", QuantizationType::SQ8)
      .value("SQ4", QuantizationType::SQ4)
      .value("RABITQ", QuantizationType::RABITQ)
      .export_values();

  // Filter enums and classes for hybrid search
  py::enum_<FilterOp>(m, "FilterOp")
      .value("EQ", FilterOp::EQ)
      .value("NE", FilterOp::NE)
      .value("GT", FilterOp::GT)
      .value("GE", FilterOp::GE)
      .value("LT", FilterOp::LT)
      .value("LE", FilterOp::LE)
      .value("IN", FilterOp::IN_SET)
      .value("NOT_IN", FilterOp::NOT_IN_SET)
      .value("CONTAINS", FilterOp::CONTAINS)
      .export_values();

  py::enum_<LogicOp>(m, "LogicOp")
      .value("AND", LogicOp::AND)
      .value("OR", LogicOp::OR)
      .value("NOT", LogicOp::NOT)
      .export_values();

  py::class_<FilterCondition>(m, "FilterCondition")
      .def(py::init<>())
      .def_readwrite("field", &FilterCondition::field)
      .def_readwrite("op", &FilterCondition::op)
      .def_readwrite("value", &FilterCondition::value)
      .def_readwrite("values", &FilterCondition::values);

  py::class_<MetadataFilter>(m, "MetadataFilter")
      .def(py::init<>())
      .def_readwrite("logic_op", &MetadataFilter::logic_op)
      .def_readwrite("conditions", &MetadataFilter::conditions)
      .def("is_empty", &MetadataFilter::is_empty)
      .def("add_eq", &MetadataFilter::add_eq, py::arg("field"), py::arg("value"))
      .def("add_gt", &MetadataFilter::add_gt, py::arg("field"), py::arg("value"))
      .def("add_ge", &MetadataFilter::add_ge, py::arg("field"), py::arg("value"))
      .def("add_lt", &MetadataFilter::add_lt, py::arg("field"), py::arg("value"))
      .def("add_le", &MetadataFilter::add_le, py::arg("field"), py::arg("value"))
      .def("add_in", &MetadataFilter::add_in, py::arg("field"), py::arg("values"))
      .def("add_sub_filter", &MetadataFilter::add_sub_filter, py::arg("sub_filter"));

  py::class_<IndexParams>(m, "IndexParams")
      .def(py::init<>())
      .def(py::init<IndexType,
                    py::dtype,
                    py::dtype,
                    QuantizationType,
                    MetricType,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    uint32_t,
                    std::string,
                    bool,
                    std::vector<std::string>>(),
           py::arg("index_type_") = IndexType::HNSW,
           py::arg("data_type_") = py::dtype::of<float>(),
           py::arg("id_type_") = py::dtype::of<uint32_t>(),
           py::arg("quantization_type_") = QuantizationType::NONE,
           py::arg("metric_") = MetricType::L2,
           py::arg("capacity_") = 100000,
           py::arg("max_nbrs_") = 32,
           py::arg("build_threads_") = 0,
           py::arg("materialized_view_build_threads_") = 0,
           py::arg("rocksdb_path_") = "",
           py::arg("has_scalar_data_") = false,
           py::arg("indexed_fields_") = std::vector<std::string>{})
      .def_readwrite("index_type_", &IndexParams::index_type_)
      .def_readwrite("data_type_", &IndexParams::data_type_)
      .def_readwrite("id_type_", &IndexParams::id_type_)
      .def_readwrite("quantization_type_", &IndexParams::quantization_type_)
      .def_readwrite("metric_", &IndexParams::metric_)
      .def_readwrite("capacity_", &IndexParams::capacity_)
      .def_readwrite("max_nbrs_", &IndexParams::max_nbrs_)
      .def_readwrite("build_threads_", &IndexParams::build_threads_)
      .def_readwrite("materialized_view_build_threads_",
                     &IndexParams::materialized_view_build_threads_)
      .def_readwrite("rocksdb_path_", &IndexParams::rocksdb_path_)
      .def_readwrite("has_scalar_data_", &IndexParams::has_scalar_data_)
      .def_readwrite("indexed_fields_", &IndexParams::indexed_fields_);

  py::class_<Client>(m, "Client")
      .def(py::init<>())
      .def("create_index",
           &Client::create_index,  //
           py::arg("name"),        //
           py::arg("param"))
      .def("load_index",                          //
           &Client::load_index,                   //
           py::arg("name"),                       //
           py::arg("param"),                      //
           py::arg("index_path"),                 //
           py::arg("data_path") = std::string(),  //
           py::arg("quant_path") = std::string());

  PyIndexInterfaceClass cls(m, "PyIndexInterface");
  cls.def(py::init<IndexParams>(), py::arg("params"))
      .def("to_string", &PyIndexInterface::to_string)
      .def("has_scalar_data", &PyIndexInterface::has_scalar_data)
      .def("get_data_dim", &PyIndexInterface::get_data_dim);

  add_index_search_methods(cls);
  add_index_fit_methods(cls);
  add_index_mutate_methods(cls);
  add_index_scalar_methods(cls);
  add_index_hybrid_methods(cls);
  add_index_io_methods(cls);
}

}  // namespace alaya::pybindings
