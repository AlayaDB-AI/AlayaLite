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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

// Laser port pulls <omp.h> transitively via qg.hpp. Include this before
// AlayaLite's own QGBuilder header (via index.hpp) so `omp_set_num_threads`
// is visible at the point AlayaLite's template definition is parsed — the
// pre-existing `include/index/graph/qg/qg_builder.hpp` uses OMP calls
// without explicitly including <omp.h>.
#ifdef ALAYA_ENABLE_LASER
  #include "alayalite/laser/_bindings.hpp"
#endif

// Vamana builder bindings. Not gated on ALAYA_ENABLE_LASER because the
// builder is header-only (no libaio, no Linux-only dependencies) and the
// spec requires `from alayalite import vamana` to succeed on non-Laser
// builds as well.
#include "alayalite/vamana/_bindings.hpp"

#include "index/graph/hnsw/hnsw_builder.hpp"
#include "index/index_type.hpp"
// #include "reg.hpp"
#include "params.hpp"
#include "space/raw_space.hpp"
#include "space/sq8_space.hpp"
#include "utils/metadata_filter.hpp"
#include "utils/metric_type.hpp"

#include "client.hpp"
#include "disk_collection.hpp"
#include "index.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_alayalitepy, m) {
  m.doc() = "AlayaLite";

#ifdef ALAYA_ENABLE_LASER
  // Laser on-disk Quantized Graph index lives under a submodule so its
  // `Index` class does not collide with AlayaLite's top-level `Index`.
  // Accessed from Python as `alayalite._alayalitepy.laser.Index`; the
  // `alayalite.laser` package re-exports it — see
  // python/src/alayalite/laser/__init__.py.
  auto laser_mod = m.def_submodule("laser", "Laser on-disk QG index");
  alaya::laser::bindings::register_laser_module(laser_mod);
#endif

  // Vamana graph builder — produces a DiskANN-format .index file.
  // Registered unconditionally; the builder has no Linux-only deps.
  auto vamana_mod = m.def_submodule("vamana", "Vamana graph builder");
  alaya::vamana::bindings::register_vamana_module(vamana_mod);

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
      .def("add_ge", &alaya::MetadataFilter::add_ge, py::arg("field"), py::arg("value"))
      .def("add_lt", &alaya::MetadataFilter::add_lt, py::arg("field"), py::arg("value"))
      .def("add_le", &alaya::MetadataFilter::add_le, py::arg("field"), py::arg("value"))
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
                    uint32_t,
                    uint32_t,
                    std::string,
                    bool,
                    std::vector<std::string>>(),
           py::arg("index_type_") = alaya::IndexType::HNSW,
           py::arg("data_type_") = py::dtype::of<float>(),
           py::arg("id_type_") = py::dtype::of<uint32_t>(),
           py::arg("quantization_type_") = alaya::QuantizationType::NONE,
           py::arg("metric_") = alaya::MetricType::L2,
           py::arg("capacity_") = 100000,
           py::arg("max_nbrs_") = 32,
           py::arg("build_threads_") = 0,
           py::arg("materialized_view_build_threads_") = 0,
           py::arg("rocksdb_path_") = "",
           py::arg("has_scalar_data_") = false,
           py::arg("indexed_fields_") = std::vector<std::string>{})
      .def_readwrite("index_type_", &alaya::IndexParams::index_type_)
      .def_readwrite("data_type_", &alaya::IndexParams::data_type_)
      .def_readwrite("id_type_", &alaya::IndexParams::id_type_)
      .def_readwrite("quantization_type_", &alaya::IndexParams::quantization_type_)
      .def_readwrite("metric_", &alaya::IndexParams::metric_)
      .def_readwrite("capacity_", &alaya::IndexParams::capacity_)
      .def_readwrite("max_nbrs_", &alaya::IndexParams::max_nbrs_)
      .def_readwrite("build_threads_", &alaya::IndexParams::build_threads_)
      .def_readwrite("materialized_view_build_threads_",
                     &alaya::IndexParams::materialized_view_build_threads_)
      .def_readwrite("rocksdb_path_", &alaya::IndexParams::rocksdb_path_)
      .def_readwrite("has_scalar_data_", &alaya::IndexParams::has_scalar_data_)
      .def_readwrite("indexed_fields_", &alaya::IndexParams::indexed_fields_);

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
      .def("has_scalar_data", &alaya::PyIndexInterface::has_scalar_data)
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
      .def("upsert",
           &alaya::PyIndexInterface::upsert,
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
      .def("batch_get_scalar_data_by_internal_ids",
           &alaya::PyIndexInterface::batch_get_scalar_data_by_internal_ids,
           py::arg("internal_ids"),
           "Batch get scalar data by internal IDs using RocksDB MultiGet")
      .def("batch_get_item_ids_by_internal_ids",
           &alaya::PyIndexInterface::batch_get_item_ids_by_internal_ids,
           py::arg("internal_ids"),
           "Batch get item_ids by internal IDs using RocksDB MultiGet")
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
      .def("get_materialized_view_partition_count",
           &alaya::PyIndexInterface::get_materialized_view_partition_count)
      .def(
          "hybrid_search",
          [](alaya::PyIndexInterface &self,
             py::array &query,
             uint32_t topk,
             uint32_t ef,
             const alaya::MetadataFilter &filter,
             bool bf,
             const std::string &filter_execution_hint) {
            return self.hybrid_search(query, topk, ef, filter, bf, filter_execution_hint);
          },
          py::arg("query"),
          py::arg("topk"),
          py::arg("ef"),
          py::arg("filter"),
          py::arg("bf") = false,
          py::arg("filter_execution_hint") = std::string())
      .def(
          "batch_hybrid_search",
          [](alaya::PyIndexInterface &self,
             py::array &queries,
             uint32_t topk,
             uint32_t ef,
             const alaya::MetadataFilter &filter,
             uint32_t num_threads,
             bool bf,
             const std::string &filter_execution_hint) {
            return self.batch_hybrid_search(queries,
                                            topk,
                                            ef,
                                            filter,
                                            num_threads,
                                            bf,
                                            filter_execution_hint);
          },
          py::arg("queries"),
          py::arg("topk"),
          py::arg("ef"),
          py::arg("filter"),
          py::arg("num_threads"),
          py::arg("bf") = false,
          py::arg("filter_execution_hint") = std::string())
      .def("close_db", &alaya::PyIndexInterface::close_db, "Close and release RocksDB resources");

  // alayalite.DiskCollection — disk-resident segmented collection (v1: Flat).
  alaya::disk::pybindings::register_disk_collection(m);
}
