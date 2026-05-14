// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "bindings/pybind_index_methods.hpp"

namespace alaya {

void add_index_scalar_methods(PyIndexInterfaceClass &cls) {
  cls.def("get_scalar_data_by_item_id",
          &PyIndexInterface::get_scalar_data_by_item_id,
          py::arg("item_id"))
      .def("get_scalar_data_by_internal_id",
           &PyIndexInterface::get_scalar_data_by_internal_id,
           py::arg("internal_id"))
      .def("batch_get_scalar_data_by_internal_ids",
           &PyIndexInterface::batch_get_scalar_data_by_internal_ids,
           py::arg("internal_ids"),
           "Batch get scalar data by internal IDs using RocksDB MultiGet")
      .def("batch_get_item_ids_by_internal_ids",
           &PyIndexInterface::batch_get_item_ids_by_internal_ids,
           py::arg("internal_ids"),
           "Batch get item_ids by internal IDs using RocksDB MultiGet")
      .def("filter_query",
           &PyIndexInterface::filter_query,
           py::arg("filter"),
           py::arg("limit"),
           "Query records by metadata filter without vector search")
      .def("get_materialized_view_partition_count",
           &PyIndexInterface::get_materialized_view_partition_count);
}

}  // namespace alaya
