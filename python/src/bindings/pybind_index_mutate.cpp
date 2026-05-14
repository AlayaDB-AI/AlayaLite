// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "bindings/pybind_index_methods.hpp"

namespace alaya {

void add_index_mutate_methods(PyIndexInterfaceClass &cls) {
  cls.def("insert",
          &PyIndexInterface::insert,
          py::arg("insert_data"),
          py::arg("ef"),
          py::arg("item_id") = py::none(),
          py::arg("document") = "",
          py::arg("metadata") = py::dict())
      .def("upsert",
           &PyIndexInterface::upsert,
           py::arg("insert_data"),
           py::arg("ef"),
           py::arg("item_id") = py::none(),
           py::arg("document") = "",
           py::arg("metadata") = py::dict())
      .def("remove", &PyIndexInterface::remove, py::arg("id"))
      .def("remove_by_item_id", &PyIndexInterface::remove_by_item_id, py::arg("item_id"))
      .def("contains", &PyIndexInterface::contains, py::arg("item_id"))
      .def("close_db", &PyIndexInterface::close_db, "Close and release RocksDB resources");
}

}  // namespace alaya
