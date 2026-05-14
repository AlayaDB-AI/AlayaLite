// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "bindings/pybind_index_methods.hpp"

namespace alaya {

void add_index_io_methods(PyIndexInterfaceClass &cls) {
  cls.def("save",
          &PyIndexInterface::save,
          py::arg("index_path"),
          py::arg("data_path"),
          py::arg("quant_path") = std::string())
      .def("load",
           &PyIndexInterface::load,
           py::arg("index_path"),
           py::arg("data_path"),
           py::arg("quant_path") = std::string());
}

}  // namespace alaya
