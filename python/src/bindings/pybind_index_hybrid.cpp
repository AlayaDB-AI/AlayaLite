// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "bindings/pybind_index_methods.hpp"

namespace alaya {

void add_index_hybrid_methods(PyIndexInterfaceClass &cls) {
  cls.def("hybrid_search",
          &PyIndexInterface::hybrid_search,
          py::arg("query"),
          py::arg("topk"),
          py::arg("ef"),
          py::arg("filter"),
          py::arg("bf") = false,
          py::arg("filter_execution_hint") = std::string())
      .def("batch_hybrid_search",
           &PyIndexInterface::batch_hybrid_search,
           py::arg("queries"),
           py::arg("topk"),
           py::arg("ef"),
           py::arg("filter"),
           py::arg("num_threads"),
           py::arg("bf") = false,
           py::arg("filter_execution_hint") = std::string());
}

}  // namespace alaya
