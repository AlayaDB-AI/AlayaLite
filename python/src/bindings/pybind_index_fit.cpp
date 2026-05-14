// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "bindings/pybind_index_methods.hpp"

namespace alaya {

void add_index_fit_methods(PyIndexInterfaceClass &cls) {
  cls.def("fit",
          &PyIndexInterface::fit,
          py::arg("vectors"),
          py::arg("ef_construction"),
          py::arg("num_threads"),
          py::arg("item_ids") = py::none(),
          py::arg("documents") = py::none(),
          py::arg("metadata_list") = py::none());
}

}  // namespace alaya
