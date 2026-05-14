// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "bindings/pybind_index_methods.hpp"

namespace alaya {

void add_index_search_methods(PyIndexInterfaceClass &cls) {
  cls.def("search",
          &PyIndexInterface::search,  //
          py::arg("query"),           //
          py::arg("topk"),            //
          py::arg("ef"))
      .def("get_data_by_id", &PyIndexInterface::get_data_by_id, py::arg("id"))
      .def("get_data_num", &PyIndexInterface::get_data_num)
      .def("batch_search",
           &PyIndexInterface::batch_search,  //
           py::arg("queries"),               //
           py::arg("topk"),                  //
           py::arg("ef"),                    //
           py::arg("num_threads"))           //
      .def("batch_search_with_distance",
           &PyIndexInterface::batch_search_with_distance,  //
           py::arg("queries"),                             //
           py::arg("topk"),                                //
           py::arg("ef"),                                  //
           py::arg("num_threads"));                        //
}

}  // namespace alaya
