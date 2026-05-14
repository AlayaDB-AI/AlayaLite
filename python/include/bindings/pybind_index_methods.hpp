// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <memory>

#include "index.hpp"

namespace py = pybind11;

namespace alaya {

using PyIndexInterfaceClass = py::class_<PyIndexInterface, std::shared_ptr<PyIndexInterface>>;

void add_index_search_methods(PyIndexInterfaceClass &cls);
void add_index_fit_methods(PyIndexInterfaceClass &cls);
void add_index_mutate_methods(PyIndexInterfaceClass &cls);
void add_index_scalar_methods(PyIndexInterfaceClass &cls);
void add_index_hybrid_methods(PyIndexInterfaceClass &cls);
void add_index_io_methods(PyIndexInterfaceClass &cls);

}  // namespace alaya
