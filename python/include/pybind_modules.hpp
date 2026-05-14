// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace alaya::pybindings {

// These registration functions are intentionally implemented in separate TUs
// (pybind_index.cpp, pybind_disk.cpp, pybind_vamana.cpp, pybind_laser.cpp)
// so ninja can compile them in parallel. The header-only index/disk/laser/vamana
// headers are heavy with template instantiations — keeping them out of pybind.cpp
// is what makes the parallel split worthwhile. Do not merge these back.
void register_index(py::module_ &m);
void register_disk(py::module_ &m);
void register_vamana(py::module_ &m);

#ifdef ALAYA_ENABLE_LASER
void register_laser(py::module_ &m);
#endif

}  // namespace alaya::pybindings
