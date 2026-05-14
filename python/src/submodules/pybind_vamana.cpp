// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "pybind_modules.hpp"

#include "../alayalite/vamana/_bindings.hpp"

namespace alaya::pybindings {

void register_vamana(py::module_ &m) { vamana::bindings::register_vamana_module(m); }

}  // namespace alaya::pybindings
