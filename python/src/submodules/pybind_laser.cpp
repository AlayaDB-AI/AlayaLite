// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "pybind_modules.hpp"

#include "../alayalite/laser/_bindings.hpp"

namespace alaya::pybindings {

void register_laser(py::module_ &m) { laser::bindings::register_laser_module(m); }

}  // namespace alaya::pybindings
