// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "pybind_modules.hpp"

#include "disk_collection.hpp"

namespace alaya::pybindings {

void register_disk(py::module_ &m) { disk::pybindings::register_disk_collection(m); }

}  // namespace alaya::pybindings
