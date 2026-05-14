// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <pybind11/pybind11.h>

#include "pybind_modules.hpp"

PYBIND11_MODULE(_alayalitepy, m) {
  m.doc() = "AlayaLite";

#ifdef ALAYA_ENABLE_LASER
  // Laser on-disk Quantized Graph index lives under a submodule so its
  // `Index` class does not collide with AlayaLite's top-level `Index`.
  // Accessed from Python as `alayalite._alayalitepy.laser.Index`; the
  // `alayalite.laser` package re-exports it — see
  // python/src/alayalite/laser/__init__.py.
  auto laser_mod = m.def_submodule("laser", "Laser on-disk QG index");
  alaya::pybindings::register_laser(laser_mod);
#endif

  // Vamana graph builder — produces a DiskANN-format .index file.
  // Registered unconditionally; the builder has no Linux-only deps.
  auto vamana_mod = m.def_submodule("vamana", "Vamana graph builder");
  alaya::pybindings::register_vamana(vamana_mod);

  // define version info
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  alaya::pybindings::register_index(m);
  alaya::pybindings::register_disk(m);
}
