#pragma once

#include <pybind11/pybind11.h>

namespace alaya::diskann::pybindings {

void register_diskann(pybind11::module_ &module);

}  // namespace alaya::diskann::pybindings
