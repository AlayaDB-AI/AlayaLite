# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# _alayalitepy - the pybind11 extension module, plus the dispatch-header codegen step that feeds it.

set(PYTHON_MODULE_NAME _alayalitepy)

# ---------------------------------------------------------------------------------------------------------------------
# Codegen: dispatch.yaml -> dispatch_generated.hpp + the Python test matrix. Outputs are committed to the source tree on
# purpose (a source checkout builds without running codegen), so the custom command only fires when an input is newer.
# The aggregating target is named alaya_codegen because plain "codegen" is a reserved target name from CMake 3.31 on
# (policy CMP0171).
# ---------------------------------------------------------------------------------------------------------------------
set(ALAYA_CODEGEN_DIR ${CMAKE_SOURCE_DIR}/tools/codegen)
set(ALAYA_CODEGEN_INPUTS
    ${ALAYA_CODEGEN_DIR}/dispatch.yaml ${ALAYA_CODEGEN_DIR}/gen.py
    ${ALAYA_CODEGEN_DIR}/templates/dispatch_factory.hpp.j2 ${ALAYA_CODEGEN_DIR}/templates/test_matrix_params.py.j2
)
set(ALAYA_CODEGEN_OUTPUTS ${CMAKE_SOURCE_DIR}/python/include/dispatch_generated.hpp
                          ${CMAKE_SOURCE_DIR}/python/tests/client/_dispatch_matrix_params.py
)

add_custom_command(
  OUTPUT ${ALAYA_CODEGEN_OUTPUTS}
  COMMAND ${Python_EXECUTABLE} ${ALAYA_CODEGEN_DIR}/gen.py
  DEPENDS ${ALAYA_CODEGEN_INPUTS}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMENT "Generating python dispatch header from dispatch.yaml"
  VERBATIM
)

add_custom_target(alaya_codegen DEPENDS ${ALAYA_CODEGEN_OUTPUTS})

# ---------------------------------------------------------------------------------------------------------------------
# The module. index_factory.cpp owns the dispatch-tree template instantiations so pybind.cpp doesn't pay for
# re-instantiating 33 PyIndex<...> specializations. pybind11 resolves Python through FindPython (PYBIND11_FINDPYTHON=ON
# set before find_package(pybind11) in AlayaDependencies.cmake), i.e. the same interpreter as everything else.
# ---------------------------------------------------------------------------------------------------------------------
pybind11_add_module(
  ${PYTHON_MODULE_NAME}
  MODULE
  NO_EXTRAS
  src/pybind.cpp
  src/diskann_bindings.cpp
  src/index_factory.cpp
)
add_dependencies(${PYTHON_MODULE_NAME} alaya_codegen)
target_include_directories(${PYTHON_MODULE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE AlayaLite pybind11::module)
alaya_apply_build_flags(${PYTHON_MODULE_NAME})
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(${PYTHON_MODULE_NAME} PRIVATE -Wno-nonnull)
endif()

# Precompile pybind11 + Eigen + STL headers. _alayalitepy is a single-TU module whose index/dispatch-generated templates
# drag in pybind11/numpy.h, Eigen, and the bulk of <vector>/<string>/<memory>/<unordered_map> hundreds of times through
# template instantiation. A PCH cuts the parse+template-instantiation cost on the recurring header surface and is the
# single biggest lever for this TU's compile time.
target_precompile_headers(
  ${PYTHON_MODULE_NAME}
  PRIVATE
  <pybind11/pybind11.h>
  <pybind11/numpy.h>
  <pybind11/stl.h>
  <pybind11/cast.h>
  <pybind11/pytypes.h>
  <Eigen/Core>
  <vector>
  <string>
  <memory>
  <unordered_map>
)

# The Laser on-disk QG pybind is compiled as a submodule of _alayalitepy. SIMD-dependent flags (-march=native, etc.) are
# applied PRIVATE so they don't leak to anything else that might link the Laser INTERFACE target downstream.
if(ALAYA_ENABLE_LASER)
  target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE alaya_laser)
  target_compile_definitions(${PYTHON_MODULE_NAME} PRIVATE ALAYA_ENABLE_LASER=1)
  target_compile_options(${PYTHON_MODULE_NAME} PRIVATE ${_ALAYA_LASER_CONSUMER_OPTIONS})
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC 11 can ICE while auto-vectorizing the LASER-enabled pybind TU. The handwritten LASER kernels still use target
    # attributes; this only disables generic tree vectorization for the binding translation unit.
    target_compile_options(${PYTHON_MODULE_NAME} PRIVATE -fno-tree-vectorize)
  endif()
endif()

install(
  TARGETS ${PYTHON_MODULE_NAME}
  LIBRARY DESTINATION alayalite
  RUNTIME DESTINATION alayalite
)
