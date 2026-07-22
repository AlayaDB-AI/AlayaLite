# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# AlayaLite - header-only vector database core with a pybind11 extension module.
#
# cmake-format: off
# This file only orchestrates; each concern lives in cmake/:
#   AlayaConan.cmake        Conan dependency provider registration (must precede project())
#   AlayaOptions.cmake      every user-facing switch + option validation
#   AlayaPreflight.cmake    early, actionable environment checks (compiler floor, Python headers)
#   AlayaToolchain.cmake    C++ standard, ccache, fast linkers, sanitizer detection
#   AlayaFlags.cmake        compile flags as the linkable alaya_build_flags INTERFACE target
#   AlayaDependencies.cmake find_package for all third-party libs (THIRD_PARTY_LIBS)
#   AlayaLaser.cmake        LASER disk-index backend wiring (alaya_laser INTERFACE target)
#   AlayaTesting.cmake      alaya_cc_target / alaya_add_test helpers used by tests/*
# cmake-format: on
#
# Quick start (dev): cmake --preset release && cmake --build --preset release && ctest --preset release The same tree
# also configures with a plain `cmake -B build/Release -DCMAKE_BUILD_TYPE=Release` (CI does), and is driven by
# scikit-build-core for wheel builds (see pyproject.toml). The 3.24 floor comes from the Conan dependency provider
# (CMAKE_PROJECT_TOP_LEVEL_INCLUDES).

cmake_minimum_required(VERSION 3.24...3.31)

include(cmake/AlayaConan.cmake)

project(AlayaLite LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE
  )
endif()

include(cmake/AlayaOptions.cmake)
include(cmake/AlayaPreflight.cmake)
alaya_preflight()

include(cmake/AlayaToolchain.cmake)
include(cmake/AlayaFlags.cmake)
include(cmake/AlayaDependencies.cmake)
include(cmake/AlayaDiskANN.cmake)
include(cmake/AlayaLaser.cmake)
include(cmake/Codecov.cmake)
include(cmake/PrintSummary.cmake)

# ---------------------------------------------------------------------------------------------------------------------
# The core library: header-only, so an INTERFACE target carrying the include tree and the third-party link surface.
# ---------------------------------------------------------------------------------------------------------------------
add_library(AlayaLite INTERFACE)
target_include_directories(
  AlayaLite INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)
target_link_libraries(AlayaLite INTERFACE ${THIRD_PARTY_LIBS} alaya_diskann)
target_compile_features(AlayaLite INTERFACE cxx_std_20)

if(BUILD_TESTING)
  include(CTest)
  find_package(GTest REQUIRED)
  set(GTEST_LIBS GTest::gtest GTest::gtest_main)
  include(cmake/AlayaTesting.cmake)
  add_subdirectory(tests)
endif()

if(BUILD_PYTHON)
  add_subdirectory(python)
endif()

print_build_summary()
