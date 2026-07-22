# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# AlayaDependencies.cmake - resolve every third-party package and aggregate them into THIRD_PARTY_LIBS.
#
# Packages resolve through the Conan dependency provider registered in AlayaConan.cmake: the first find_package below
# triggers `conan install` transparently and the generated config packages land in ${CMAKE_BINARY_DIR}/conan. OpenMP
# (system / Homebrew) and libaio (system, handled by the disk-I/O targets) are not in the Conan graph and fall through
# the provider to CMake's builtin lookup. THIRD_PARTY_LIBS stays a plain list because test helper code and the AlayaLite
# INTERFACE target both consume it.

include_guard(GLOBAL)

if(APPLE)
  file(GLOB OPENMP_BREW_PATHS "/usr/local/opt/libomp" # Intel Mac
       "/opt/homebrew/opt/libomp" # Apple Silicon
  )
  list(APPEND CMAKE_PREFIX_PATH ${OPENMP_BREW_PATHS})
  foreach(openmp_brew_path IN LISTS OPENMP_BREW_PATHS)
    list(APPEND CMAKE_INCLUDE_PATH "${openmp_brew_path}/include")
    list(APPEND CMAKE_LIBRARY_PATH "${openmp_brew_path}/lib")
  endforeach()
endif()

find_package(concurrentqueue REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP QUIET)
find_package(RocksDB REQUIRED)
find_package(libcoro REQUIRED)

if(BUILD_PYTHON)
  # Reuse the interpreter alaya_preflight() resolved (FindPython / Python_* family) instead of letting pybind11 run its
  # own discovery, so the module, the fixtures, and Conan all agree on one Python.
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 REQUIRED)
endif()

set(THIRD_PARTY_LIBS
    spdlog::spdlog_header_only
    concurrentqueue::concurrentqueue
    Eigen3::Eigen
    RocksDB::rocksdb
    libcoro::libcoro
)

if(TARGET OpenMP::OpenMP_CXX)
  list(APPEND THIRD_PARTY_LIBS OpenMP::OpenMP_CXX)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(liburing REQUIRED)
  list(APPEND THIRD_PARTY_LIBS liburing::liburing)
endif()
