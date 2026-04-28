cmake_minimum_required(VERSION 3.15...3.31)

# Project definition
set(PROJECT_NAME AlayaLite)
project(${PROJECT_NAME} LANGUAGES CXX)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE
  )
endif()

# Include utility functions
include(cmake/PrintSummary.cmake)
include(cmake/Codecov.cmake)

option(ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)
option(BUILD_PYTHON "Build Python extension module" ON)
option(BUILD_TESTING "Build tests" OFF)

# Laser on-disk QG index module. Linux-only because it depends on libaio (kernel io_submit / io_getevents). Non-Linux
# builds skip it silently.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(ALAYA_ENABLE_LASER_DEFAULT ON)
else()
  set(ALAYA_ENABLE_LASER_DEFAULT OFF)
endif()
option(ALAYA_ENABLE_LASER "Build the Laser disk-index module (requires Linux + libaio)" ${ALAYA_ENABLE_LASER_DEFAULT})

# ============================================================================
# 1. C++ STANDARD AND COMPILER CONFIGURATION
# ============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  add_compile_options(/EHsc /utf-8 /O2)
else()
  add_compile_options(
    -Wall
    -Ofast
    -march=native
    -funroll-loops
    -fopenmp
  )
endif()

# ============================================================================
# 1. Dependency Management (Auto Conan Setup)
# ============================================================================
include(cmake/ConanSetup.cmake)
if(APPLE)
  file(GLOB OPENMP_BREW_PATHS "/usr/local/opt/libomp" # Intel Mac
       "/opt/homebrew/opt/libomp" # Apple Silicon
  )
  list(APPEND CMAKE_PREFIX_PATH ${OPENMP_BREW_PATHS})
endif()
find_package(concurrentqueue REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(RocksDB REQUIRED)
if(BUILD_PYTHON)
  find_package(pybind11 REQUIRED)
endif()

set(THIRD_PARTY_LIBS
    spdlog::spdlog_header_only
    concurrentqueue::concurrentqueue
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    RocksDB::rocksdb
)

if(UNIX AND NOT APPLE)
  find_package(libcoro REQUIRED)
  find_package(liburing REQUIRED)
  list(APPEND THIRD_PARTY_LIBS libcoro::libcoro liburing::liburing)
endif()

# ============================================================================
# Laser module: libaio system dependency (Linux-only, gated by ALAYA_ENABLE_LASER)
# ============================================================================
if(ALAYA_ENABLE_LASER)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "ALAYA_ENABLE_LASER=ON but this is not Linux. "
                        "Configure with -DALAYA_ENABLE_LASER=OFF to skip the Laser module."
    )
  endif()
  find_library(AIO_LIBRARY aio)
  find_path(AIO_INCLUDE_DIR libaio.h)
  if(NOT AIO_LIBRARY OR NOT AIO_INCLUDE_DIR)
    message(
      FATAL_ERROR
        "Laser module requires libaio. Install it via:\n" "  sudo apt-get install libaio-dev   (Debian / Ubuntu)\n"
        "  sudo dnf install libaio-devel     (Fedora / RHEL)\n"
        "or configure with -DALAYA_ENABLE_LASER=OFF to skip the Laser module."
    )
  endif()
  if(NOT TARGET AIO::aio)
    add_library(AIO::aio UNKNOWN IMPORTED)
    set_target_properties(
      AIO::aio PROPERTIES IMPORTED_LOCATION "${AIO_LIBRARY}" INTERFACE_INCLUDE_DIRECTORIES "${AIO_INCLUDE_DIR}"
    )
  endif()

  # INTERFACE target that exposes the ported Laser header tree plus its runtime dependency surface. Consumers (pybind
  # module, alignment test, examples) link against this to pick up include paths and transitive libs.
  add_library(alaya_laser INTERFACE)
  target_include_directories(
    alaya_laser INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(
    alaya_laser
    INTERFACE AIO::aio
              Eigen3::Eigen
              OpenMP::OpenMP_CXX
              concurrentqueue::concurrentqueue
              spdlog::spdlog_header_only
  )
  target_compile_features(alaya_laser INTERFACE cxx_std_20)
  # NOTE: Laser's FastScan / L2 kernels dispatch on __AVX2__ / __AVX512F__ and therefore require SIMD compile flags at
  # the TU that instantiates them. Those flags (and EIGEN_DONT_PARALLELIZE, needed so Eigen's thread pool doesn't
  # contend with Laser's OpenMP) are applied PRIVATE on the two actual consumers (python/CMakeLists.txt and
  # tests/laser/CMakeLists.txt) instead of on this INTERFACE, to avoid polluting every downstream that links
  # alaya_laser. _ALAYA_LASER_CONSUMER_OPTIONS lists the flags each consumer must apply itself.
  set(_ALAYA_LASER_CONSUMER_OPTIONS
      -march=native -ftree-vectorize -DEIGEN_DONT_PARALLELIZE
      CACHE INTERNAL "Compile options a Laser-consuming target must apply PRIVATEly"
  )
endif()

# ============================================================================
# PROJECT TARGETS
# ============================================================================
include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} INTERFACE ${THIRD_PARTY_LIBS})
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

# ============================================================================
# SUBDIRECTORIES
# ============================================================================
if(BUILD_TESTING)
  include(CTest)
  find_package(GTest REQUIRED)
  set(GTEST_LIBS GTest::gtest GTest::gtest_main)
  add_subdirectory(tests)
endif()

option(BUILD_TOOLS "Build standalone C++ CLI tools under tools/" ON)
if(BUILD_TOOLS)
  add_subdirectory(tools/build_vamana_index)
endif()

if(BUILD_PYTHON)
  add_subdirectory(python)
endif()
print_build_summary()
