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
option(ALAYA_NATIVE_ARCH "Compile with -march=native for host-specific builds" OFF)
option(ALAYA_ALLOW_NATIVE_PACKAGE "Allow BUILD_PYTHON with host-specific -march=native" OFF)

if(BUILD_PYTHON
   AND ALAYA_NATIVE_ARCH
   AND NOT ALAYA_ALLOW_NATIVE_PACKAGE
)
  message(FATAL_ERROR "ALAYA_NATIVE_ARCH=ON makes Python package builds host-specific and unsafe to distribute. "
                      "Use -DALAYA_ALLOW_NATIVE_PACKAGE=ON only for local, non-distributable builds."
  )
endif()

# Laser on-disk QG index module. Linux x86_64 defaults to libaio; macOS defaults to the portable thread-pool backend;
# Windows x64 defaults to the IOCP backend. Other platforms skip it silently by default.
if((CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
   OR CMAKE_SYSTEM_NAME STREQUAL "Darwin"
   OR (CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
)
  set(ALAYA_ENABLE_LASER_DEFAULT ON)
else()
  set(ALAYA_ENABLE_LASER_DEFAULT OFF)
endif()
option(ALAYA_ENABLE_LASER "Build the Laser disk-index module" ${ALAYA_ENABLE_LASER_DEFAULT})
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(ALAYA_LASER_USE_THREADPOOL_DEFAULT ON)
else()
  set(ALAYA_LASER_USE_THREADPOOL_DEFAULT OFF)
endif()
option(ALAYA_LASER_USE_THREADPOOL "Use the portable thread-pool LASER I/O backend"
       ${ALAYA_LASER_USE_THREADPOOL_DEFAULT}
)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
  set(ALAYA_LASER_USE_IOCP_DEFAULT ON)
else()
  set(ALAYA_LASER_USE_IOCP_DEFAULT OFF)
endif()
option(ALAYA_LASER_USE_IOCP "Use the Windows IOCP LASER I/O backend" ${ALAYA_LASER_USE_IOCP_DEFAULT})
if(ALAYA_LASER_USE_IOCP AND NOT (CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64"
                                )
)
  message(FATAL_ERROR "ALAYA_LASER_USE_IOCP=ON requires Windows AMD64/x86_64. On other platforms use "
                      "-DALAYA_LASER_USE_THREADPOOL=ON or the libaio default."
  )
endif()

# ============================================================================
# Build accelerators: ccache + fast linker (auto-detected)
#
# The pybind11 module (_alayalitepy) is a single translation unit that expands dispatch.hpp's nested macros into ~288
# template instantiations. Single-TU means Ninja's -j N cannot parallelize it, so we lean on ccache to make repeat
# builds cheap and on mold/lld to shrink the link phase.
# ============================================================================
option(ALAYA_USE_CCACHE "Use ccache as compiler launcher when available" ON)
if(ALAYA_USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    # `cmake -E env` wraps each compile so CCACHE_SLOPPINESS is in scope for ccache itself, not just CMake's configure
    # step. pch_defines + time/ mtime/ctime sloppiness is required for PCH to actually hit cache.
    set(CMAKE_C_COMPILER_LAUNCHER
        ${CMAKE_COMMAND}
        -E
        env
        CCACHE_SLOPPINESS=pch_defines,time_macros,include_file_mtime,include_file_ctime
        ${CCACHE_PROGRAM}
    )
    set(CMAKE_CXX_COMPILER_LAUNCHER
        ${CMAKE_COMMAND}
        -E
        env
        CCACHE_SLOPPINESS=pch_defines,time_macros,include_file_mtime,include_file_ctime
        ${CCACHE_PROGRAM}
    )
    message(STATUS "ccache enabled: ${CCACHE_PROGRAM}")
  endif()
endif()

option(ALAYA_USE_FAST_LINKER "Use mold or lld when available" ON)
if(ALAYA_USE_FAST_LINKER AND NOT MSVC)
  find_program(MOLD_PROGRAM mold)
  find_program(LLD_PROGRAM ld.lld)
  if(MOLD_PROGRAM)
    add_link_options(-fuse-ld=mold)
    message(STATUS "Fast linker: mold (${MOLD_PROGRAM})")
  elseif(LLD_PROGRAM)
    add_link_options(-fuse-ld=lld)
    message(STATUS "Fast linker: lld (${LLD_PROGRAM})")
  endif()
endif()

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
  set(ALAYA_SIMD_COMPILE_OPTIONS)
  if(ALAYA_NATIVE_ARCH)
    list(APPEND ALAYA_SIMD_COMPILE_OPTIONS -march=native)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    list(APPEND ALAYA_SIMD_COMPILE_OPTIONS -mavx2 -mfma)
  endif()

  set(ALAYA_SANITIZER_BUILD OFF)
  set(ALAYA_ASAN_RUNTIME)
  set(ALAYA_OPTIMIZATION_OPTION -Ofast)
  if(CMAKE_CXX_FLAGS MATCHES "(^| )-fsanitize=")
    set(ALAYA_SANITIZER_BUILD ON)
    set(ALAYA_OPTIMIZATION_OPTION -O1)
    if(CMAKE_CXX_FLAGS MATCHES "-fsanitize=[^ ]*address")
      execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
        OUTPUT_VARIABLE ALAYA_ASAN_RUNTIME
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      # GCC echoes the input name verbatim when the file is not on its search path; clear it so downstream LD_PRELOAD
      # guards stay correct.
      if(NOT EXISTS "${ALAYA_ASAN_RUNTIME}")
        set(ALAYA_ASAN_RUNTIME)
      endif()
    endif()
  endif()

  add_compile_options(-Wall ${ALAYA_OPTIMIZATION_OPTION} ${ALAYA_SIMD_COMPILE_OPTIONS} -funroll-loops)
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
if(BUILD_PYTHON)
  find_package(pybind11 REQUIRED)
endif()

set(THIRD_PARTY_LIBS spdlog::spdlog_header_only concurrentqueue::concurrentqueue Eigen3::Eigen RocksDB::rocksdb)

if(TARGET OpenMP::OpenMP_CXX)
  list(APPEND THIRD_PARTY_LIBS OpenMP::OpenMP_CXX)
endif()

if(UNIX AND NOT APPLE)
  find_package(libcoro REQUIRED)
  find_package(liburing REQUIRED)
  list(APPEND THIRD_PARTY_LIBS libcoro::libcoro liburing::liburing)
endif()

# ============================================================================
# Laser module backend selection and system dependencies (gated by ALAYA_ENABLE_LASER)
# ============================================================================
if(ALAYA_ENABLE_LASER)
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ALAYA_LASER_USE_THREADPOOL
        ON
        CACHE BOOL "Use the portable thread-pool LASER I/O backend" FORCE
    )
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
    set(ALAYA_LASER_USE_IOCP
        ON
        CACHE BOOL "Use the Windows IOCP LASER I/O backend" FORCE
    )
  elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(
      FATAL_ERROR
        "ALAYA_ENABLE_LASER=ON but this platform is not supported yet. "
        "Supported platforms: Linux x86_64 (libaio), macOS (thread pool), Windows x64 (IOCP). "
        "Configure with -DALAYA_ENABLE_LASER=OFF to skip the Laser module."
    )
  endif()

  if(NOT TARGET OpenMP::OpenMP_CXX)
    if(APPLE)
      message(FATAL_ERROR "Laser module requires OpenMP. On macOS install Homebrew libomp "
                          "(brew install libomp) or configure with -DALAYA_ENABLE_LASER=OFF."
      )
    else()
      find_package(OpenMP REQUIRED)
    endif()
  endif()

  if(ALAYA_LASER_USE_IOCP)
    set(_alaya_laser_backend_message "IOCP (Windows x64)")
  elseif(ALAYA_LASER_USE_THREADPOOL)
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      set(_alaya_laser_backend_message "thread pool (macOS)")
    else()
      set(_alaya_laser_backend_message "thread pool (portable fallback)")
    endif()
  else()
    find_library(AIO_LIBRARY aio)
    find_path(AIO_INCLUDE_DIR libaio.h)
    if(NOT AIO_LIBRARY OR NOT AIO_INCLUDE_DIR)
      message(
        FATAL_ERROR
          "Laser module requires libaio for the default Linux backend. Install it via:\n"
          "  sudo apt-get install libaio-dev   (Debian / Ubuntu)\n"
          "  sudo dnf install libaio-devel     (Fedora / RHEL)\n"
          "or configure with -DALAYA_ENABLE_LASER=OFF to skip the Laser module, or "
          "-DALAYA_LASER_USE_THREADPOOL=ON to use the portable backend on Linux."
      )
    endif()
    if(NOT TARGET AIO::aio)
      add_library(AIO::aio UNKNOWN IMPORTED)
      set_target_properties(
        AIO::aio PROPERTIES IMPORTED_LOCATION "${AIO_LIBRARY}" INTERFACE_INCLUDE_DIRECTORIES "${AIO_INCLUDE_DIR}"
      )
    endif()
    set(_alaya_laser_backend_message "libaio (Linux x86_64)")
  endif()

  message(STATUS "LASER I/O backend: ${_alaya_laser_backend_message}")

  if(ALAYA_LASER_USE_IOCP)
    # CreateIoCompletionPort / GetQueuedCompletionStatusEx / ReadFile live in kernel32, which MSVC links implicitly; no
    # extra link libraries are required.
    set(_alaya_laser_backend_libs)
    set(_alaya_laser_backend_definition ALAYA_LASER_USE_IOCP=1)
  elseif(ALAYA_LASER_USE_THREADPOOL)
    set(_alaya_laser_backend_libs)
    set(_alaya_laser_backend_definition ALAYA_LASER_USE_THREADPOOL=1)
  else()
    set(_alaya_laser_backend_libs AIO::aio)
    set(_alaya_laser_backend_definition ALAYA_LASER_USE_LIBAIO=1)
  endif()

  # INTERFACE target that exposes the ported Laser header tree plus its runtime dependency surface. Consumers (pybind
  # module, alignment test, examples) link against this to pick up include paths and transitive libs.
  add_library(alaya_laser INTERFACE)
  target_include_directories(
    alaya_laser INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(
    alaya_laser
    INTERFACE ${_alaya_laser_backend_libs}
              Eigen3::Eigen
              OpenMP::OpenMP_CXX
              concurrentqueue::concurrentqueue
              spdlog::spdlog_header_only
  )
  target_compile_definitions(alaya_laser INTERFACE ${_alaya_laser_backend_definition})
  target_compile_features(alaya_laser INTERFACE cxx_std_20)
  # -mavx2 -mfma is the LASER baseline for Eigen and non-handwritten SIMD code; handwritten LASER kernels use
  # function-level target attributes for runtime AVX-512 dispatch.
  if(MSVC)
    # MSVC: /arch:AVX2 enables AVX2 + FMA at the codegen level. There is no MSVC analogue of -ftree-vectorize (it
    # auto-vectorizes under /O2 by default). EIGEN_DONT_PARALLELIZE is passed via /D.
    set(_ALAYA_LASER_CONSUMER_OPTIONS
        /arch:AVX2 /DEIGEN_DONT_PARALLELIZE
        CACHE INTERNAL "Compile options a Laser-consuming target must apply PRIVATEly"
    )
  else()
    set(_ALAYA_LASER_CONSUMER_OPTIONS
        ${ALAYA_SIMD_COMPILE_OPTIONS} -ftree-vectorize -DEIGEN_DONT_PARALLELIZE
        CACHE INTERNAL "Compile options a Laser-consuming target must apply PRIVATEly"
    )
  endif()
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

if(BUILD_PYTHON)
  add_subdirectory(python)
endif()
print_build_summary()
