# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# DiskANN's portable reader backend and platform runtime dependencies.

include_guard(GLOBAL)

set(_alaya_diskann_backend_libs libcoro::libcoro)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(_alaya_diskann_backend_definition ALAYA_LASER_USE_IOCP=1)
  set(_alaya_diskann_backend_message "IOCP + Win32 positioned writes")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR ALAYA_LASER_USE_THREADPOOL)
  set(_alaya_diskann_backend_definition ALAYA_LASER_USE_THREADPOOL=1)
  set(_alaya_diskann_backend_message "thread-pool reads + POSIX positioned writes")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_library(AIO_LIBRARY aio)
  find_path(AIO_INCLUDE_DIR libaio.h)
  if(NOT AIO_LIBRARY OR NOT AIO_INCLUDE_DIR)
    message(FATAL_ERROR "DiskANN requires libaio for the default Linux reader. Install libaio-dev/libaio-devel, "
                        "or configure with -DALAYA_LASER_USE_THREADPOOL=ON."
    )
  endif()
  if(NOT TARGET AIO::aio)
    add_library(AIO::aio UNKNOWN IMPORTED)
    set_target_properties(
      AIO::aio PROPERTIES IMPORTED_LOCATION "${AIO_LIBRARY}" INTERFACE_INCLUDE_DIRECTORIES "${AIO_INCLUDE_DIR}"
    )
  endif()
  list(APPEND _alaya_diskann_backend_libs AIO::aio liburing::liburing)
  set(_alaya_diskann_backend_definition ALAYA_LASER_USE_LIBAIO=1)
  set(_alaya_diskann_backend_message "libaio reads + O_DIRECT positioned writes")
else()
  set(_alaya_diskann_backend_definition ALAYA_LASER_USE_THREADPOOL=1)
  set(_alaya_diskann_backend_message "portable thread-pool reads + POSIX positioned writes")
endif()

message(STATUS "DiskANN I/O backend: ${_alaya_diskann_backend_message}")

add_library(alaya_diskann INTERFACE)
target_include_directories(
  alaya_diskann INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)
target_link_libraries(alaya_diskann INTERFACE ${_alaya_diskann_backend_libs})
target_compile_definitions(alaya_diskann INTERFACE ${_alaya_diskann_backend_definition})
target_compile_features(alaya_diskann INTERFACE cxx_std_20)
