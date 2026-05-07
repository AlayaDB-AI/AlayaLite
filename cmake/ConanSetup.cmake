# ConanSetup.cmake - Run conan_install.py and include the generated toolchain

# cmake_layout output path: Windows -> build/generators/, Others -> build/<type>/generators/
if(WIN32)
  set(CONAN_GENERATORS_DIR "${CMAKE_SOURCE_DIR}/build/generators")
else()
  set(CONAN_GENERATORS_DIR "${CMAKE_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}/generators")
endif()
set(CONAN_TOOLCHAIN_FILE "${CONAN_GENERATORS_DIR}/conan_toolchain.cmake")

# Run conan install if toolchain doesn't exist
if(NOT EXISTS "${CONAN_TOOLCHAIN_FILE}")
  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED
  )

  set(CONAN_INSTALL_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/conan_build/conan_install.py")
  message(STATUS "Running: ${CONAN_INSTALL_SCRIPT}")

  execute_process(
    COMMAND ${Python3_EXECUTABLE} "${CONAN_INSTALL_SCRIPT}" --project-dir "${CMAKE_SOURCE_DIR}" --build-type
            "${CMAKE_BUILD_TYPE}"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE _conan_result
  )

  if(NOT _conan_result EQUAL 0)
    message(FATAL_ERROR "Conan install failed with exit code: ${_conan_result}")
  endif()
endif()

# CMake caches `<Package>_DIR` entries. If a build directory is reconfigured from Debug to Release (or the reverse),
# those cached entries can keep pointing at the previous Conan generator directory, stripping dependency include paths
# from generated compile commands.
file(GLOB _conan_config_files "${CONAN_GENERATORS_DIR}/*Config.cmake" "${CONAN_GENERATORS_DIR}/*-config.cmake")
foreach(conan_config_file IN LISTS _conan_config_files)
  get_filename_component(_conan_config_name "${conan_config_file}" NAME)
  string(REGEX REPLACE "(-config|Config)\\.cmake$" "" _conan_package_name "${_conan_config_name}")
  set(_conan_package_dir_var "${_conan_package_name}_DIR")
  if(DEFINED CACHE{${_conan_package_dir_var}})
    get_property(
      _conan_cached_dir
      CACHE "${_conan_package_dir_var}"
      PROPERTY VALUE
    )
    if(_conan_cached_dir AND NOT _conan_cached_dir STREQUAL CONAN_GENERATORS_DIR)
      unset(${_conan_package_dir_var} CACHE)
    endif()
  endif()
endforeach()

# Include toolchain
if(NOT EXISTS "${CONAN_TOOLCHAIN_FILE}")
  message(FATAL_ERROR "Conan toolchain not found: ${CONAN_TOOLCHAIN_FILE}")
endif()
include("${CONAN_TOOLCHAIN_FILE}")
