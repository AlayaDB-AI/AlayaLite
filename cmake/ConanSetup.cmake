# ConanSetup.cmake - Run conan_install.py and include the generated toolchain

if(_CONAN_SETUP_DONE)
  return()
endif()

# cmake_layout output path: Windows -> build/generators/, Others -> build/<type>/generators/
if(WIN32)
  set(CONAN_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/build/generators/conan_toolchain.cmake")
else()
  set(CONAN_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}/generators/conan_toolchain.cmake")
endif()

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

# Include toolchain
if(NOT EXISTS "${CONAN_TOOLCHAIN_FILE}")
  message(FATAL_ERROR "Conan toolchain not found: ${CONAN_TOOLCHAIN_FILE}")
endif()
include("${CONAN_TOOLCHAIN_FILE}")

set(_CONAN_SETUP_DONE
    TRUE
    CACHE INTERNAL ""
)
