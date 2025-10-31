# Project configuration options and validation This module defines project-wide options and validates their dependencies

# Configuration options
option(ENABLE_UNIT_TESTS "Enable unit tests" ON)
option(ENABLE_COVERAGE "Enable test coverage analysis (requires ENABLE_UNIT_TESTS)" OFF)

# Validate option dependencies
if(ENABLE_COVERAGE AND NOT ENABLE_UNIT_TESTS)
  message(WARNING "ENABLE_COVERAGE requires ENABLE_UNIT_TESTS.")
  set(ENABLE_UNIT_TESTS
      ON
      CACHE BOOL "Enable unit tests" FORCE
  )
endif()

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE
  )
  message(STATUS "Build type not specified, defaulting to Release")
endif()

if(ENABLE_COVERAGE)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "Build type" FORCE
  )

  # Configure coverage flags based on compiler
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -fprofile-update=atomic")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage -fprofile-update=atomic")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
  endif()

endif()

# Print configuration summary Print comprehensive project information
# ~~~
message(STATUS "")
message(STATUS "+===========================================================+")
message(STATUS "|                  ${PROJECT_NAME} Build Configuration")
message(STATUS "+===========================================================+")
message(STATUS "| Build Information:")
message(STATUS "|   • Build Type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "|   • C++ Standard    : C++${CMAKE_CXX_STANDARD}")
message(STATUS "|   • Compiler        : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "|   • CMake Version   : ${CMAKE_VERSION}")
message(STATUS "|")
message(STATUS "| Compiler Flags:")
message(STATUS "|   • Common Flags    : ${CMAKE_CXX_FLAGS}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "|   • Release Flags   : ${CMAKE_CXX_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "|   • Debug Flags     : ${CMAKE_CXX_FLAGS_DEBUG}")
endif()
message(STATUS "|   • CMAKE_EXE_LINKER_FLAGS : ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "|")
message(STATUS "| Directory Structure:")
message(STATUS "|   • Source Dir      : ${PROJECT_SOURCE_DIR}")
message(STATUS "|   • Binary Dir      : ${PROJECT_BINARY_DIR}")
message(STATUS "|   • Runtime Output  : ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "|")
message(STATUS "| Features:")
message(STATUS "|   • PIC Enabled     : ${CMAKE_POSITION_INDEPENDENT_CODE}")
message(STATUS "|   • Export Commands : ${CMAKE_EXPORT_COMPILE_COMMANDS}")
if(MSVC)
  message(STATUS "|   • VS Toolset      : ${CMAKE_GENERATOR_TOOLSET}")
endif()
message(STATUS "|")
message(STATUS "| Project Options:")
message(STATUS "|   • Unit Tests      : ${ENABLE_UNIT_TESTS}")
message(STATUS "|   • Code Coverage   : ${ENABLE_COVERAGE}")
message(STATUS "+===========================================================+")
message(STATUS "")
# ~~~
