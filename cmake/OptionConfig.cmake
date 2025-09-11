# Project configuration options and validation
# This module defines project-wide options and validates their dependencies

# Configuration options
option(ENABLE_UNIT_TESTS "Enable unit tests" OFF)
option(ENABLE_COVERAGE "Enable test coverage analysis (requires ENABLE_UNIT_TESTS)" OFF)

# Validate option dependencies
if(ENABLE_COVERAGE AND NOT ENABLE_UNIT_TESTS)
  message(WARNING "ENABLE_COVERAGE requires ENABLE_UNIT_TESTS. Enabling unit tests automatically.")
  set(ENABLE_UNIT_TESTS ON CACHE BOOL "Enable unit tests" FORCE)
endif()

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  message(STATUS "Build type not specified, defaulting to Release")
endif()


# Print configuration summary
message(STATUS "=========PROJECT CONFIGURATION===========")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Unit tests: ${ENABLE_UNIT_TESTS}")
message(STATUS "Code coverage: ${ENABLE_COVERAGE}")
message(STATUS "========================================")
