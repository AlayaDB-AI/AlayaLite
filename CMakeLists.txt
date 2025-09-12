cmake_minimum_required(VERSION 3.15...3.31)

# Project definition
set(PROJECT_NAME AlayaLite)
project(${PROJECT_NAME} LANGUAGES CXX)

# ============================================================================
# INCLUDE MODULAR CONFIGURATION FILES
# ============================================================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerConfig) # Project options and validation
include(OptionConfig) # Project options and validation
include(ConanDepConfig) # Conan integration and dependency management

# ============================================================================
# PROJECT TARGETS CONFIGURATION
# ============================================================================
message(STATUS "Configuring project targets...")

include_directories(${CMAKE_SOURCE_DIR}/include)

# Define main library target as interface (header-only)
add_library(${PROJECT_NAME} INTERFACE)
target_link_libraries(${PROJECT_NAME} INTERFACE ${THIRD_PARTY_LIBS})

# Set target properties for modern CMake
target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)

# Add compile features
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)
message(STATUS "Main target '${PROJECT_NAME}' configured as interface library")

# Enable CTest for testing framework
include(CTest)

# Include subdirectories based on configuration
if(ENABLE_UNIT_TESTS)
  add_subdirectory(tests)
  message(STATUS "Added tests subdirectory")
endif()

# Always include Python bindings
add_subdirectory(python)
message(STATUS "Added python subdirectory")

message(STATUS "${PROJECT_NAME} configuration completed successfully!")
