cmake_minimum_required(VERSION 3.15...3.31)

set(project_name AlayaLite)
project(${project_name} LANGUAGES CXX)

# options
option(ENABLE_COVERAGE "Enable profiling test coverage" OFF)
option(ENABLE_UNIT_TESTS "Enable unit tests" OFF)

# set CMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Build type not specified, defaulting to 'Release'.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
endif()

# --- Conan Integration Start ---
# This block automatically calls 'conan install' during configuration.
set(CONAN_OUTPUT_FOLDER "${CMAKE_BINARY_DIR}/conan_generate")
if(NOT EXISTS "${CONAN_OUTPUT_FOLDER}/conan_toolchain.cmake")
    message(STATUS "Conan files not found, running 'conan install'")
    execute_process(COMMAND conan profile detect -e)
    execute_process(
        COMMAND conan install ${CMAKE_SOURCE_DIR}
                --output-folder=${CONAN_OUTPUT_FOLDER}
                --build=missing
                --settings=build_type=${CMAKE_BUILD_TYPE}
                --settings=compiler.cppstd=20
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE conan_result
    )
    if(conan_result)
        message(FATAL_ERROR "Conan install failed")
    endif()
endif()

# Include the Conan toolchain file. This MUST be before project().
# For simplicity, we re-invoke project() after including the toolchain.
include("${CONAN_OUTPUT_FOLDER}/conan_toolchain.cmake")
project(${project_name} LANGUAGES CXX)
# --- Conan Integration End ---

# please install clang-18 or g++-13
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # -fPIC
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG ${CMAKE_CXX_FLAGS_RELEASE}")  
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 ${CMAKE_CXX_FLAGS_DEBUG}")
add_compile_options(${CONAN_USER_MARCH_FLAGS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_definitions(-DPROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# define third_party
find_package(concurrentqueue REQUIRED)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(libcoro REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)
set(THIRD_PARTY_LIBS spdlog::spdlog fmt::fmt concurrentqueue::concurrentqueue libcoro::libcoro)
set(GTEST_LIBS GTest::gtest GTest::gtest_main)

# include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
add_library(${project_name} INTERFACE)
target_link_libraries(${project_name} INTERFACE ${THIRD_PARTY_LIBS})
if (ENABLE_UNIT_TESTS)
    add_subdirectory(tests)
endif()
add_subdirectory(python)
