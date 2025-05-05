if(PYTHON_EXECUTABLE)
  set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

find_package(
  Python3
  COMPONENTS Interpreter Development.Module
  REQUIRED)

add_library(_alayalitepy MODULE
  src/pybind.cpp
)

target_include_directories(_alayalitepy AFTER PRIVATE include)
target_link_libraries(
  _alayalitepy
  PRIVATE
  pybind11::module
  pybind11::lto
  AlayaLite
)
pybind11_extension(_alayalitepy)

if(NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
  # Strip unnecessary sections of the binary on Linux/macOS
  pybind11_strip(_alayalitepy)
endif()

set_target_properties(_alayalitepy PROPERTIES CXX_VISIBILITY_PRESET "hidden"
  CUDA_VISIBILITY_PRESET "hidden")

if(NOT VERSION_INFO)
  set(VERSION_INFO "0.0.0dev")
endif()

if (NO_COROUTINE)
  target_compile_definitions(_alayalitepy PRIVATE VERSION_INFO="${VERSION_INFO}" __NO_COROUTINE__)
else()
  target_compile_definitions(_alayalitepy PRIVATE VERSION_INFO="${VERSION_INFO}")
endif()

