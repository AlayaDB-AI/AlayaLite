set(python_module_name _alayalitepy)

if(PYTHON_EXECUTABLE)
  set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

find_package(
  Python3
  COMPONENTS Interpreter Development.Module
  REQUIRED
)

add_library(${python_module_name} MODULE src/pybind.cpp)
target_include_directories(${python_module_name} PRIVATE include)

target_link_libraries(
  ${python_module_name}
  PRIVATE
  pybind11::module
  pybind11::lto
  AlayaLite
)

set_target_properties(${python_module_name}
  PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET "hidden"
  INSTALL_RPATH "$ORIGIN"
  BUILD_WITH_INSTALL_RPATH TRUE
)

if(WIN32)
    set_target_properties(${python_module_name} PROPERTIES SUFFIX ".pyd")
endif()

install(TARGETS ${python_module_name}
    LIBRARY DESTINATION alayalite
    RUNTIME DESTINATION alayalite
)
