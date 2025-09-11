set(PYTHON_MODULE_NAME _alayalitepy)

find_package(
  Python3
  COMPONENTS Interpreter Development.Module
  REQUIRED
)

add_library(${PYTHON_MODULE_NAME} MODULE src/pybind.cpp)
target_include_directories(${PYTHON_MODULE_NAME} PRIVATE include)

target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE pybind11::module pybind11::lto AlayaLite)

set_target_properties(
  ${PYTHON_MODULE_NAME}
  PROPERTIES PREFIX ""
             CXX_VISIBILITY_PRESET "hidden"
             INSTALL_RPATH "$ORIGIN"
             BUILD_WITH_INSTALL_RPATH TRUE
)

if(WIN32)
  set_target_properties(${PYTHON_MODULE_NAME} PROPERTIES SUFFIX ".pyd")
endif()

install(
  TARGETS ${PYTHON_MODULE_NAME}
  LIBRARY DESTINATION alayalite
  RUNTIME DESTINATION alayalite
)
