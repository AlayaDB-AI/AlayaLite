set(PYTHON_MODULE_NAME _alayalitepy)

find_package(
  Python3
  COMPONENTS Interpreter Development.Module
  REQUIRED
)

# Automatically process various attributes (such as suffixes, visibility, etc.)
pybind11_add_module(${PYTHON_MODULE_NAME} src/pybind.cpp)
target_include_directories(
  ${PYTHON_MODULE_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
          ${pybind11_INCLUDE_DIRS}
          ${spdlog_INCLUDE_DIRS}
          ${fmt_INCLUDE_DIRS}
          ${concurrentqueue_INCLUDE_DIRS}
          ${Eigen3_INCLUDE_DIRS}
          ${RocksDB_INCLUDE_DIRS}
          ${libcoro_INCLUDE_DIRS}
          ${liburing_INCLUDE_DIRS}
          $<TARGET_PROPERTY:spdlog::spdlog_header_only,INTERFACE_INCLUDE_DIRECTORIES>
          $<TARGET_PROPERTY:fmt::fmt,INTERFACE_INCLUDE_DIRECTORIES>
          $<TARGET_PROPERTY:concurrentqueue::concurrentqueue,INTERFACE_INCLUDE_DIRECTORIES>
          $<TARGET_PROPERTY:Eigen3::Eigen,INTERFACE_INCLUDE_DIRECTORIES>
          $<TARGET_PROPERTY:RocksDB::rocksdb,INTERFACE_INCLUDE_DIRECTORIES>
)

if(UNIX AND NOT APPLE)
  target_include_directories(
    ${PYTHON_MODULE_NAME} PRIVATE $<TARGET_PROPERTY:libcoro::libcoro,INTERFACE_INCLUDE_DIRECTORIES>
                                  $<TARGET_PROPERTY:liburing::liburing,INTERFACE_INCLUDE_DIRECTORIES>
  )
endif()

target_compile_definitions(
  ${PYTHON_MODULE_NAME}
  PRIVATE SPDLOG_FMT_EXTERNAL
          ${spdlog_COMPILE_DEFINITIONS}
          ${fmt_COMPILE_DEFINITIONS}
          ${concurrentqueue_COMPILE_DEFINITIONS}
          ${RocksDB_COMPILE_DEFINITIONS}
          ${libcoro_COMPILE_DEFINITIONS}
          ${liburing_COMPILE_DEFINITIONS}
          $<TARGET_PROPERTY:spdlog::spdlog_header_only,INTERFACE_COMPILE_DEFINITIONS>
          $<TARGET_PROPERTY:fmt::fmt,INTERFACE_COMPILE_DEFINITIONS>
          $<TARGET_PROPERTY:concurrentqueue::concurrentqueue,INTERFACE_COMPILE_DEFINITIONS>
          $<TARGET_PROPERTY:RocksDB::rocksdb,INTERFACE_COMPILE_DEFINITIONS>
          $<TARGET_PROPERTY:libcoro::libcoro,INTERFACE_COMPILE_DEFINITIONS>
          $<TARGET_PROPERTY:liburing::liburing,INTERFACE_COMPILE_DEFINITIONS>
)

target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE AlayaLite pybind11::headers pybind11::module)

install(
  TARGETS ${PYTHON_MODULE_NAME}
  LIBRARY DESTINATION alayalite
  RUNTIME DESTINATION alayalite
)
