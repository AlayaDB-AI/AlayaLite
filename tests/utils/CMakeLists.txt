set(UTILS_TEST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/query_utils_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/log_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/evaluate_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/metric_type_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_utils_test.cpp
)

add_executable(utils_test ${UTILS_TEST_SRC})

if(ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(utils_test PRIVATE --coverage -O0 -g)
    target_link_libraries(utils_test PRIVATE gcov ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(utils_test PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(utils_test PRIVATE -fprofile-instr-generate)
    target_link_libraries(utils_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
  endif()
else()
  target_link_libraries(utils_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
endif()
