set(INDEX_TEST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/fusion_graph_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nndescent_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nsg_test.cpp
)

add_executable(index_test ${INDEX_TEST_SRC})

if(ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(index_test PRIVATE --coverage -O0 -g)
    target_link_libraries(index_test PRIVATE gcov ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(index_test PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(index_test PRIVATE -fprofile-instr-generate)
    target_link_libraries(index_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
  endif()
else()
  target_link_libraries(index_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
endif()
