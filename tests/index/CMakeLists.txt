set(INDEX_TEST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/fusion_graph_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/graph_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nndescent_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nsg_test.cpp
)

add_executable(index_test ${INDEX_TEST_SRC})

target_link_libraries(index_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})

# Add test to CTest
add_test(NAME IndexTest COMMAND index_test)
