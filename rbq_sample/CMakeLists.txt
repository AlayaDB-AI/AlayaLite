set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/exec)
add_executable(hnsw_rabitq_indexing hnsw_rabitq_indexing.cpp)
add_executable(hnsw_rabitq_querying hnsw_rabitq_querying.cpp)

target_link_libraries(hnsw_rabitq_indexing PRIVATE ${THIRD_PARTY_LIBS})
target_link_libraries(hnsw_rabitq_querying PRIVATE ${THIRD_PARTY_LIBS})