add_executable(fusion_graph_test ${CMAKE_CURRENT_SOURCE_DIR}/fusion_graph_test.cpp)
add_executable(graph_test ${CMAKE_CURRENT_SOURCE_DIR}/graph_test.cpp)
add_executable(hnsw_test ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_test.cpp)
add_executable(nndescent_test ${CMAKE_CURRENT_SOURCE_DIR}/nndescent_test.cpp)
add_executable(nsg_test ${CMAKE_CURRENT_SOURCE_DIR}/nsg_test.cpp)
add_executable(rabitq_test ${CMAKE_CURRENT_SOURCE_DIR}/rabitq_test.cpp)
add_executable(rabitq_performance_test ${CMAKE_CURRENT_SOURCE_DIR}/rabitq_performance_test.cpp)
add_executable(rabitq_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/rabitq_benchmark.cpp)
add_executable(
  hybrid_quantization_performance_test ${CMAKE_CURRENT_SOURCE_DIR}/hybrid_quantization_performance_test.cpp
)

target_link_libraries(fusion_graph_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(graph_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(hnsw_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(nndescent_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(nsg_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(rabitq_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(rabitq_performance_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(rabitq_benchmark PRIVATE ${THIRD_PARTY_LIBS})
target_link_libraries(hybrid_quantization_performance_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})

add_coverage_to_target(fusion_graph_test)
add_coverage_to_target(graph_test)
add_coverage_to_target(hnsw_test)
add_coverage_to_target(nndescent_test)
add_coverage_to_target(nsg_test)
add_coverage_to_target(rabitq_test)
add_coverage_to_target(rabitq_performance_test)
add_coverage_to_target(hybrid_quantization_performance_test)

add_test(NAME index_test_fusion_graph COMMAND $<TARGET_FILE:fusion_graph_test>)
add_test(NAME index_test_graph COMMAND $<TARGET_FILE:graph_test>)
add_test(NAME index_test_hnsw COMMAND $<TARGET_FILE:hnsw_test>)
add_test(NAME index_test_nndescent COMMAND $<TARGET_FILE:nndescent_test>)
add_test(NAME index_test_nsg COMMAND $<TARGET_FILE:nsg_test>)
add_test(NAME index_test_rabitq COMMAND $<TARGET_FILE:rabitq_test>)
add_test(NAME index_test_rabitq_performance COMMAND $<TARGET_FILE:rabitq_performance_test>)
set_tests_properties(index_test_rabitq_performance PROPERTIES LABELS "performance")

# test_vamana_alignment — Gate 1 alignment harness, hand-written main with
# exit codes 10/11/12 for L1/L2/L3 tiers (not a GTest target). Requires CLI
# args (--alaya_index, --diskann_index, --data_path, --query_path, --gt_path),
# so it is not registered with add_test — run manually.
add_executable(test_vamana_alignment ${CMAKE_CURRENT_SOURCE_DIR}/test_vamana_alignment.cpp)
target_link_libraries(test_vamana_alignment PRIVATE ${PROJECT_NAME})
target_compile_features(test_vamana_alignment PRIVATE cxx_std_20)
