add_executable(vamana_build_dispatch_test ${CMAKE_CURRENT_SOURCE_DIR}/vamana_build_dispatch_test.cpp)
target_link_libraries(vamana_build_dispatch_test PRIVATE ${PROJECT_NAME} ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
add_coverage_to_target(vamana_build_dispatch_test)
add_test(NAME vamana_test_build_dispatch COMMAND $<TARGET_FILE:vamana_build_dispatch_test>)

# test_vamana_alignment — Gate 1 alignment harness, hand-written main with exit codes 10/11/12 for L1/L2/L3 tiers (not a
# GTest target). Requires CLI args (--alaya_index, --diskann_index, --data_path, --query_path, --gt_path), so it is not
# registered with add_test — run manually.
add_executable(test_vamana_alignment ${CMAKE_CURRENT_SOURCE_DIR}/test_vamana_alignment.cpp)
target_link_libraries(test_vamana_alignment PRIVATE ${PROJECT_NAME})
target_compile_features(test_vamana_alignment PRIVATE cxx_std_20)
