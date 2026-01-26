add_executable(l2_sqr_test ${CMAKE_CURRENT_SOURCE_DIR}/l2_sqr_test.cpp)
add_executable(l2_sqr_full_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/l2_sqr_full_benchmark.cpp)
add_executable(l2_sqr_sq8_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/l2_sqr_sq8_benchmark.cpp)
add_executable(l2_sqr_sq4_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/l2_sqr_sq4_benchmark.cpp)

target_link_libraries(l2_sqr_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(l2_sqr_full_benchmark PRIVATE ${THIRD_PARTY_LIBS})
target_link_libraries(l2_sqr_sq8_benchmark PRIVATE ${THIRD_PARTY_LIBS})
target_link_libraries(l2_sqr_sq4_benchmark PRIVATE ${THIRD_PARTY_LIBS})

add_test(NAME simd_test_l2_sqr COMMAND $<TARGET_FILE:l2_sqr_test>)
