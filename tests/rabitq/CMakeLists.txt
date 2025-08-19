
set(EXECUTOR_TEST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/deep1M_test.cpp
)

add_executable(rabitq_test ${EXECUTOR_TEST_SRC})

if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(rabitq_test PRIVATE --coverage -O0 -g)
        target_link_libraries(rabitq_test PRIVATE gcov ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(rabitq_test PRIVATE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(rabitq_test PRIVATE -fprofile-instr-generate)
        target_link_libraries(rabitq_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
    endif()
else()
    target_link_libraries(rabitq_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
endif()
