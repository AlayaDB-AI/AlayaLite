
set(EXECUTOR_TEST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/search_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/update_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mutex_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/task_queue_test.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/task_promise_base_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/worker_test.cpp
)

add_executable(executor_test ${EXECUTOR_TEST_SRC})

if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(executor_test PRIVATE --coverage -O0 -g)
        target_link_libraries(executor_test PRIVATE gcov ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(executor_test PRIVATE -fprofile-instr-generate -fcoverage-mapping)
        target_link_options(executor_test PRIVATE -fprofile-instr-generate)
        target_link_libraries(executor_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
    endif()
else()
    target_link_libraries(executor_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
endif()
