set(EXECUTOR_TEST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/coro_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/search_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/update_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mutex_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/scheduler_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/task_queue_test.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/task_promise_base_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/worker_test.cpp
)

add_executable(executor_test ${EXECUTOR_TEST_SRC})

target_link_libraries(executor_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})

# Add test to CTest
add_test(NAME ExecutorTest COMMAND executor_test)
