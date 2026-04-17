add_executable(recovery_test ${CMAKE_CURRENT_SOURCE_DIR}/recovery_test.cpp)
target_link_libraries(recovery_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
add_coverage_to_target(recovery_test)

add_test(NAME recovery_test_wal COMMAND $<TARGET_FILE:recovery_test> --gtest_filter=WalTest.*)
add_test(NAME recovery_test_snapshot_manifest COMMAND $<TARGET_FILE:recovery_test>
                                                      --gtest_filter=SnapshotManifestTest.*
)
add_test(NAME recovery_test_recovery_manager COMMAND $<TARGET_FILE:recovery_test> --gtest_filter=RecoveryManagerTest.*)
set_tests_properties(
  recovery_test_wal recovery_test_snapshot_manifest recovery_test_recovery_manager PROPERTIES LABELS "recovery"
)
