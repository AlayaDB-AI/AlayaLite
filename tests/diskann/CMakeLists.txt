# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# DiskANN disk index tests.
#
# Layout / PQ / node-cache / trace tests are header-only over std + SIMD and need no I/O backend. Beam search / index /
# e2e use the LASER AlignedFileReader (libaio on Linux) and therefore need the LASER consumer surface (alaya_laser, its
# compile options, and ALAYA_ENABLE_LASER).

alaya_cc_target(
  test_diskann_layout
  GTEST
  SRCS test_diskann_layout.cpp
)
alaya_cc_target(
  test_diskann_pq
  GTEST
  SRCS test_diskann_pq.cpp
)
alaya_cc_target(
  test_diskann_node_cache
  GTEST
  SRCS test_diskann_node_cache.cpp
)
alaya_cc_target(
  test_diskann_tombstone_slot
  GTEST
  SRCS test_diskann_tombstone_slot.cpp
)
alaya_cc_target(
  test_diskann_update_trace
  GTEST
  SRCS test_diskann_update_trace.cpp
)
alaya_cc_target(
  test_diskann_portable_update
  GTEST
  SRCS test_diskann_portable_update.cpp
)

alaya_add_test(
  NAME test_diskann_layout
  TARGET test_diskann_layout
  LABELS diskann
)
alaya_add_test(
  NAME test_diskann_pq
  TARGET test_diskann_pq
  LABELS diskann
)
alaya_add_test(
  NAME test_diskann_node_cache
  TARGET test_diskann_node_cache
  LABELS diskann
)
alaya_add_test(
  NAME test_diskann_tombstone_slot
  TARGET test_diskann_tombstone_slot
  LABELS diskann
)
alaya_add_test(
  NAME test_diskann_update_trace
  TARGET test_diskann_update_trace
  LABELS diskann
)
alaya_add_test(
  NAME test_diskann_portable_update
  TARGET test_diskann_portable_update
  LABELS diskann
)

if(ALAYA_ENABLE_LASER)
  alaya_cc_target(
    test_diskann_beam_search
    GTEST LASER
    SRCS test_diskann_beam_search.cpp
  )
  alaya_cc_target(
    test_diskann_index
    GTEST LASER
    SRCS test_diskann_index.cpp
  )
  alaya_cc_target(
    test_diskann_e2e
    GTEST LASER
    SRCS test_diskann_e2e.cpp
  )
  alaya_cc_target(
    test_diskann_disk_page_io
    GTEST LASER
    SRCS test_diskann_disk_page_io.cpp
  )
  alaya_cc_target(
    test_diskann_tombstone_search
    GTEST LASER
    SRCS test_diskann_tombstone_search.cpp
  )
  alaya_cc_target(
    test_diskann_update_e2e
    GTEST LASER
    SRCS test_diskann_update_e2e.cpp
  )

  alaya_add_test(
    NAME test_diskann_beam_search
    TARGET test_diskann_beam_search
    LABELS diskann
  )
  alaya_add_test(
    NAME test_diskann_index
    TARGET test_diskann_index
    LABELS diskann
  )
  alaya_add_test(
    NAME test_diskann_disk_page_io
    TARGET test_diskann_disk_page_io
    LABELS diskann
  )
  alaya_add_test(
    NAME test_diskann_tombstone_search
    TARGET test_diskann_tombstone_search
    LABELS diskann
  )
  alaya_add_test(
    NAME test_diskann_update_e2e
    TARGET test_diskann_update_e2e
    LABELS diskann
  )
  # The end-to-end build+search test is heavier; give it a generous timeout.
  alaya_add_test(
    NAME test_diskann_e2e
    TARGET test_diskann_e2e
    LABELS diskann e2e
    TIMEOUT 600
  )

  # Standalone SIFT1M benchmarks (manual; NOT registered with ctest). bench_diskann_sift measures recall / latency /
  # disk-reads-per-query so the index can be compared against the official DiskANN `search_disk_index` tool on the same
  # data and metrics. bench_diskann_sift_update replays Yi-format update traces (see make_diskann_sift_update_trace)
  # through mixed search+update workloads. Build the target, then execute the binary explicitly.
  alaya_cc_target(
    bench_diskann_sift
    LASER
    SRCS bench_diskann_sift.cpp
  )
  alaya_cc_target(
    bench_diskann_sift_update
    LASER
    SRCS bench_diskann_sift_update.cpp
  )
  alaya_cc_target(make_diskann_sift_update_trace SRCS make_diskann_sift_update_trace.cpp)
endif()
