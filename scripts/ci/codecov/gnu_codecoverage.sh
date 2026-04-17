#!/bin/bash
# alayalite/scripts/ci/codecov/gnu_codecoverage.sh
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../../..")"
BUILD_DIR="${ROOT_DIR}/build"
BUILD_JOBS="${BUILD_JOBS:-4}"
CTEST_JOBS="${CTEST_JOBS:-4}"
CTEST_LABELS="${CTEST_LABELS:-unit|storage|recovery|simd|space|utils}"
CTEST_EXCLUDE_REGEX="${CTEST_EXCLUDE_REGEX:-^utils_test_dataset_utils$}"
GCOV_TOOL="${GCOV_TOOL:-/usr/bin/gcov-13}"

COVERAGE_TARGETS=(
  recovery_test
  l2_sqr_test
  ip_test
  fht_test
  cpu_features_test
  quant_test
  raw_space_test
  sq4_space_test
  sq8_space_test
  rabitq_space_test
  storage_test
  static_storage_test
  rocksdb_storage_test
  query_utils_test
  log_test
  evaluate_test
  metric_type_test
  data_utils_test
  index_encoding_test
  metadata_filter_test
  rotator_utils_test
  lut_utils_test
  math_test
)


# rebuild the project
rm -rf "${BUILD_DIR}" && mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"
cmake .. \
  -DBUILD_TESTING=ON \
  -DBUILD_PYTHON=OFF \
  -DENABLE_COVERAGE=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=gcc-13 \
  -DCMAKE_CXX_COMPILER=g++-13
cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}" --target "${COVERAGE_TARGETS[@]}"

# run the tests in parallel
ctest_cmd=(
  ctest
  --test-dir "${BUILD_DIR}"
  --output-on-failure
  -L "${CTEST_LABELS}"
  -j"${CTEST_JOBS}"
)

if [[ -n "${CTEST_EXCLUDE_REGEX}" ]]; then
  ctest_cmd+=(-E "${CTEST_EXCLUDE_REGEX}")
fi

"${ctest_cmd[@]}"
lcov --capture \
     --quiet \
     --ignore-errors mismatch,mismatch,gcov,gcov \
     --directory "${BUILD_DIR}" \
     --base-directory "${ROOT_DIR}" \
     --include "${ROOT_DIR}/include/*" \
     --output-file "${ROOT_DIR}/coverage_c++.info" \
     --gcov-tool "${GCOV_TOOL}"

lcov --summary "${ROOT_DIR}/coverage_c++.info"
