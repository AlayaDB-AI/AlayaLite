#!/bin/bash
# alayalite/scripts/ci/codecov/gnu_codecoverage.sh
set -e
set -x

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../../..")"
BUILD_DIR="${ROOT_DIR}/build"


# rebuild the project
rm -rf ${BUILD_DIR} && mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_UNIT_TESTS=ON -DENABLE_COVERAGE=ON -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13
make -j

# run the tests
ctest --verbose --output-on-failure
lcov  --capture \
     --ignore-errors mismatch \
     --directory ${BUILD_DIR} \
     --output-file ${BUILD_DIR}/coverage_all.info \
     --gcov-tool /usr/bin/gcov-13

lcov --remove ${BUILD_DIR}/coverage_all.info \
     '*/.conan2/*' \
     '*/usr/include/*' \
     '*/tests/*' \
     --output-file ${ROOT_DIR}/coverage_c++.info

rm -rf ${BUILD_DIR}/coverage_all.info

# genhtml ${BUILD_DIR}/coverage.info -o ${BUILD_DIR}/coverage_html
