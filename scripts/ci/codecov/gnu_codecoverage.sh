#!/bin/bash
# alayalite/scripts/ci/codecov/gnu_codecoverage.sh
set -e
set -x
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../../..")"
BUILD_DIR="${ROOT_DIR}/build"

# rebuild the project
rm -rf ${BUILD_DIR} && mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_UNIT_TESTS=ON -DENABLE_COVERAGE=ON
make -j

# run the tests
ctest --verbose --output-on-failure -R utils_test
lcov  --capture \
     --directory ${BUILD_DIR} \
     --output-file ${BUILD_DIR}/coverage_all.info \

lcov --remove ${BUILD_DIR}/coverage_all.info \
     '*/.conan2/*' \
     '*/gtest*' \
     '*/fmt*' \
     '*/usr/include/*' \
     '*/usr/local/include/*' \
     --output-file ${ROOT_DIR}/coverage_c++.info

# genhtml ${BUILD_DIR}/coverage.info -o ${BUILD_DIR}/coverage_html
