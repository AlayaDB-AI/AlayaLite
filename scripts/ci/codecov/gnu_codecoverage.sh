#!/bin/bash
# alayalite/scripts/ci/codecov/gnu_codecoverage.sh
set -e
set -x
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../../..")"

# now only test UtilsTest
cd ${ROOT_DIR}/build

ctest -R UtilsTest --verbose --output-on-failure
lcov -c -d ${ROOT_DIR}/build -o ${ROOT_DIR}/build/coverage_all.info
lcov --remove ${ROOT_DIR}/build/coverage_all.info \
     '*/.conan2/*' \
     '*/gtest*' \
     '*/fmt*' \
     '*/usr/include/*' \
     '*/usr/local/include/*' \
     -o ${ROOT_DIR}/build/coverage.info

genhtml ${ROOT_DIR}/build/coverage.info -o ${ROOT_DIR}/build/coverage_html
