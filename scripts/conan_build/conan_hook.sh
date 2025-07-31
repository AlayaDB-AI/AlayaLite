#!/bin/bash
set -e
set -x

output_dir="${1:-build/generator}"

PWD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SOURCE_DIR="$(dirname "$(dirname "${PWD_DIR}")")"

arch="${CIBW_ARCHS_LINUX:-x86_64}"

h_profile="$PWD_DIR/conan_profile.${arch}"
b_profile="$PWD_DIR/conan_profile.x86_64"

conan install ${PROJECT_SOURCE_DIR} \
    --build=missing \
    -pr:h="${h_profile}" \
    -pr:b="${b_profile}" \
    --output-folder=${output_dir}
