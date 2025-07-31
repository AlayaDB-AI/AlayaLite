#!/bin/bash
set -e
set -x

output_dir="${1:-build/generator}"

PWD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SOURCE_DIR="$(dirname "$(dirname "${PWD_DIR}")")"

# arch="${CIBW_ARCHS_LINUX:-x86_64}"
arch=$(uname -m)

# TODO: Cross-compilation
h_profile="$PWD_DIR/conan_profile.${arch}"  # Target machine
b_profile="$PWD_DIR/conan_profile.x86_64"  # Build machine

conan install ${PROJECT_SOURCE_DIR} \
    --build=missing \
    -pr:h="${h_profile}" \
    -pr:b="${b_profile}" \
    --output-folder=${output_dir}
