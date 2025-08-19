#!/bin/bash
set -e
set -x

output_dir="${1:-build/generator}"
operating_system="${2:-Linux}"

PWD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SOURCE_DIR="$(dirname "$(dirname "${PWD_DIR}")")"



if [ ${operating_system} = "Macos" ]; then
    # Only support Apple Silicon
    h_profile="$PWD_DIR/conan_profile_mac.aarch64"  # Build machine
    b_profile="$PWD_DIR/conan_profile_mac.aarch64"  # Target machine
else
    arch=$(uname -m)
    h_profile="$PWD_DIR/conan_profile.${arch}"  # Target machine
    b_profile="$PWD_DIR/conan_profile.${arch}"  # Build machine
fi



conan install ${PROJECT_SOURCE_DIR} \
    --build=missing \
    -pr:h="${h_profile}" \
    -pr:b="${b_profile}" \
    --output-folder=${output_dir}
