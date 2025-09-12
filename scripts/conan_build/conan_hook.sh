#!/bin/bash
set -e
set -x

output_dir="${1:-build/generator}"
operating_system="${2:-Linux}"
build_type="${3:-Release}"

PWD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SOURCE_DIR="$(dirname "$(dirname "${PWD_DIR}")")"

if [ ${build_type} != "Release" ] && [ ${build_type} != "Debug" ]; then
    echo "build_type: ${build_type} not valid"
    exit 1
fi

if [ ${operating_system} = "Macos" ]; then
    # Only support Apple Silicon
    h_profile="$PWD_DIR/conan_profile_mac.aarch64"  # Build machine
    b_profile="$PWD_DIR/conan_profile_mac.aarch64"  # Target machine
elif [ ${operating_system} = "Linux" ]; then
    arch=$(uname -m)
    h_profile="$PWD_DIR/conan_profile.${arch}"  # Target machine
    b_profile="$PWD_DIR/conan_profile.${arch}"  # Build machine
else
    echo "operating_system: ${operating_system} not valid"
    exit 1
fi



conan install ${PROJECT_SOURCE_DIR} \
    --build=missing \
    -pr:h="${h_profile}" \
    -pr:b="${b_profile}" \
    -s build_type=${build_type} \
    --output-folder=${output_dir}
