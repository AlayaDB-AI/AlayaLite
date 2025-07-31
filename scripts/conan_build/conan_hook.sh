#!/bin/bash

output_dir="${1:-build/generator}"
# 获取当前脚本所在目录（与 Python 的 __file__ 类似）
PWD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 获取环境变量 CIBW_ARCHS，默认为 x86_64
arch="${CIBW_ARCHS:-x86_64}"

# 拼接 profile 路径
h_profile="$PWD_DIR/conan_profile.${arch}"
b_profile="$PWD_DIR/conan_profile.x86_64"

# 执行 conan install 命令
conan install . \
    --build=missing \
    -pr:h="${h_profile}" \
    -pr:b="${b_profile}" \
    --output-folder=${output_dir}
