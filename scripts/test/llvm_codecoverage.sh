#!/bin/bash
set +e
# 参数设置
ROOT_DIR=$(dirname $(dirname $(dirname "$(realpath "$0")")))
BUILD_DIR="${ROOT_DIR}/build"
BIN_DIR="${BUILD_DIR}/bin"         
REPORT_DIR="${BUILD_DIR}/coverage_llvm"
PROFDATA_FILE="merged.profdata" 

# 清空旧数据
rm -rf "$REPORT_DIR"/*.profraw 2>/dev/null
mkdir -p "$REPORT_DIR"

# 遍历可执行文件并运行
find "$BIN_DIR" -type f -executable | while read -r executable; do
	echo "Analyzing: $executable"
	LLVM_PROFILE_FILE="$REPORT_DIR/$(basename "$executable").profraw" \
		"$executable" >/dev/null 2>&1 || true
done

# 合并所有 .profraw 文件
llvm-profdata merge "$REPORT_DIR"/*.profraw -o "$REPORT_DIR/$PROFDATA_FILE"

# 生成 HTML 报告
llvm-cov show --format=html --output-dir="$REPORT_DIR" \
	${BIN_DIR}/* \
	--ignore-filename-regex='build/*' \
	-instr-profile="$REPORT_DIR/$PROFDATA_FILE" \
	>$REPORT_DIR/index.html

echo "The report is generated at $REPORT_DIR/index.html"
