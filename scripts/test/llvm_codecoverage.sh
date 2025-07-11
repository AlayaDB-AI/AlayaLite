#!/bin/bash
set +e
# Parameter settings
ROOT_DIR=$(dirname $(dirname $(dirname "$(realpath "$0")")))
BUILD_DIR="${ROOT_DIR}/build"
BIN_DIR="${BUILD_DIR}/bin"         
REPORT_DIR="${BUILD_DIR}/coverage_llvm"
PROFDATA_FILE="merged.profdata" 

# Clear old data
rm -rf "$REPORT_DIR"/*.profraw 2>/dev/null
mkdir -p "$REPORT_DIR"

# Traverse and run executable files
find "$BIN_DIR" -type f -executable | while read -r executable; do
	echo "Analyzing: $executable"
	LLVM_PROFILE_FILE="$REPORT_DIR/$(basename "$executable").profraw" \
		"$executable" >/dev/null 2>&1 || true
done

# Merge all .profraw files
llvm-profdata merge "$REPORT_DIR"/*.profraw -o "$REPORT_DIR/$PROFDATA_FILE"

# Generate HTML report
llvm-cov show --format=html --output-dir="$REPORT_DIR" \
	${BIN_DIR}/* \
	--ignore-filename-regex='build/*' \
	-instr-profile="$REPORT_DIR/$PROFDATA_FILE" \
	>$REPORT_DIR/index.html

echo "The report is generated at $REPORT_DIR/index.html"
