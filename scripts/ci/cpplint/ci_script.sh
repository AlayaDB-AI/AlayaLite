#!/bin/bash
# alayalite/scripts/ci/cpplint/ci_script.sh
set -e
set -x
SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
ROOT_DIR="$(realpath "$SCRIPT_DIR/../../..")"

# cpp lint path
tests_path=$ROOT_DIR/tests
include_path=$ROOT_DIR/include

echo "Cpplint code style check through shell"
echo "run path: $ROOT_DIR"

########################################
# header only and cpp only check
########################################
# check if src directory exists header file
for file in `find $tests_path -name "*.h" -o -name "*.hpp"`; do
    # echo $file
    echo "Do not add header file in src directory"
    exit 1
done

# check if include directory exists cpp file
for file in `find $include_path -name "*.cpp"`; do
    echo "Do not add cpp file in include directory"
    exit 1
done

ret=0
####################
# cpplint check
####################
cpplint --filter=-legal/copyright --verbose=3 --recursive "$tests_path,$include_path" || ret=1

####################
# clang-format check
####################
python3 "$SCRIPT_DIR/run_clang_format.py" --clang_format_binary clang-format --exclude_globs "$SCRIPT_DIR/ignore_files.txt" --source_dir "$tests_path,$include_path" || ret=1

####################
# clang tidy check
####################
python3 "$SCRIPT_DIR/run_clang_tidy.py" -clang-tidy-binary clang-tidy -style file -p "$ROOT_DIR/build" || ret=1

exit $ret
