#!/bin/bash
PS4='+ ${BASH_SOURCE}:${LINENO}: ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
trap 'echo -e "\033[0;33m+Executing: $BASH_COMMAND\033[0m"' DEBUG

ROOT_DIR=$(pwd)
WHEEL_OUTPUT_DIR="pyalaya/alayalite"
echo "Root directory: $ROOT_DIR"

# remove previous build
file=$(ls ${WHEEL_OUTPUT_DIR}/alayalite*.whl 2>/dev/null)
if [ -n "$file" ]; then
	echo -e "\e[43mthe previous wheel file: $file will be deleted and re-generated!\e[0m"
    rm $file
    rm -rf pyalaya/src/alayalite.egg-info pyalaya/src/_alayalitepy.cpython*.so
else
    echo -e "\e[43mthe wheel file does not exist, it will be generated!\e[0m"
fi

# build the package
python -m build -o ${WHEEL_OUTPUT_DIR} --wheel

# reinstall the package
pip install ${WHEEL_OUTPUT_DIR}/alayalite-*.whl --force-reinstall

# test the package
python -c "import alayalite; print('alayalite version:', alayalite.__version__)"
