#!/usr/bin/env bash
set -o pipefail

python_bin="${PYTHON_BIN:-.venv/bin/python}"
log_path="${PYTEST_CRASH_LOG:-pytest-crash.log}"
pytest_cmd=(
  "$python_bin"
  -X faulthandler
  -m pytest
  --cov
  --cov-branch
  --cov-report=xml
)
gdb_opts=(-q -batch -ex "set pagination off" -ex "set print thread-events off")

dump_python_environment() {
  echo "::group::Python environment"
  "$python_bin" --version || true
  "$python_bin" -c "import platform, sys; print(sys.executable); print(sys.version); print(platform.platform())" || true
  "$python_bin" -c "import alayalite, alayalite._alayalitepy as m; print(alayalite.__file__); print(m.__file__)" || true
  echo "::endgroup::"
}

dump_native_crash() {
  echo "::group::Native crash backtrace"
  shopt -s nullglob
  cores=(core.*)

  if [ "${#cores[@]}" -gt 0 ]; then
    ls -lh "${cores[@]}" || true
    for core in "${cores[@]}"; do
      echo "Backtrace for ${core}"
      gdb "${gdb_opts[@]}" \
        -ex "thread apply all bt full" \
        -ex "info sharedlibrary" \
        "$python_bin" "$core" || true
    done
  else
    echo "No core file was produced; rerunning under gdb for an immediate native backtrace."
    gdb "${gdb_opts[@]}" \
      -ex "run" \
      -ex "thread apply all bt full" \
      -ex "info sharedlibrary" \
      --args "${pytest_cmd[@]}" || true
  fi

  echo "::endgroup::"
}

ulimit -c unlimited
sudo sysctl -w "kernel.core_pattern=${PWD}/core.%e.%p" || true

set +e
"${pytest_cmd[@]}" 2>&1 | tee "$log_path"
status=${PIPESTATUS[0]}
set -e

if [ "$status" -ne 0 ]; then
  dump_python_environment
  if [ "$status" -eq 139 ] || [ "$status" -eq 134 ]; then
    dump_native_crash
  fi
fi

exit "$status"
