#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

source ../../../setup.sh

# reset coverage data and generate gcc note file
rm -f ./libsafety/*.gcda
if [ "$1" == "--ubsan" ]; then
  scons -j$(nproc) -D --ubsan
else
  scons -j$(nproc) -D
fi

# run safety tests and generate coverage data
pytest -n8 --ignore-glob=misra/*

# generate and open report
if [ "$1" == "--report" ]; then
  mkdir -p coverage-out
  gcovr -r ../ --html-nested coverage-out/index.html
  sensible-browser coverage-out/index.html
fi

# test coverage
GCOV="gcovr -r ../ --fail-under-line=100 -e ^libsafety -e ^../board"
if ! GCOV_OUTPUT="$($GCOV)"; then
  echo -e "FAILED:\n$GCOV_OUTPUT"
  exit 1
else
  echo "SUCCESS: All checked files have 100% coverage!"
fi
