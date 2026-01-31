#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

source ../../../setup.sh

# reset coverage data and generate gcc note file
rm -f ./libsafety/*.gcda
scons -j$(nproc) -D

# run safety tests and generate coverage data
pytest -n8 --ignore-glob=misra/*

if [ "$(uname)" = "Darwin" ]; then
  GCOV_EXEC="/opt/homebrew/opt/llvm@18/bin/llvm-cov gcov"
else
  GCOV_EXEC="llvm-cov-18 gcov"
fi

# generate and open report
if [ "$1" == "--report" ]; then
  mkdir -p coverage-out
  gcovr -r ../ --gcov-executable "$GCOV_EXEC" --html-nested coverage-out/index.html
  sensible-browser coverage-out/index.html
fi

# test coverage
GCOV="gcovr -r $DIR/../ --gcov-executable \"$GCOV_EXEC\" -d --fail-under-line=100 -e ^libsafety"
if ! GCOV_OUTPUT="$(eval $GCOV)"; then
  echo -e "FAILED:\n$GCOV_OUTPUT"
  exit 1
else
  echo "SUCCESS: All checked files have 100% coverage!"
fi
