#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

source ../../../setup.sh

# reset coverage data
rm -f ./libsafety/*.gcda

# run safety tests and generate coverage data
python -m unittest discover -s .

# NOTE: we accept that these tools will have slight differences,
# and in return, we get to use the stock toolchain instead of
# installing LLVM on all users' machines
if [ "$(uname)" = "Darwin" ]; then
  GCOV_EXEC="llvm-cov gcov"
else
  GCOV_EXEC="gcov"
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
