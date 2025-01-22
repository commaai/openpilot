#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

# reset coverage data and generate gcc note file
rm -f ../libpanda/*.gcda
scons -j$(nproc) -D --coverage

# run safety tests and generate coverage data
HW_TYPES=( 6 9 )
for hw_type in "${HW_TYPES[@]}"; do
  echo "Testing HW_TYPE: $hw_type"
  HW_TYPE=$hw_type pytest test_*.py
done

# generate and open report
if [ "$1" == "--report" ]; then
  geninfo ../libpanda/ -o coverage.info
  genhtml coverage.info -o coverage-out
  sensible-browser coverage-out/index.html
fi

# test coverage
GCOV_OUTPUT=$(gcov -n ../libpanda/panda.c)
INCOMPLETE_COVERAGE=$(echo "$GCOV_OUTPUT" | paste -s -d' \n' | grep -E "File.*(safety\/safety_.*)|(safety)\.h" | grep -v "100.00%" || true)
if [ -n "$INCOMPLETE_COVERAGE" ]; then
  echo "FAILED: Some files have less than 100% coverage:"
  echo "$INCOMPLETE_COVERAGE"
  exit 1
else
  echo "SUCCESS: All checked files have 100% coverage!"
fi
