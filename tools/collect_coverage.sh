#!/bin/bash
set -e

find opendbc/can/dbc_out -name "*.gcda" -delete
find rednose/helpers -name "*.gcda" -delete
find rednose_repo/rednose/helpers -name "*.gcda" -delete
find cereal/gen -name "*.gcda" -delete

lcov --capture --directory . --gcov-tool tools/gcov_for_clang.sh --output-file coverage.info
lcov --remove coverage.info '*/third_party/*' --output-file coverage.info
lcov --remove coverage.info '/usr/*' --output-file coverage.info

coverage xml || true
