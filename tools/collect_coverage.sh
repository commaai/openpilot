#!/bin/bash

find opendbc/can/dbc_out -name "*.gcda"
find rednose/helpers -name "*.gcda"
find rednose_repo/rednose/helpers -name "*.gcda"
find cereal/gen -name "*.gcda"
lcov --capture --directory . --gcov-tool tools/gcov_for_clang.sh --output-file coverage.info
lcov --remove coverage.info '*/third_party/*' --output-file coverage.info
lcov --remove coverage.info '/usr/*' --output-file coverage.info
coverage xml
