#!/usr/bin/env bash

# Loop over all HW_TYPEs, see board/boards/board_declarations.h
# Make sure test fails if one HW_TYPE fails
set -e

scons -u --test

for hw_type in {0..7}; do
  echo "Testing HW_TYPE: $hw_type"
  HW_TYPE=$hw_type python -m unittest discover .
done
