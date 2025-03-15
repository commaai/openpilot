#!/usr/bin/env bash
set -e

# Loops over all HW_TYPEs, see board/boards/board_declarations.h
for hw_type in {0..7}; do
  echo "Testing HW_TYPE: $hw_type"
  HW_TYPE=$hw_type python3 -m unittest discover .
done
