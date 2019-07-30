#!/usr/bin/env sh

# Loop over all hardware types:
#   HW_TYPE_UNKNOWN 0U
#   HW_TYPE_WHITE_PANDA 1U
#   HW_TYPE_GREY_PANDA 2U
#   HW_TYPE_BLACK_PANDA 3U
#   HW_TYPE_PEDAL 4U

# Make sure test fails if one HW_TYPE fails
set -e

for hw_type in 0 1 2 3 4
do
  echo "Testing HW_TYPE: $hw_type"
  HW_TYPE=$hw_type python -m unittest discover .
done
