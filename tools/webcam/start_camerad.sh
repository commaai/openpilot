#!/bin/bash
export DUAL="0"
export BLOCK="${BLOCK},camerad"
export YUV_BUFFER_COUNT="20"

#Change camera index according to your setting
export CAMERA_ROAD_ID="0"
export CAMERA_DRIVER_ID="1"
export CAMERA_WIDE_ID="2"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
python3 $DIR/camerad.py
