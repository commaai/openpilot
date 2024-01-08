#!/bin/bash

export BLOCK="${BLOCK},camerad" #block camerad in terminal that calls manager.py
export DUAL="0"
export USE_WEBCAM="1" #export this in shell that calls manager.py
export YUV_BUFFER_COUNT="20"

#Change camera index according to your setting
export CAMERA_ROAD_ID="0"
export CAMERA_DRIVER_ID="1"
export CAMERA_WIDE_ID="2"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

python3 $DIR/camerad.py