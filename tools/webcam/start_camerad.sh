#!/usr/bin/env bash

# export the block below when call manager.py
export BLOCK="${BLOCK},camerad"
export USE_WEBCAM="1"

# Change camera index according to your setting
export CAMERA_ROAD_ID="/dev/video0"
export CAMERA_DRIVER_ID="/dev/video1"
#export DUAL_CAMERA="/dev/video2" # optional, camera index for wide road camera

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

$DIR/camerad.py
