#!/bin/bash

export ROAD_CAM='nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink'
echo "ROAD_CAM set to gstreamer pipeline:"
echo "$ROAD_CAM"

echo "--------------------------------"
echo "To use the camera, run:"
echo "USE_WEBCAM=1 USE_FAKE_PANDA=1 ROAD_CAM="$ROAD_CAM" ./system/manager/manager.py"
echo "--------------------------------"
