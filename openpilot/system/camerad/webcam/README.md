# Run openpilot with webcam on PC

## Setup openpilot
- Follow [this readme](../README.md) to install and build the requirements

## Connect the hardware
- Connect the camera first
- Connect your computer to panda

## GO

```
USE_WEBCAM=1 system/manager/manager.py
```
- Start the car, then the UI should show the road webcam's view
- Adjust and secure the webcam
- Finish calibration and engage!

## Specify Cameras

Use the `ROAD_CAM` (default 0) and optional `DRIVER_CAM`, `WIDE_CAM` environment variables to specify which camera is which (ie. `ROAD_CAM=1` uses `/dev/video1`, on Ubuntu, for the road camera):
```
USE_WEBCAM=1 ROAD_CAM=1 system/manager/manager.py
```
