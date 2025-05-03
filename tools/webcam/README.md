# Run openpilot with webcam on PC

What's needed:
- Ubuntu 24.04 ([WSL2 is not supported](https://github.com/commaai/openpilot/issues/34216)) or macOS
- GPU (recommended)
- One USB webcam, at least 720p and 78 degrees FOV (e.g. Logitech C920/C615, NexiGo N60)
- [Car harness](https://comma.ai/shop/products/comma-car-harness)
- [panda](https://comma.ai/shop/panda)
- USB-A to USB-A cable to connect panda to your computer

## Setup openpilot
- Follow [this readme](../README.md) to install and build the requirements
- Install OpenCL Driver (Ubuntu)
```
sudo apt install pocl-opencl-icd
```

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
