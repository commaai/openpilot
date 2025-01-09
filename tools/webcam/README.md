# Run openpilot with webcam on PC

What's needed:
- Ubuntu 24.04 ([WSL2 is not supported](https://github.com/commaai/openpilot/issues/34216))
- GPU (recommended)
- Two USB webcams, at least 720p and 78 degrees FOV (e.g. Logitech C920/C615)
- [Car harness](https://comma.ai/shop/products/comma-car-harness) with black panda to connect to your car
- [Panda paw](https://comma.ai/shop/products/panda-paw) or USB-A to USB-A cable to connect panda to your computer
That's it!

## Setup openpilot
- Follow [this readme](../README.md) to install and build the requirements
- Install OpenCL Driver
```
sudo apt install pocl-opencl-icd
```

## Connect the hardware
- Connect the road facing camera first, then the driver facing camera
- Connect your computer to panda

## GO
```
USE_WEBCAM=1 system/manager/manager.py
```
- Start the car, then the UI should show the road webcam's view
- Adjust and secure the webcams.
- Finish calibration and engage!

## Specify Cameras

Use the `ROAD_CAM`, `DRIVER_CAM`, and optional `WIDE_CAM` environment variables to specify which camera is which (ie. `DRIVER_CAM=2` uses `/dev/video2` for the driver-facing camera):
```
USE_WEBCAM=1 ROAD_CAM=4 WIDE_CAM=6 system/manager/manager.py
```
