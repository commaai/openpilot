# Run openpilot with webcam on PC (TODO: test these steps on a fresh Ubuntu install to verify)

What's needed:
- Ubuntu 24.04 (note, [WSL2 is not yet supported](http://TODO:INSERT_ISSUE_URL))
- GPU (recommended)
- Two USB webcams, at least 720p and 78 degrees FOV (e.g. Logitech C920/C615)
- [Car harness](https://comma.ai/shop/products/comma-car-harness) with black panda to connect to your car
- [Panda paw](https://comma.ai/shop/products/panda-paw) or USB-A to USB-A cable to connect panda to your computer
That's it!

## Setup openpilot
- Follow [this readme](https://github.com/commaai/openpilot/tree/master/tools) to install and build the requirements
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
- Start the car, then the UI should show the road webcam's view (TODO: should it!? This needs fixing then)
- Adjust and secure the webcams (you can run tools/webcam/front_mount_helper.py to help mount the driver camera) (TODO: restore this file)
- Finish calibration and engage!

## Specify Cameras

To specify individual cameras, use the `ROAD_CAM`, `DRIVER_CAM`, and optional `WIDE_CAM` environment variables (ie. `DRIVER_CAM=2` uses `/dev/video2` for the driver-facing camera):
```
ROAD_CAM=4 DRIVER_CAM=2 system/manager/manager.py
```
