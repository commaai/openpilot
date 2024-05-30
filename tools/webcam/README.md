# Run openpilot with webcam on PC

What's needed:
- Ubuntu 20.04
- GPU (recommended)
- Two USB webcams, at least 720p and 78 degrees FOV (e.g. Logitech C920/C615)
- [Car harness](https://comma.ai/shop/products/comma-car-harness) with black panda to connect to your car
- [Panda paw](https://comma.ai/shop/products/panda-paw) or USB-A to USB-A cable to connect panda to your computer
That's it!

## Setup openpilot
```
cd ~
git clone https://github.com/commaai/openpilot.git
```
- Follow [this readme](https://github.com/commaai/openpilot/tree/master/tools) to install the requirements
- Install [OpenCL Driver](https://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/15532/l_opencl_p_18.1.0.015.tgz)

## Build openpilot for webcam
```
cd ~/openpilot
USE_WEBCAM=1 scons -j$(nproc)
```

## Connect the hardware
- Connect the road facing camera first, then the driver facing camera
- (default indexes are 1 and 2; can be modified in system/camerad/cameras/camera_webcam.cc)
- Connect your computer to panda

## GO
```
cd ~/openpilot/system/manager
NOSENSOR=1 USE_WEBCAM=1 ./manager.py
```
- Start the car, then the UI should show the road webcam's view
- Adjust and secure the webcams (you can run tools/webcam/front_mount_helper.py to help mount the driver camera)
- Finish calibration and engage!
