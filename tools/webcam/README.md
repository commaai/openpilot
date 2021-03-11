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
- Add line "export PYTHONPATH=$HOME/openpilot" to your ~/.bashrc
- Install tensorflow 2.2 and nvidia drivers: nvidia-xxx/cuda10.0/cudnn7.6.5
- Install [OpenCL Driver](http://registrationcenter-download.intel.com/akdlm/irc_nas/vcp/15532/l_opencl_p_18.1.0.015.tgz)
- Install [OpenCV4](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/) (ignore the Python part)

## Build openpilot for webcam
```
cd ~/openpilot
```
- check out selfdrive/camerad/cameras/camera_webcam.cc lines 72 and 146 before building if any camera is upside down
```
USE_WEBCAM=1 scons -j$(nproc)
```

## Connect the hardware
- Connect the road facing camera first, then the driver facing camera
- (default indexes are 1 and 2; can be modified in selfdrive/camerad/cameras/camera_webcam.cc)
- Connect your computer to panda

## GO
```
cd ~/openpilot/selfdrive/manager
PASSIVE=0 NOSENSOR=1 USE_WEBCAM=1 ./manager.py
```
- Start the car, then the UI should show the road webcam's view
- Adjust and secure the webcams (you can run tools/webcam/front_mount_helper.py to help mount the driver camera)
- Finish calibration and engage!
