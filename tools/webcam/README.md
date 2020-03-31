Run openpilot with webcam on PC/laptop
=====================
What's needed:  
- Ubuntu 16.04  
- Python 3.7.3  
- GPU (recommended)  
- Two USB webcams, at least 720p and 78 degrees FOV (e.g. Logitech C920/C615)  
- [Car harness](https://comma.ai/shop/products/comma-car-harness) w/ black panda (or the outdated grey panda/giraffe combo) to connect to your car  
- [Panda paw](https://comma.ai/shop/products/panda-paw) (or USB-A to USB-A cable) to connect panda to your computer  
- Tape, Charger, ...  
That's it!  

## Clone openpilot and install the requirements  
```
cd ~  
git clone https://github.com/commaai/openpilot.git  
```
- Follow [this readme](https://github.com/commaai/openpilot/tree/master/tools) to install the requirements  
- Add line "export PYTHONPATH=$HOME/openpilot" to your ~/.bashrc  
- You also need to install tensorflow-gpu 2.1.0 (if not working, try 2.0.0) and nvidia drivers: nvidia-xxx/cuda10.0/cudnn7.6.5  
- Install [OpenCL Driver](http://registrationcenter-download.intel.com/akdlm/irc_nas/12556/opencl_runtime_16.1.2_x64_rh_6.4.0.37.tgz)  
- (Note: the code assumes cl platforms order to be 0.GPU/1.CPU when running clinfo; if reverse, change the -1 to -2 in selfdrive/modeld/modeld.cc#L130; helping us refactor this mess is encouraged)  
- Install [OpenCV4](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/) (ignore the Python part)  

## Build openpilot for webcam  
```
cd ~/openpilot  
```
- check out selfdrive/camerad/cameras/camera_webcam.cc line72&146 before building if any camera is upside down  
```
scons use_webcam=1  
touch prebuilt  
```

## Connect the hardwares  
- Connect the road facing camera first, then the driver facing camera  
- (default indexes are 1 and 2; can be modified in selfdrive/camerad/cameras/camera_webcam.cc)  
- Connect your computer to panda  

## GO  
```
cd ~/openpilot/tools/webcam  
./accept_terms.py # accept the user terms so that thermald can detect the car started  
cd ~/openpilot/selfdrive  
PASSIVE=0 NOSENSOR=1 WEBCAM=1 ./manager.py  
```
- Start the car, then the UI should show the road webcam's view  
- Adjust and secure the webcams (you can run tools/webcam/front_mount_helper.py to help mount the driver camera)  
- Finish calibration and engage!  
