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
# Follow [this readme](https://github.com/commaai/openpilot/tree/master/tools) to install the requirements 
# Add line "export PYTHONPATH=$HOME/openpilot" to your ~/.bashrc 
# You may also need to install tensorflow-gpu 2.0.0 
# Install [OpenCV4](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/) 
```
## Build openpilot for webcam
```
cd ~/openpilot
scons use_webcam=1
touch prebuilt
```
## Connect the hardwares 
```
# Connect the road facing camera first, then the driver facing camera 
# (default indexes are 1 and 2; can be modified in selfdrive/camerad/cameras/camera_webcam.cc)
# Connect your computer to panda
```
## GO 
```
cd ~/openpilot/tools/webcam 
./accept_terms.py # accept the user terms so that thermald can detect the car started 
cd ~/openpilot/selfdrive 
PASSIVE=0 NOSENSOR=1 ./manager.py 
# Start the car, then the UI should show the road webcam's view 
# Adjust and secure the webcams (you can run tools/webcam/front_mount_helper.py to help mount the driver camera)
# Finish calibration and engage!
```
