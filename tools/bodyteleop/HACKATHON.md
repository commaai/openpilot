# Comma Hackathon Guide

Welcome to COMMA_HACK_4! This guide will provide you with some useful information to help you get started with openpilot and the comma body.

## Openpilot Basics
Openpilot processes communicate via cereal using SubMaster/PubMaster and VisionIpcClient/Server. Logs and video data are saved to /data/media/0/realdata in 1-minute segments. To read logs and video, use LogReader and FrameReader (see tools/lib/README.md for more information).

You can attach to the openpilot screen where all printouts and logs appear by running `tmux a` on your device. To detach, press `~ d`. After making changes to the code, you may need to restart openpilot for your changes to take effect. Attach to the tmux with `tmux a`, kill openpilot with ctrl-c (you may need to do this twice), and then restart openpilot with `./launch_openpilot.sh`. To compile any C++ changes, run scons with `./selfdrive/manager/build.py`.

## Body Basics
Your body should be on the `comma_hack_4` branch. We recommend forking off of this branch, as it has many useful changes for the hackathon that have not been upstreamed to master yet. You can enable/disable openpilot by pressing the button on the body's base. A face will appear on the screen when openpilot is enabled.

You can control your body manually using WASD in your browser. Make sure you're on the same network as your body, enable openpilot, and go to `https:<device-ip-address>:5000`. To find your device's IP address, go to settings -> network -> advanced.

The body movement is controlled by tools/bodyteleop/bodycontrolsd.py, which listens for data from other processes and then publishes the `testJoystick` messages that move the body's wheels. You can modify this file however you like.

## Creating a New Process
Add new processes in selfdrive/manager/process_config.py. You can send data between processes with PubMaster and SubMaster. The customReservedRawData0/1/2 messages are a convenient way to send raw bytes around, or you can make your own cereal messages by adding them to cereal/log.capnp and cereal/services.py. You may also have to bump NUM_READERS in cereal/messaging/msgq.h. Note that each cereal message can have many subscribers, but only one publisher. If modeld is publishing modelV2 messages, you'll get errors if you try to publish that message from another process at the same time.

You can get the data from any of the three cameras using VisionIpcClient. The camera data is published as YUV420, so if you need it as RGB, you can install opencv-python-headless with pip and use `cv2.cvtColor()` to convert the images.

## Creating a New Model Runner
See selfdrive/modeld/yolo.py for an example. You can run models on device, or run them remotely on your computer using the cereal bridge.

The easiest way to run models on device is with onnxruntime. You can use ONNXModel from selfdrive/modeld/models/onnxmodel.py or use onnxruntime directly. This should be easy to set up but runs on CPU so may be very slow for big models.

You can also run models with thneed, which uses the Comma 3X's GPU. See selfdrive/modeld/SConscript for an example of how to compile onnx models to thneed. Thneed supports models with a single flat output vector, and the size of the output vector must be a multiple of 4. Inputs and outputs must be in float32. Libthneed has to be specially linked for ThneedModel to work properly. The yolo runner is a good example - set up your process as a NativeProcess in process_config.py and add a shell script like `selfdrive/modeld/yolo` that sets LD_PRELOAD and then execs your python script.

## Remote Models / Teleop
You can also stream data to and from your computer with the cereal bridge. See tools/bodyteleop/README.md for instructions on how to set this up. yolo.py is a good example of a model runner that can run either on device with thneed, or remotely using onnxruntime and the cereal bridge. If you have an nvidia gpu, install `onnxruntime-gpu` on your machine to run big models much faster. Note that this does not work with macOS.

Good luck!
