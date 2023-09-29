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
You can also stream data to and from your computer with the cereal bridge. yolo.py is a good example of a model runner that can run either on device with thneed, or remotely using onnxruntime and the cereal bridge.

Prerequisites:
* If you have an nvidia gpu, install `onnxruntime-gpu` on your machine to run big models much faster.
* devcontainer lacks GPU support on macOS, but there're still some options to achieve GPU accelerated inference. One idea would be to create native inference server (this could be a basic flask app accepting requests for inference), and then communicate with it from devcontainer via `host.docker.internal:PORT`

A very basic requirement to build cool applications on the body is to be able to:
* Receive image frames from the body to a remote computer
* Run the model on the frames, stream the outputs to the body via cereal messaging
* Control the body based on these model outputs

This section walks you through this process, running YOLO as an example. This guide assumes that both devices (comma body + remote pc) are on the same network.

### 1. Receive images
```
# On the remote computer, run
# Use --nvidia if you have nvidia graphics card for better performance
./tools/camerastream/compressed_vipc.py --cams 2 <body-ip>
```
This decodes the `driverEncodeData` packets from the body, and pushes it to a local VisionIPC server.

### 2. Setup cereal bridge

To be able to send cereal messages from remote computer to the body, cereal bridge must be opened. 

SSH into the body from the remote computer you intent to run models on. Then start cereal bridge using command below. Replace services... with the list of published sockets you intent to share with the body. For the yolo example that would be: customReservedRawData1

Note: if you're in a tmux session, detach now
```
REMOTE_IP="$(echo $SSH_CLIENT | awk '{print $1}')"
tmux new-window -k -d -t comma -n remotebridge "/data/openpilot/cereal/messaging/bridge $REMOTE_IP <services...>"
```

To kill the bridge session, run:

```
tmux kill-window -t comma:remotebridge
```

### 3. Run Model -> Send Outputs
```
# On another terminal window of the remote computer, run
./selfdrive/modeld/yolo.py
```
This creates a local VisionIPC client and receives the driver camera frames. We run the `YOLO5N` model on these frames using `onnxruntime`. There's post-processing code to parse the model outputs, compute bounding boxes and filter out objects with low probability. Finally, we publish the outputs to a socket by the message name `customReservedRawData1`.

Note: cereal services in ZMQ mode listen on ports 8001-8079, make sure those are available
