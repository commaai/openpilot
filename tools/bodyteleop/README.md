# bodyteleop

## Components
- `web.py` is the tele-operation server that starts automatically when the commabody goes onroad, which can be found at `https://<body-ip>:5000`.
- `bodyav.py` has all the audio/video webRTC tracks
- `static/` contains the teleop ui
- `bodycontrolsd.py` gets all relevant input messages, processes them and sends the final `testJoystick` message (which the body executes).


## Running a model remotely
A very basic requirement to build cool applications on the body is to be able to:
- Receive image frames from the body to a remote computer
- Run the model on the frames, stream the outputs to the body via cereal messaging
- Control the body based on these model outputs

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
This creates a local VisionIPC client and receives the driver camera frames. We run the `YOLO5N` model on these frames using `onnxruntime`. There's post-processing code to parse the model outputs, compute bounding boxes and filter out objects with low probability. Finally, we publish the outputs to a socket by the message name `bodyReserved1`.

Note: cereal services in ZMQ mode listen on ports 8001-8079, make sure those are available

### 4. Control Body
```
ToDo: Complete this section
