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

This section walks you through this process, running YOLO as an example.

### 1. Receive images
```
# On the remote computer, run 
./tools/camerastream/compressed_vipc.py --nvidia --cams 2 <body-ip>
```
This decodes the `driverEncodeData` packets from the body, and pushes it to a local VisionIPC server.


### 2. Run Model -> Send Outputs
```
# On another terminal window of the remote computer, run
./tools/bodyteleop/remote_models/yolo/yolo.py
```
This creates a local VisionIPC client and receives the driver camera frames. We run the `YOLO5N` model on these frames using `onnxruntime`. There's post-processing code to parse the model outputs, compute bounding boxes and filter out objects with low probability. Finally, we publish the outputs to a socket by the message name `bodyReserved1`.


### 3. Control Body
```
ToDo: Complete this section