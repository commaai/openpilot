## Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

## Driving Model (vision model + temporal policy model)
### Vision inputs (Full size: 799906 x float32)
* **image stream**
  * Two consecutive images (256 * 512 * 3 in RGB) recorded at 20 Hz : 393216 = 2 * 6 * 128 * 256
    * Each 256 * 512 image is represented in YUV420 with 6 channels : 6 * 128 * 256
      * Channels 0,1,2,3 represent the full-res Y channel and are represented in numpy as Y[::2, ::2], Y[::2, 1::2], Y[1::2, ::2], and Y[1::2, 1::2]
      * Channel 4 represents the half-res U channel
      * Channel 5 represents the half-res V channel
* **wide image stream**
  * Two consecutive images (256 * 512 * 3 in RGB) recorded at 20 Hz : 393216 = 2 * 6 * 128 * 256
    * Each 256 * 512 image is represented in YUV420 with 6 channels : 6 * 128 * 256
      * Channels 0,1,2,3 represent the full-res Y channel and are represented in numpy as Y[::2, ::2], Y[::2, 1::2], Y[1::2, ::2], and Y[1::2, 1::2]
      * Channel 4 represents the half-res U channel
      * Channel 5 represents the half-res V channel
### Policy inputs
* **desire**
  * one-hot encoded buffer to command model to execute certain actions, bit needs to be sent for the past 5 seconds (at 20FPS) : 100 * 8
* **traffic convention**
  * one-hot encoded vector to tell model whether traffic is right-hand or left-hand traffic : 2
* **lateral control params**
  * speed and steering delay for predicting the desired curvature: 2
* **previous desired curvatures**
  * vector of previously predicted desired curvatures: 100 * 1
* **feature buffer**
  * a buffer of intermediate features including the current feature to form a 5 seconds temporal context (at 20FPS) : 100 * 512


### Driving Model output format (Full size: XXX x float32)
Refer to **slice_outputs** and **parse_vision_outputs/parse_policy_outputs** in modeld.


## Driver Monitoring Model
* `dmonitoring_model.onnx` is the source model and can be inspected with ONNX tooling.
* openpilot's runtime uses tinygrad artifacts generated from the ONNX model:
  * `dmonitoring_model_tinygrad.pkl` for model inference
  * `dmonitoring_model_metadata.pkl` for input shapes and output slices
  * `dm_warp_<camera_width>x<camera_height>_tinygrad.pkl` for the driver-camera luminance warp

### input format
* **input_img**: single image W = 1440 H = 960 luminance channel (Y) warped from the driver-camera NV12/YUV frame:
  * full input size is 1440 * 960 = 1382400
  * represented as a 1 x 1382400 uint8 tensor ranging from 0 to 255
* **calib**: camera calibration angles (roll, pitch, yaw) from liveCalibration: 1 x 3 float32 tensor

### output format
* 553 x float32 raw model outputs. Output names and slices are stored in the ONNX metadata and exported to `dmonitoring_model_metadata.pkl`.
* Per-driver outputs, first for `lhd` and then for `rhd`:
  * face descriptors: 12 floats
    * face orientation [pitch, yaw, roll] in camera frame: 3
    * face position [dx, dy] relative to image center: 2
    * normalized face size: 1
    * standard deviations for above outputs: 6
  * face visible probability: 1
  * left eye visible probability: 1
  * right eye visible probability: 1
  * wearing sunglasses probability: 1
  * left eye closed probability: 1
  * right eye closed probability: 1
  * using phone probability: 1
  * sleep probability: 1
* common outputs:
  * wheel-on-right probability: 1
