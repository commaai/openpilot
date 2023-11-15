## Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

## Supercombo
### Supercombo input format (Full size: 799906 x float32)
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
* **desire**
  * one-hot encoded buffer to command model to execute certain actions, bit needs to be sent for the past 5 seconds (at 20FPS) : 100 * 8
* **traffic convention**
  * one-hot encoded vector to tell model whether traffic is right-hand or left-hand traffic : 2
* **feature buffer**
  * A buffer of intermediate features that gets appended to the current feature to form a 5 seconds temporal context (at 20FPS) : 99 * 512
* **nav features**
  * 1 * 150
* **nav instructions**
  * 1 * 256


### Supercombo output format (Full size: XXX x float32)
Read [here](https://github.com/commaai/openpilot/blob/90af436a121164a51da9fa48d093c29f738adf6a/selfdrive/modeld/models/driving.h#L236) for more.


## Driver Monitoring Model
* .onnx model can be run with onnx runtimes
* .dlc file is a pre-quantized model and only runs on qualcomm DSPs

### input format
* single image W = 1440 H = 960 luminance channel (Y) from the planar YUV420 format:
  * full input size is 1440 * 960 = 1382400
  * normalized ranging from 0.0 to 1.0 in float32 (onnx runner) or ranging from 0 to 255 in uint8 (snpe runner)
* camera calibration angles (roll, pitch, yaw) from liveCalibration: 3 x float32 inputs

### output format
* 84 x float32 outputs = 2 + 41 * 2 ([parsing example](https://github.com/commaai/openpilot/blob/22ce4e17ba0d3bfcf37f8255a4dd1dc683fe0c38/selfdrive/modeld/models/dmonitoring.cc#L33))
  * for each person in the front seats (2 * 41)
    * face pose: 12 = 6 + 6
      * face orientation [pitch, yaw, roll] in camera frame: 3
      * face position [dx, dy] relative to image center: 2
      * normalized face size: 1
      * standard deviations for above outputs: 6
    * face visible probability: 1
    * eyes: 20 = (8 + 1) + (8 + 1) + 1 + 1
      * eye position and size, and their standard deviations: 8
      * eye visible probability: 1
      * eye closed probability: 1
    * wearing sunglasses probability: 1
    * face occluded probability: 1
    * touching wheel probability: 1
    * paying attention probability: 1
    * (deprecated) distracted probabilities: 2
    * using phone probability: 1
    * distracted probability: 1
  * common outputs 2
    * poor camera vision probability: 1
    * left hand drive probability: 1
