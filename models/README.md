## Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

## Supercombo
### Supercombo input format (Full size: 393738 x float32)
* **image stream**
  * Two consecutive images (256 * 512 * 3 in RGB) recorded at 20 Hz : 393216 = 2 * 6 * 128 * 256
    * Each 256 * 512 image is represented in YUV420 with 6 channels : 6 * 128 * 256
      * Channels 0,1,2,3 represent the full-res Y channel and are represented in numpy as Y[::2, ::2], Y[::2, 1::2], Y[1::2, ::2], and Y[1::2, 1::2]
      * Channel 4 represents the half-res U channel
      * Channel 5 represents the half-res V channel
* **desire**
  * one-hot encoded vector to command model to execute certain actions, bit only needs to be sent for 1 frame : 8
* **traffic convention**
  * one-hot encoded vector to tell model whether traffic is right-hand or left-hand traffic : 2
* **recurrent state**
  * The recurrent state vector that is fed back into the GRU for temporal context : 512


### Supercombo output format (Full size: 6472 x float32)
* **plan**
  * 5 potential desired plan predictions : 4955 = 5 * 991
    * predicted mean and standard deviation of the following values at 33 timesteps : 990 = 2 * 33 * 15
      * x,y,z position in current frame (meters)
      * x,y,z velocity in local frame (meters/s)
      * x,y,z acceleration local frame (meters/(s*s))
      * roll, pitch , yaw in current frame (radians)
      * roll, pitch , yaw rates in local frame (radians/s)
    * probability[^1] of this plan hypothesis being the most likely: 1
* **lanelines**
  * 4 lanelines (outer left, left, right, and outer right): 528 = 4 * 132
    * predicted mean and standard deviation for the following values at 33 x positions : 132 = 2 * 33 * 2
      * y position in current frame (meters)
      * z position in current frame (meters)
* **laneline probabilties**
  * 2 probabilities[^1] that each of the 4 lanelines exists : 8 = 4 * 2
    * deprecated probability
    * used probability
* **road-edges**
  * 2 road-edges (left and right): 264 = 2 * 132
    * predicted mean and standard deviation for the following values at 33 x positions : 132 = 2 * 33 * 2
      * y position in current frame (meters)
      * z position in current frame (meters)
* **leads**
  * 2 hypotheses for potential lead cars : 102 = 2 * 51
    * predicted mean and stadard deviation for the following values at 0,2,4,6,8,10s : 48 = 2 * 6 * 4
      * x position of lead in current frame (meters)
      * y position of lead in current frame (meters)
      * speed of lead (meters/s)
      * acceleration of lead(meters/(s*s))
    * probabilities[^1] this hypothesis is the most likely hypothesis at 0s, 2s or 4s from now : 3
* **lead probabilities**
  * probability[^1] that there is a lead car at 0s, 2s, 4s from now : 3 = 1 * 3
* **desire state**
  * probability[^1] that the model thinks it is executing each of the 8 potential desire actions : 8
* **meta** [^2]
  * Various metadata about the scene : 80 = 1 + 35 + 12 + 32
    * Probability[^1] that openpilot is engaged : 1
    * Probabilities[^1] of various things happening between now and 2,4,6,8,10s : 35 = 5 * 7
      * Disengage of openpilot with gas pedal
      * Disengage of openpilot with brake pedal
      * Override of openpilot steering
      * 3m/(s*s) of deceleration
      * 4m/(s*s) of deceleration
      * 5m/(s*s) of deceleration
    * Probabilities[^1] of left or right blinker being active at 0,2,4,6,8,10s : 12 = 6 * 2
    * Probabilities[^1] that each of the 8 desires is being executed at 0,2,4,6s : 32 = 4 * 8

* **pose** [^2]
  * predicted mean and standard deviation of current translation and rotation rates : 12 = 2 * 6
    * x,y,z velocity in current frame (meters/s)
    * roll, pitch , yaw rates in current frame (radians/s)
* **recurrent state**
  * The recurrent state vector that is fed back into the GRU for temporal context : 512

[^1]: All probabilities are in logits, so you need to apply sigmoid or softmax functions to get actual probabilities
[^2]: These outputs come directly from the vision blocks, they do not have access to temporal state or the desire input

