# Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

## Supercombo input format (Full size: 393738 x float32)
##### Image strean
  * Two consecutive images (256 * 512 * 3 in RGB) recorded at 20 Hz : 393216 = 2 * 6 * 128 * 256
    * Each 256 * 512 image is represented in YUV with 6 channels : 6 * 128 * 256
      * 1/6 : top left pixels of Y channel
      * 2/6 : top right pixels of Y channel
      * 3/6 : bottom left pixels of Y channel
      * 4/6 : bottom right pixels of Y channel
      * 5/6 : pixels of U channel
      * 6/6 : pixels of V channel


## Supercombo output format (Full size: 6472 x float32)
##### plan
  * 5 potential plan predictions : 4955 = 5 * 991
    * predicted mean and standard deviation of the following values at 33 timesteps : 990 = 2 * 33 * 15
      * x,y,z position in current frame (meters)
      * x,y,z velocity in local frame (meters/s)
      * x,y,z acceleration local frame (meters/(s*s))
      * roll, pitch , yaw in current frame (radians)
      * roll, pitch , yaw rates in current frame (radians/s)
    * probability of this plan hypothesis being the most likely: 1
##### lanelines
  * 4 lanelines (outer left, left, right, and outer right): 528 = 4 * 132
    * predicted mean and standard deviation fir the following values at 33 x positions : 132 = 2 * 33 * 2
      * y position in current frame (meters)
      * z position in current frame (meters)
##### laneline probabilties
  * 2 probabilities that each of the 4 lanelines exists : 8 = 4 * 2
    * deprecated probability
    * used probibility
##### road-edges 
  * 2 road-edges (left and right): 264 = 2 * 132
    * predicted mean and standard deviation fir the following values at 33 x positions : 132 = 2 * 33 * 2
      * y position in current frame (meters)
      * z position in current frame (meters)
###### leads
  * 2 hypotheses for potential lead cars : 102 = 2 * 51
    * predicted mean and stadard deviation for the following values at 6 timesteps : 48 = 2 * 6 * 4 
      * x position of lead in current frame (meters)
      * y position of lead in current frame (meters)
      * speed of lead in current frame (meters/s)
      * acceleration of lead in current frame (meters/(s*s))
    * probabilities this hypothesis is the most likely hypothesis at 0s, 2s or 4s from now : 3 
##### lead probabilities : 3
  * probability that there is a lead car at 0s, 2s, 4s from now : 3 = 1 * 3
##### desire state : 8
  * probability that the model thinks it is executing each of the 8 potential desire actions
##### meta
  * Various meta information : 80
##### pose
  * predicted mean and standard deviation of current translation and rotation rates : 12 = 2 * 6
    * x,y,z velocity in current frame (meters/s)
    * roll, pitch , yaw rates in current frame (radians/s)
##### recurrent state : 512
  * The recurrent state vector that is fed back into the GRU for temporal context
