## Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

Driving models include image, desire, and feature history in the ONNX graph. The runtime feeds each `next_<input>` output back into its matching input for the next frame.
