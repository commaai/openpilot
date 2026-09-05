## Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

### Precompiled big model

Custom Chestnut model branches can include `big_driving_tinygrad.pkl.chunk*` and
`big_driving_tinygrad.precompiled.json`. When the marker matches the complete
chunk set and AMD target, SCons uses the host-compiled JIT instead of compiling ONNX on-device.
