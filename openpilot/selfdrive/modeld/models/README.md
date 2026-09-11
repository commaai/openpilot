## Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

The big driving model ships as one LFS pickle, compiled for Chestnut's USB AMD GPU
(`gfx1200`) and both camera resolutions. Regenerate it from xx with
`ml_tools/openpilot_compile/compile_torchtitan_supercombo.py --rldriving <checkpoint>
--onnx-path /tmp/big_driving_supercombo.onnx --compiled-path <openpilot>/openpilot/selfdrive/modeld/models/big_driving_tinygrad.pkl`.
Use openpilot's pinned tinygrad version. The ONNX is only an intermediate export.
