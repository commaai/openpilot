## Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

Driving models include image, desire, and feature history in the ONNX graph. The runtime feeds each `next_<input>` output back into its matching input for the next frame.

After exporting a driving model, add history with `python add_history.py exported.onnx driving_supercombo.onnx` (requires `onnx`). Use `big_driving_supercombo.onnx` for the big model. The default frame skip is 4, matching 20 Hz inference and 5 Hz context.
