## Neural networks in openpilot
To view the architecture of the ONNX networks, you can use [netron](https://netron.app/)

The big driving model ships as precompiled `big_driving_tinygrad.pkl.chunk*`
files for Chestnut's USB AMD GPU. SCons compiles the small driving and driver
monitoring models locally. The big-model pickle contains fused frame preparation
and policy JITs for both 1928x1208 and 1344x760 cameras, compiled for `gfx1200`.

To regenerate it, first export the desired checkpoint from the xx repository:

```sh
python ml_tools/openpilot_compile/compile_torchtitan_supercombo.py \
  --rldriving a4c5f1d1-1f5d-4807-9593-54afa27099d5/12864 \
  --path b9facbcc-4d47-410e-b3ce-dfcbad12ba92/56320 \
  --onnx-path /tmp/big_driving_supercombo.onnx
```

Then, from the openpilot checkout with its pinned tinygrad submodule and a
connected Chestnut GPU:

```sh
python openpilot/selfdrive/modeld/precompile_modeld.py /tmp/big_driving_supercombo.onnx
```

This uses the USB AMD settings from `xx/ml_tools/openpilot_compile/compile_tinygrad.py`,
compiles both camera resolutions, checks pickle replay with two random seeds,
and writes the LFS chunks, chunk manifest, and `big_driving_tinygrad.json`.
Commit those files together. The JSON records the source checkpoint, ONNX and
pickle hashes, compiler fingerprints, and chunk sizes. Unit tests require
regeneration when tinygrad or the model compiler sources change.

The exported ONNX is an intermediate file; it is not shipped in openpilot.
Non-Chestnut CI and release builds exclude the precompiled big model.
