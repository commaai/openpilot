# Alpamayo 2 Edge replay and eGPU benchmark

Offline prototype for [NVIDIA Alpamayo 2 Edge](https://github.com/NVlabs/alpamayo2-edge). The vision encoder, language model, and diffusion expert run in tinygrad on Chestnut. NVIDIA's Python package supplies CPU preprocessing, tokenization, and trajectory decoding. `reference.py` runs NVIDIA's PyTorch implementation on a CUDA GPU for comparison.

This does not replace modeld or publish driving commands. Alpamayo predicts 64 poses over 6.4 seconds and has a different output contract from openpilot's model. The two-camera profile is also outside NVIDIA's released six-camera input specification; successful inference does not establish driving quality.

## Setup

Use a separate Python 3.12 environment so the reference dependencies do not change openpilot's runtime environment. From the openpilot repository root:

```bash
git clone https://github.com/NVlabs/alpamayo2-edge.git /tmp/alpamayo2-edge-upstream
git -C /tmp/alpamayo2-edge-upstream checkout 02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e
uv sync --locked --no-dev --project /tmp/alpamayo2-edge-upstream
export ALPAMAYO_PYTHON=/tmp/alpamayo2-edge-upstream/.venv/bin/python
export PYTHONPATH="$PWD/tinygrad_repo:$PWD:$PWD/opendbc_repo:$PWD/msgq_repo"
uv pip install --python "$ALPAMAYO_PYTHON" requests zstandard pycapnp==2.1.0
/tmp/alpamayo2-edge-upstream/.venv/bin/hf download nvidia/Alpamayo2-Edge \
  --revision b92b6aa326bd3e2235db28f248ad5a30b5890e64 --local-dir /tmp/alpamayo2-edge-model
```

Initialize openpilot's submodules and build its native dependencies before packing a replay. Use the repository's pinned tinygrad revision (`ec91b74b5b6b0fc924b72d3f27689679ca2438b9` at the initial benchmark). The BF16 checkpoint contains about 5.3 GB of weights; execution also needs memory for activations, attention caches, and JIT captures.

## Pack an openpilot segment

```bash
"$ALPAMAYO_PYTHON" openpilot/tools/alpamayo/replay.py \
  --rlog /absolute/path/rlog.zst \
  --road-video /absolute/path/fcamera.hevc \
  --wide-video /absolute/path/ecamera.hevc \
  --seconds 10 --navigation 'Continue straight' \
  --output /tmp/alpamayo-openpilot.npz
```

The segment must contain `narrowRoadEncodeIdx`, `wideRoadEncodeIdx`, valid `deviceMotion`, and calibrated `extrinsicsCalibration`. The packer selects five frames from each front camera at approximately -2, -1, -0.5, -0.1, and 0 seconds, and interpolates 20 ego poses at 10 Hz. It integrates logged velocity, applies calibration, and changes forward/right/down coordinates to forward/left/up. It checks timestamp coverage and motion gaps. Raw camera images retain openpilot's optics; no camera reprojection or missing surround views are synthesized.

## Run

```bash
DEV=USB+AMD:LLVM GMMU=0 AM_POWER_LIMIT=100 \
  "$ALPAMAYO_PYTHON" openpilot/tools/alpamayo/benchmark.py \
  --checkpoint /tmp/alpamayo2-edge-model --input /tmp/alpamayo-openpilot.npz \
  --greedy --warmup 2 --runs 3 --output /tmp/alpamayo-chestnut.json

CUDA_VISIBLE_DEVICES=1 "$ALPAMAYO_PYTHON" openpilot/tools/alpamayo/reference.py \
  --checkpoint /tmp/alpamayo2-edge-model --input /tmp/alpamayo-openpilot.npz \
  --greedy --warmup 2 --runs 3 --output /tmp/alpamayo-reference.json
```

Use `DEV=CUDA CUDA_VISIBLE_DEVICES=1` with `benchmark.py` to check the tinygrad implementation on an internal NVIDIA GPU. These results must be labeled separately from Chestnut results. `--synthetic` replaces `--input` for a constant-image smoke benchmark; `--profile nvidia --synthetic` exercises NVIDIA's full 22-image, six-camera input layout. Synthetic timings are not representative driving workloads, especially when reasoning lengths differ.

`--usb-pcie` uses synchronous PCIe transfers for inputs, weights, and outputs, bypassing tinygrad's SRAM staging path. On the initial Chestnut setup, the standard path failed buffer-boundary checks even with firmware `8516b768` and a 5 Gb/s USB connection. The direct path passed random-data roundtrips and GPU arithmetic at those boundaries, but uploaded at only about 3.7 MB/s: allow roughly 24 minutes to load the full checkpoint. Its timings must be identified as using this workaround. Model kernels still execute on Chestnut.

Reports preserve cold/warmup runs separately and include trajectories, reasoning text, generated-token counts, settings, and environment metadata. Native timings separate upload, vision, language prefill, reasoning, diffusion, and trajectory decoding. Model loading and preprocessing are recorded separately. Reference timing covers GPU-resident inference; it excludes input transfer and preprocessing. Greedy sampling makes text repeatable, but NumPy and PyTorch diffusion noise differ even with the same seed, so trajectories are not directly comparable unless noise is explicitly matched.

Before trusting eGPU timings, check the connected firmware against `openpilot/system/hardware/chestnut/flash.py` and run tinygrad's existing USB integrity checks:

```bash
cd tinygrad_repo
DEV=USB+AMD:LLVM GMMU=0 AM_POWER_LIMIT=100 "$ALPAMAYO_PYTHON" -m unittest \
  test.external.external_test_usb_asm24.TestUSBIntegrity.testBoundaries \
  test.external.external_test_usb_asm24.TestUSBIntegrity.testRepeatedBatches -v
```

Do not interpret timings from failed integrity checks or nonfinite predictions as model performance. The native implementation uses eager attention and growing KV caches; its timings measure this prototype, not an optimized Alpamayo deployment.
