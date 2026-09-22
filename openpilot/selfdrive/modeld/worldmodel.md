# Context-9 transformer policy at 5 Hz

This branch runs actor `58e4f1d2-6827-495e-bc3d-6138883170f0/12864`, the latest
complete checkpoint available on 2026-09-22. Its frozen backbone is
`68a03682-c802-4638-a4f7-c04707a8a579/15360`; the image encoder is
`c04337f8-b83f-4e34-b07a-5f7396978d67/-1`. This run uses `actor` weights and has
no separate target actor.

`models/worldmodel/model.pkl` contains FP8 E4M3 matrix weights, the INT8 encoder,
higher-precision small parameters, compiled GPU kernels, and Linux ARM64/x86-64
host programs. Its size is 3,993,156,695 bytes (3.99 GB / 3.72 GiB). The artifact
is stored in Git LFS and targets the USB AMD gfx1200 GPU. `hparams.json` and the
PKL metadata pin the checkpoints and training input contract.

## Setup

From the repository root:

```bash
git submodule update --init --recursive
git lfs pull
```

The planner starts onroad by default and reserves the USB GPU. The ordinary
model continues on the device's standard backend, supplying lanes, leads,
metadata and odometry at 20 Hz. Its plan and action remain the fallback during
loading, history warmup, and whenever the worldmodel message is invalid or
stale. Valid worldmodel predictions automatically control the plan and action;
this is not a shadow-only configuration.

`WORLDMODEL_DIR=/absolute/path/to/compiled-model` selects another compatible
artifact directory. Set `WORLDMODEL_DIR=` in the manager environment to disable
the worldmodel. Run the publisher alone with:

```bash
python -m openpilot.selfdrive.modeld.worldmodeld
```

Loading uploads the weight arena in 32 MiB chunks using a precompiled transfer
program. Startup also links and warms the inference kernels, then clears the
history. The device does not require source weights, ONNX, Clang, or LLVM.
The GPU architecture and pinned tinygrad revision must match the offline build.

## Model and control integration

Each input combines narrow and wide RGB images at 256 x 128. The encoder adds
one latent to a nine-frame history. All nine frames pass through the 56-block
backbone and a three-block transformer policy head. There is no noise prefix,
image decoder, diffusion loop, or cross-window KV cache.

The policy head uses BF16 residuals, RMSNorm, Q/K normalization, and frame-causal
attention. Every frame sees its own spatial tokens and earlier frames. The full
backbone output is retained. Only the final policy block discards earlier query
outputs, while retaining all keys and values. Large policy matrices use the
native FP8 kernels; final output projections and scales compute in FP32.

The head consumes two `action_t` values. The publisher combines vehicle lateral
and longitudinal delays, output smoothing, camera age, the previous measured
inference duration, and the 100 ms half-period. It publishes the conditioned
990-value plan, four-value action distribution, and conditioning times.
`modeld` parses the learned action, converts lateral acceleration to curvature,
and applies the existing output smoothing and stop logic. It retains plan-based
action derivation for older artifacts without an action output.

The publisher takes every fourth 20 Hz camera frame. Service health checks use
the 5 Hz service frequency. Stale-plan expiry is 450 ms: the existing 50 ms
camera-delivery budget plus two 200 ms periods for inference and holding the
result until its replacement. A camera gap over 400 ms or a camera restart
resets history; nine new observations are required for validity. `modeld`
receives the latest worldmodel output after running the small model, so a plan
arriving during that inference is available for the freshness check. The
prediction time grid is unchanged.

History advances at the trained 5 Hz, spanning 1.6 seconds. The fused attention
kernel preserves frame causality, softmax reduction order, BF16 probability
rounding, and FP32 accumulation order. It produces bitwise-identical outputs
to the unfused quantized implementation on the synthetic validation sequence.

## Offline compilation

Export the source bundle with xx's `ml_tools/openpilot_compile/compile_worldmodel.py`:

```bash
python ml_tools/openpilot_compile/compile_worldmodel.py \
  --rldriving 58e4f1d2-6827-495e-bc3d-6138883170f0/12864 \
  --output /absolute/path/to/context9-export
```

With this branch, its pinned tinygrad, and the target USB GPU on the build host:

```bash
python -m openpilot.selfdrive.modeld.compile_worldmodel \
  /absolute/path/to/context9-export \
  openpilot/selfdrive/modeld/models/worldmodel/model.pkl
```

The host needs Clang and LLVM with RDNA4 support. Compilation uses
`TC_OPT=2 TC_MIN_GLOBALS=32 JIT_BATCH_SIZE=0` and cross-compiles host programs
for ARM64. A neighboring `model.reference.npz` contains 32 synthetic images and
varying action delays, with outputs for cold and repeating histories. This
reference file is not deployed. Rebuild when changing the model or tinygrad.

## Validation and hardware limits

Chestnut CI runs the existing stock-model replay and camera tests with
`WORLDMODEL_DIR=`. It then explicitly enables the worldmodel for
`openpilot/selfdrive/test/test_worldmodel.py` on the same MICI GPU device.
That test starts `worldmodeld` with the real cameras and ordinary model,
waits for valid worldmodel output to be consumed by `modeld`, and checks
25 seconds of 5 Hz plans, finite plan/action values, freshness, inference
deadlines, and continued worldmodel use. Any worldmodel compiler call fails
the test. Non-Chestnut jobs disable the worldmodel and skip its LFS download.

Offline validation compares 64 predictions across cold and repeating histories.
Every plan and learned action matches the unfused implementation bit for bit.
A fresh process runs with compilation disabled, checks the 100 W power cap,
and verifies the outputs against the compiler reference. Changing the two
action delays with image history held constant changes both predictions;
restoring the delays reproduces the original outputs exactly. Python garbage
collection is disabled in the publisher to avoid pauses during inference.

At 100 W, 160 fresh-process predictions measured 196.25 ms median, 198.08 ms
p95 and 201.09 ms maximum, with one 200 ms deadline missed. These timings
exclude camera preprocessing; reliable 5 Hz is not yet established. The PKL
compiled in 159.84 seconds and loaded in 13.39 seconds without compiler calls.

The previous checkpoint's 4 Hz Chestnut results and the older planner's 5 Hz
results do not validate this artifact. This checkpoint must pass the dedicated
Chestnut test at its configured 5 Hz before its timing is considered verified.
Quantization and learned action timing also require driving-data validation.

The planner defaults `AM_POWER_LIMIT` to 100 W and uses automatic GPU clocks.
An explicit environment setting overrides the cap. The test setup has one
100 W, 12 V supply for both the GPU and bridge. The previous model lost PCIe
routing during repeated tests at caps of 111 W and above. Short passes at higher
caps were not reliable. Power delivery is the leading reset hypothesis, but
voltage droop was not measured.

The pinned commaai/tinygrad fork checks USB errors and transfer lengths, bounds
completion waits, and rejects further work after failure. It also fixes shifted
history assignment so overlapping GPU waves cannot overwrite data still being
read. These fixes do not prevent hardware PCIe resets.

The x86-64 timing measurements use prepared images and exclude camera
preprocessing and concurrent openpilot operation. Chestnut CI covers the real
camera/model pipeline; sustained reliability, full onroad system load, and
driving behavior still require validation. Start with parked-car integration
testing with controls disengaged.
