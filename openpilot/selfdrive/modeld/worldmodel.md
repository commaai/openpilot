# Experimental 4B planner

The `worldmodeld` process runs checkpoint
`3b53ed52-c1d7-4765-8069-5bd6109d86ce/15360` with tinygrad on the USB RDNA4 GPU.
It publishes a plan at a target rate of 5 Hz. `modeld` converts that plan into
acceleration and curvature through the existing action helper, accounting for the
plan's age. The standard model continues publishing vision and odometry at 20 Hz
and supplies the plan whenever worldmodel output is invalid or stale.

The default uses **five live history frames**. The checkpoint was trained with five
future-conditioning images followed by ten past images. Live inference represents
the five future slots and five omitted older history slots with fixed Gaussian noise
at timestep 1. Encoded observations use timestep 0, and pose conditioning is masked.
The fixed prefix is cached once; all five observed frames are recomputed at their
current positional inputs. There is one transformer evaluation per plan, with all
56 blocks and the plan head, no image decoder, and no diffusion sampling loop.
This shorter-context input policy has not been validated for driving.

The transformer projections use FP8 weights and dynamic FP8 activations, FP32
accumulation, and BF16 outputs. RDNA4 kernels reuse 128 x 128 x 64 tiles in shared
memory and fuse scaling and bias. Live attention uses a fused RDNA4 kernel with
64 x 64 tiles, BF16 probabilities, FP32 accumulation, and FP8 KV storage. It retains
the original softmax summation and value-product rounding order; cache prefill uses
the original implementation. The image encoder uses FP16 matrix multiplies with
FP32 normalization and residuals.

On the connected 8 GB gfx1200 eGPU at a 100 W power limit, 600 synthetic
camera-to-plan runs paced at 5 Hz measured **169.36 ms median**, **175.79 ms maximum**,
and no runs above 200 ms.
This includes two NV12 camera images, NumPy preprocessing, USB input transfer,
image encoding, history update, the 4B planner, and plan download. These measurements
exclude concurrent on-device openpilot workloads. On identical inputs without camera
preprocessing, fused attention reduced the full runner's median from 219.67 to
165.39 ms at 100 W.

The fused attention path matched the original full-model output bit-for-bit across
32 synthetic frames, including a history reset. JIT replay matches eager execution
exactly after the images and history change.
The NumPy NV12 conversion and warp matched OpenCV pixel-for-pixel across twenty
random transforms. One five-history sample differed from the packaged BF16 model
with the same inputs and FP8 KV cache by 4.06% relative L2. These checks establish
runtime and numerical behavior on synthetic inputs, not driving quality.

Preparation requires access to the reporter checkpoint and internal encoder registry,
plus PyTorch and safetensors for conversion:

```bash
python -m openpilot.selfdrive.modeld.prepare_worldmodel /data/worldmodel
```

Set `WORLDMODEL_DIR=/data/worldmodel` in the manager environment to enable the
prototype and reserve the eGPU for it. The process uses tinygrad's USB AMD LLVM
backend and requires an LLVM library with RDNA4 support. The pinned tinygrad
submodule includes the native FP8 renderer support. The ordinary small driving
model supplies vision and actions during loading and history warmup.
