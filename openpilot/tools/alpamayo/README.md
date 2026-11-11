# Alpamayo 2 Edge replay and eGPU benchmark

Offline prototype for [NVIDIA Alpamayo 2 Edge](https://github.com/NVlabs/alpamayo2-edge). The vision encoder, language model, and diffusion expert run in tinygrad on NVIDIA GPUs or Chestnut. NVIDIA's Python package supplies CPU preprocessing, tokenization, and trajectory decoding. `reference.py` runs NVIDIA's PyTorch implementation on a CUDA GPU for comparison.

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

Initialize openpilot's submodules and build its native dependencies before packing a replay. Use the repository's pinned tinygrad revision. This branch pins `89846dcfb383a98ee01cfec3bba440ae7dcf9b1e` from commaai's `alpamayo-egpu-runtime` branch to fix USB graph submission; the earlier 5090 measurements used its parent, `ec91b74b5b6b0fc924b72d3f27689679ca2438b9`. The BF16 checkpoint contains about 5.3 GB of weights; execution also needs memory for activations, attention caches, and JIT captures.

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

### Native tinygrad FP8 on RTX 5090

```bash
DEV=NVK:0+NV:CUDA "$ALPAMAYO_PYTHON" openpilot/tools/alpamayo/benchmark.py \
  --checkpoint /tmp/alpamayo2-edge-model --input /tmp/alpamayo-openpilot.npz \
  --greedy --precision fp8 --vision-batch 10 --warmup 3 --runs 10 \
  --output /tmp/alpamayo-tinygrad-fp8.json
```

On the benchmark machine, native `NVK:0` selects physical GPU 1, UUID `GPU-f7dbd668-2d4e-06e8-e258-c45142a5d370`. Native NV enumeration differs from `nvidia-smi`; the report records `nvidia_gpu_index`. This path was tested with CUDA 13.1, Triton 3.4.0, and the pinned tinygrad revision above.

tinygrad owns model execution, buffers, fixed-capacity KV caches, and JIT replay. CUDA custom kernels handle E4M3 matrix multiplication, single-token decoding, normalization, and activation scaling. Triton compiles the fused BF16 attention kernel; tinygrad loads its binary and executes it through the native NV runtime. Model inference makes no PyTorch GPU calls. The CPU input and trajectory adapters still use NVIDIA's Python package.

`--precision fp8` quantizes the same 504 aligned linear layers as the reference, with dynamic per-tensor activation scaling and FP32 accumulation. Attention and KV caches stay BF16. Input transfers, image-to-token assembly, prefill, decoding, and all ten diffusion steps use reusable graphs. Every inference recomputes all ten images; there is no image reuse or reasoning truncation. Each call overwrites its prompt cache and masks unused cache slots. KV state is only reused while generating that call's reasoning and trajectory.

Measured on 2026-11-10 using the same two replay samples as the NVIDIA reference below:

| Replay time | Generated tokens | Median | p95 | Inferences/s |
| --- | --- | --- | --- | --- |
| 10 seconds | 6 | 183.34 ms | 184.33 ms | 5.45 |
| 20 seconds | 20 | 242.51 ms | 244.11 ms | 4.12 |

Each result uses ten measured runs after three warmups on physical GPU 1. Native timing includes input transfer and CPU trajectory decoding; preprocessing, weight loading, graph capture, and one garbage collection after warmup are recorded separately. The short sample is 4.83 times faster than the earlier 885.99 ms BF16 tinygrad baseline with fixed caches. It remains slower than the 96.51/129.80 ms NVIDIA FP8 reference, whose timing excludes input transfer.

All measured trajectories were finite and repeatable. Validation alternated both samples through one model, replaced contents of existing CPU input buffers, and changed the generation limit from 256 to 64 tokens without changing the resulting trajectory. Both reasoning texts matched NVIDIA's reference. With identical diffusion noise, native BF16 versus NVIDIA BF16 mean/max waypoint distances were 0.017/0.049 m and 0.036/0.095 m; native FP8 versus NVIDIA FP8 distances were 0.122/0.402 m and 0.180/0.560 m, respectively, over the 6.4-second horizon. These are numerical comparisons on two replay samples.

### Comma4 and Chestnut FP8

On 2026-11-10, a comma4 with Chestnut gfx1200, 8 GB VRAM, firmware `8516b768`, a direct 5 Gb/s USB connection, and a 100 W power limit completed the same 10-second replay sample: two cameras, ten images, 2,026 prompt tokens, six generated tokens, and ten diffusion steps.

| Stage | Previous scalar FP32 multiplication | FP8 WMMA |
| --- | --- | --- |
| Input upload | 166.07 ms | 169.44 ms |
| Vision | 7,810.55 ms | 1,830.66 ms |
| Prefill | 55,114.21 ms | 3,766.50 ms |
| Reasoning | 1,657.56 ms | 1,603.58 ms |
| Diffusion | 714.25 ms | 377.70 ms |
| CPU trajectory decode | 12.69 ms | 12.93 ms |
| Total | **65,503.29 ms** | **7,761.87 ms** |
| Inferences/s | 0.0153 | 0.1288 |
| Total p95 | 65,509.74 ms | 7,803.47 ms |

Stage and total times are medians. The scalar baseline at `b27d932dee` used three measured runs; default FP8 WMMA at `48fae630c7` used ten. Both used three warmups, the same prequantized FP8 weights and scales, and verified every uploaded checkpoint byte on the GPU. Timing includes input upload and CPU trajectory decoding, and excludes loading, preprocessing, compilation, graph capture, and garbage collection after warmup.

The earlier AMD implementation cast FP8 operands to FP32 before matrix multiplication, preventing FP8 matrix instructions. On gfx1200/gfx1201, multi-row linear layers now keep FP8 operands and request FP32 accumulation, allowing tinygrad to select WMMA. `TC_OPT=2` permits padding the 2,026-token prefill to the hardware tile size. That initial WMMA change left single-token decoding on its scalar fallback; attention and KV caches remain BF16. The NVIDIA custom kernels are unchanged.

This reduced total latency by 8.44 times and prefill by 14.63 times. All ten measured outputs were finite and identical to each other, with the same reasoning as the scalar baseline. With identical diffusion noise, mean/max waypoint displacement from that baseline was 0.0166/0.0360 m across 64 poses. This is one replay sample; 0.129 inferences/s remains below the 5 FPS target.

The runtime fix submits compute before USB copy staging, which can otherwise block while waiting for computation that has not yet been submitted. The combined input upload/cast graph passed five runs with changing inputs and exact array comparisons. All seven existing USB integrity tests also passed. Keep the compiler cache on `/data`: comma4's small `/home` overlay filled during compilation. The longer GPU wait limit also covers cold execution and graph capture:

```bash
DEV=USB+AMD:LLVM GMMU=0 AM_POWER_LIMIT=100 TC_OPT=2 CACHEDB=/data/alpamayo-cache/cache.db \
  "$ALPAMAYO_PYTHON" openpilot/tools/alpamayo/benchmark.py \
  --checkpoint /tmp/alpamayo2-edge-model --input /tmp/alpamayo-openpilot.npz \
  --precision fp8 --vision-batch 10 --greedy --gpu-timeout-ms 300000 \
  --warmup 3 --runs 10 --output /tmp/alpamayo-comma4.json
```

This command quantizes the original BF16 checkpoint during loading. The measured runs instead uploaded the same prequantized weights in bulk. The WMMA run used typed buffer views for each weight so kernels could be reused across layers without recompiling for each shard offset. Initialization timing and compilation cost differ; both paths use the same inference code.

#### Packed feed-forward kernels

The default WMMA kernels still read matrix operands inefficiently and chose small tiles for the 2,026-token prefill. `amd_gemm.py` uses 64-by-64 output tiles and packs operands into 16-by-16 fragments with contiguous loads across a wave. The vision and language feed-forward weights are packed once during loading; a logical view preserves access for single-token decoding. Activation packing is included in inference timing. The kernel accumulates in FP32 and fuses scaling, optional bias, and output conversion. It applies to feed-forward calls with at least 128 rows on gfx1200/gfx1201; NVIDIA execution is unchanged.

On comma4 and Chestnut, five matrix checks covering both language feed-forward shapes, both vision feed-forward shapes, and partial tiles produced bit-identical BF16 outputs to default WMMA. A full language layer with released weights, 2,026 random input tokens, attention, and cache writes improved from 137.16 to 44.73 ms with bit-identical output. Its two feed-forward GPU kernels measured 0.877 and 0.842 ms in the layer profile.

The full 10-second replay sample then completed ten measured runs after three warmups:

| Stage | Default FP8 WMMA | Packed feed-forward kernels |
| --- | --- | --- |
| Input upload | 169.44 ms | 168.70 ms |
| Vision | 1,830.66 ms | 1,635.56 ms |
| Prefill | 3,766.50 ms | 1,229.02 ms |
| Reasoning | 1,603.58 ms | 1,628.02 ms |
| Diffusion | 377.70 ms | 380.41 ms |
| CPU trajectory decode | 12.93 ms | 12.58 ms |
| Total median | **7,761.87 ms** | **5,058.10 ms** |
| Total p95 | 7,803.47 ms | 5,067.97 ms |
| Inferences/s | 0.1288 | 0.1977 |

At `0b8c76d4d4`, prefill improved by 3.06 times and total inference by 1.53 times. The workload, checkpoint bytes, precision, six reasoning tokens, ten diffusion steps, and timing boundaries were unchanged. Every measured reasoning string, trajectory, and rotation matrix exactly matched the default WMMA baseline and every other measured run. All outputs were finite.

#### Tiled attention and packed projections

`amd_attention.py` uses BF16 WMMA for both attention matrix multiplies, with 64-by-64 output tiles and packed operands. Grouped KV heads are indexed without repeating their buffers. A fused row softmax writes probabilities directly in the layout consumed by the second multiply. Scores and probabilities retain BF16 rounding, with FP32 accumulation and softmax reductions. The score matrix is still materialized. The packed FP8 linear kernels also handle vision and language Q/K/V and output projections.

The attention path applies to BF16 inputs with at least 128 query rows on gfx1200/gfx1201 using the AMD LLVM renderer, with tensor cores enabled and no explicit mask. At `08b0693922`, single-token decoding, diffusion, and explicitly masked attention retained the portable path. NVIDIA execution is unchanged.

Kernel checks covered batched inputs, grouped KV heads, partial tiles, causal and noncausal attention, and changing inputs during JIT replay. A complete language layer with released weights and 2,026 random input tokens improved from 45.02 to 10.00 ms. Its maximum/mean absolute output differences were 0.00390625/0.00000265 relative to the previous implementation.

The same comma4 and Chestnut replay benchmark completed three warmups and ten measured inferences:

| Stage | Packed feed-forward kernels | Tiled attention and packed projections |
| --- | --- | --- |
| Input upload | 168.70 ms | 169.16 ms |
| Vision | 1,635.56 ms | 382.86 ms |
| Prefill | 1,229.02 ms | 247.79 ms |
| Reasoning | 1,628.02 ms | 1,643.00 ms |
| Diffusion | 380.41 ms | 382.92 ms |
| CPU trajectory decode | 12.58 ms | 13.12 ms |
| Total median | **5,058.10 ms** | **2,844.98 ms** |
| Total p95 | 5,067.97 ms | 2,853.72 ms |
| Inferences/s | 0.1977 | 0.3515 |

Prefill improved by 4.96 times, vision by 4.27 times, and total inference by 1.78 times. Checkpoint bytes, precision, input, six reasoning tokens, ten diffusion steps, and timing boundaries were unchanged. All measured outputs were finite and identical to each other, and reasoning matched the previous implementation. With identical diffusion noise, mean/max waypoint displacement from that implementation was 0.0548/0.2041 m over 64 poses across the 6.4-second horizon. The changed attention reduction order is not bit-identical; this comparison covers one replay sample. Reasoning accounted for about 58% of latency in that revision.

#### Single-token decoding

`amd_decode.py` adds matrix-vector kernels that read the existing packed FP8 weights, plus a row-major variant for the vocabulary head. Both accumulate in FP32 and fuse scaling, bias, and output conversion. Their explicit inputs prevent activation quantization and scale reductions from being recomputed inside the multiply. This also accelerates the vocabulary head at the end of prefill.

AMD decoding now reads a fixed-capacity KV cache and masks unused slots, avoiding reductions with a changing key dimension. Specialized single-query BF16 attention uses a row softmax and divides the value reduction into 128-key chunks. The final reduction accumulates those chunks in FP32. These kernels use the AMD LLVM renderer on gfx1200/gfx1201; the GEMV path requires widths divisible by 128, and the attention path handles 128-wide heads with a shared key mask. NVIDIA execution and the diffusion path are unchanged.

Twenty GEMV checks covered five shapes, packed and row-major weights, bias, changing inputs and scales during JIT replay, and zero inputs. All outputs were bit-identical to the previous FP32-accumulating implementation after BF16 conversion. Ten attention checks covered batched grouped KV heads, partial tiles, changing cache masks, and zero/constant inputs; maximum absolute error was 0.001953125. Masked future cache entries were filled with large values to check that they could not affect the result.

The same comma4 and Chestnut replay completed three warmups and ten measured inferences:

| Stage | Tiled attention and packed projections | Specialized decoding |
| --- | --- | --- |
| Input upload | 169.16 ms | 168.88 ms |
| Vision | 382.86 ms | 381.06 ms |
| Prefill | 247.79 ms | 227.55 ms |
| Reasoning | 1,643.00 ms | 157.21 ms |
| Diffusion | 382.92 ms | 383.51 ms |
| CPU trajectory decode | 13.12 ms | 12.96 ms |
| Total median | **2,844.98 ms** | **1,336.07 ms** |
| Total p95 | 2,853.72 ms | 1,341.62 ms |
| Inferences/s | 0.3515 | 0.7485 |

Reasoning improved by 10.45 times and total inference by 2.13 times. The input, checkpoint bytes, precision, six reasoning tokens, ten diffusion steps, and timing boundaries were unchanged. Every measured output was finite and identical to the other repeats, and reasoning matched the previous implementation. With identical noise, mean/max waypoint displacement was 0.00912/0.02753 m across 64 poses. This is one replay sample. Vision and diffusion now account for about 57% of latency; 0.7485 inferences/s remains below the 5 FPS target.

### NVIDIA accelerated reference on RTX 5090

`reference.py` defaults to NVIDIA's SDPA attention backend. `--attention eager` reproduces the initial reference below. Use NVIDIA's released `torch.compile`, CUDA graphs, and FlashAttention 2 acceleration with `--accel tuned`; this leaves the weights in BF16. Install the pinned FlashAttention wheel from [NVIDIA's acceleration guide](https://github.com/NVlabs/alpamayo2-edge/blob/02f55a7aeabc1a2b9f7a80445c08e40bddc89f5e/docs/acceleration.md) into the separate reference environment first.

```bash
# Select the intended GPU by its UUID from nvidia-smi.
CUDA_VISIBLE_DEVICES="$ALPAMAYO_GPU_UUID" \
  TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/alpamayo2-edge/inductor" \
  TRITON_CACHE_DIR="$HOME/.cache/alpamayo2-edge/triton" \
  "$ALPAMAYO_PYTHON" openpilot/tools/alpamayo/reference.py \
  --checkpoint /tmp/alpamayo2-edge-model --input /tmp/alpamayo-openpilot.npz \
  --greedy --attention sdpa --accel tuned --warmup 3 --runs 10 \
  --output /tmp/alpamayo-reference-tuned.json
```

On 2026-11-10, the same real two-camera sample (ten images, 2,026 prompt tokens, six generated tokens, ten diffusion steps) measured:

| NVIDIA runtime | Median | p95 | Inferences/s |
| --- | --- | --- | --- |
| PyTorch SDPA | 268.10 ms | 268.75 ms | 3.73 |
| PyTorch SDPA + `--accel tuned` | 138.87 ms | 140.19 ms | 7.20 |

Each result uses ten measured runs after three warmups on RTX 5090 GPU 1, UUID `GPU-f7dbd668-2d4e-06e8-e258-c45142a5d370`. Before testing, this GPU had 8.1 GiB allocated and 6% utilization from existing workloads; GPU 0 was busy and was not used. The tuned run built three CUDA graphs in 41.58 seconds, with no skipped stages or eager fallbacks. The report records the physical GPU UUID and acceleration details. All measured outputs were finite and repeatable; with identical noise, the tuned/default mean waypoint difference was 3.7 mm (maximum 6.7 mm), and reasoning matched. These are inference timings for one replay sample, excluding CPU preprocessing and input transfer.

NVIDIA's advertised Thor deployment uses a separate TensorRT/Edge-LLM FP8 recipe with speculative decoding and rolling image reuse. The BF16 results above use the public PyTorch accelerator and recompute all ten images every inference.

### FP8 reference

`--precision fp8` applies TorchAO W8A8 quantization before NVIDIA's accelerator compiles the model. The 504 aligned linear layers (2.37 billion weights) use E4M3 FP8 weights and dynamically scaled FP8 activations. Attention and KV caches remain BF16; embeddings, normalization, and the two unaligned action projections retain their original precision. Linear inputs honor BF16 autocast so TorchAO does not propagate FP32 intermediates through the quantized layers.

Use TorchAO 0.13.0 with the pinned PyTorch 2.8 environment above:

```bash
uv pip install --python "$ALPAMAYO_PYTHON" --no-deps torchao==0.13.0
CUDA_VISIBLE_DEVICES="$ALPAMAYO_GPU_UUID" \
  TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/alpamayo2-edge/inductor" \
  TRITON_CACHE_DIR="$HOME/.cache/alpamayo2-edge/triton" \
  "$ALPAMAYO_PYTHON" openpilot/tools/alpamayo/reference.py \
  --checkpoint /tmp/alpamayo2-edge-model --input /tmp/alpamayo-openpilot.npz \
  --greedy --precision fp8 --attention flash_attention_2 --accel tuned --compile-vision \
  --warmup 3 --runs 10 --trace /tmp/alpamayo-fp8-trace.json \
  --output /tmp/alpamayo-fp8.json
```

`--compile-vision` also compiles the vision encoder layers and projector; NVIDIA's accelerator handles language prefill, decoding, and the diffusion expert. `--trace` profiles one additional inference after the timed runs and records the observed FP8 CUDA kernel names. Compilation, graph capture, and profiling are excluded from steady-state latency. This is a PyTorch/TorchAO FP8 runtime, separate from NVIDIA's TensorRT deployment and the native tinygrad prototype.

Matched measurements on the same RTX 5090 GPU 1, with FlashAttention 2 and compiled vision enabled for both precisions:

| Replay time | Generated tokens | Precision | Median | p95 | Inferences/s |
| --- | --- | --- | --- | --- | --- |
| 10 seconds | 6 | BF16 | 117.05 ms | 118.90 ms | 8.54 |
| 10 seconds | 6 | FP8 | 96.51 ms | 96.63 ms | 10.36 |
| 20 seconds | 20 | BF16 | 158.31 ms | 161.75 ms | 6.32 |
| 20 seconds | 20 | FP8 | 129.80 ms | 129.87 ms | 7.70 |

Both samples come from `0eb6c3999e01ba0f/00000009--85cc17e716/54`, with two cameras, five frames per camera, 2,026 prompt tokens, and ten diffusion steps. Each measurement uses three warmups and ten timed runs. The GPU trace confirms E4M3 tensor-core GEMMs with BF16 outputs; all three NVIDIA CUDA graphs ran without fallbacks. Every repeated output was finite and identical within its configuration. Latency includes trajectory decoding but excludes preprocessing and input transfer. Longer reasoning keeps the second sample below 10 Hz.

BF16 and FP8 produced identical reasoning on both samples. With identical diffusion noise, the mean/max waypoint differences were 0.188/0.598 m at 10 seconds and 0.378/1.142 m at 20 seconds across the 6.4-second prediction horizon. This limited comparison does not establish driving accuracy. FP8 improved throughput by 21–22% relative to BF16 with the same compilation settings; the earlier 138.87 ms BF16 result also lacked compiled vision.

Before trusting eGPU timings, check the connected firmware against `openpilot/system/hardware/chestnut/flash.py` and run tinygrad's existing USB integrity checks:

```bash
cd tinygrad_repo
DEV=USB+AMD:LLVM GMMU=0 AM_POWER_LIMIT=100 "$ALPAMAYO_PYTHON" -m unittest \
  test.external.external_test_usb_asm24.TestUSBIntegrity.testBoundaries \
  test.external.external_test_usb_asm24.TestUSBIntegrity.testRepeatedBatches -v
```

Do not interpret timings from failed integrity checks or nonfinite predictions as model performance. Chestnut uses the AMD kernels described above for vision, prefill, and single-token decoding, with portable fallbacks for other calls; the NVIDIA custom kernels do not accelerate the AMD eGPU.

## Initial results (2026-11-10)

For one real two-camera replay sample, greedy reasoning and ten diffusion steps:

| Runtime | Result |
| --- | --- |
| Chestnut gfx1200, 8 GB, direct PCIe | First complete inference: 158.55 s, including initial compilation; checkpoint load: 23.56 min |
| Chestnut repeat | Second warmup timed out during language prefill; no steady-state result |
| RTX 5090, NVIDIA PyTorch reference | 327.16 ms median, three measured runs after two warmups |
| RTX 5090, native tinygrad prototype | 9.51 s median, three measured runs after two warmups |

The completed Chestnut run returned 64 finite poses and the same reasoning text as CUDA. With identical diffusion noise, its mean waypoint distance from the native CUDA output was 3.7 cm (maximum 12.3 cm). That initial implementation peaked at 6.48 GB of live buffers on CUDA for this input. These checks cover one replay sample, not driving quality. The internal GPU had other existing workloads.

To reproduce the initial cold-run path, add `--usb-pcie --warmup 0 --runs 1` to the native command. The later comma4 results above supersede these initial transfer and repeat-inference failures; real-time inference remains unresolved.
