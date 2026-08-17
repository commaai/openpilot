# Runbook: Llama 3 8B Training on DigitalOcean MI350X

## Machine Specs
- 8x MI350X GPUs (gfx950, device ID 75b0), 288GB VRAM each
- 2TB RAM, 192 CPUs, 2TB disk
- ROCm 7.14 at `/opt/rocm` (NOT `/opt/rocm-7.1.1` like the submission scripts assume)
- Python 3.12

## Phase 1: System Setup

### 1.1 Install packages
```bash
apt-get update
apt-get install -y python3-pip python3-venv git tmux rclone clang
```

### 1.2 Install Python deps
```bash
python3 -m pip install --break-system-packages --ignore-installed typing-extensions numpy tqdm wandb tiktoken sentencepiece
```
Note: `--ignore-installed typing-extensions` is needed because the base image ships typing-extensions 4.10.0 without a RECORD file, so pip cannot uninstall it.

### 1.3 Install ROCm dev headers
The base image has ROCm runtime but NOT the HIP dev headers. Need:
```bash
apt-get install -y amdrocm-core-dev
```
This installs `hip/hip_runtime.h` at `/opt/rocm/core-7.14/include/hip/hip_runtime.h`.
The symlink `/opt/rocm/include` → `/opt/rocm/core-7.14/include` makes it available at `/opt/rocm/include/hip/hip_runtime.h`.

### 1.4 Configure ROCm comgr
ROCm 7.14 ships comgr 3.3 at `/opt/rocm/lib/libamd_comgr.so`. tinygrad's DLL loader needs explicit env vars to find it (it searches for `libcomgr.so*` by default, not `libamd_comgr.so*`). Set these in the run command:
```bash
export COMGR_PATH=/opt/rocm/lib/libamd_comgr.so
export COMGR_3_PATH=/opt/rocm/lib/libamd_comgr.so
```
Also add ROCm libs to ldconfig so comgr's shared library dependencies resolve:
```bash
cat > /etc/ld.so.conf.d/rocm.conf << 'EOF'
/opt/rocm/lib
/opt/rocm/lib/llvm/lib
/opt/rocm/lib/rocm_sysdeps/lib
EOF
ldconfig
```

### 1.5 Install geohot tmux config
```bash
curl -sL https://raw.githubusercontent.com/geohot/configuration/master/.tmux.conf -o ~/.tmux.conf
```

### 1.6 Reload amdgpu driver
tinygrad's HCQ backend needs `/dev/kfd` which is created by the amdgpu kernel driver.
If the driver was unloaded, reload it:
```bash
modprobe amdgpu
ls /dev/kfd  # should exist
```

## Phase 2: Clone tinygrad
```bash
cd /root
git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
python3 -m pip install --break-system-packages -e .
```

## Phase 3: Download C4 Dataset

The C4 data is on the MLCommons Cloudflare R2 bucket in Megatron-LM indexed format.

```bash
rclone config create mlc-training s3 provider=Cloudflare \
  access_key_id=76ea42eadb867e854061a1806220ee1e \
  secret_access_key=a53625c4d45e3ca8ac0df8a353ea3a41ffc3292aa25259addd8b7dc5a6ce2936 \
  endpoint=c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com

mkdir -p /raid/datasets/c4-8b
rclone copy mlc-training:mlcommons-training-wg-public/llama3_1/datasets/c4/llama3_1_8b/ /raid/datasets/c4-8b/ -P
```

Files downloaded (~85GB total, ~6 minutes):
- `c4-train.en_6_text_document.bin` (79 GB)
- `c4-train.en_6_text_document.idx` (870 MB)
- `c4-validation-91205-samples.en_text_document.bin` (159 MB)
- `c4-validation-91205-samples.en_text_document.idx` (1.8 MB)
- `LICENSE.txt`, `NOTICE.txt`

**Wait for rclone to fully complete before starting training.** Starting training while the dataset is still downloading will read a truncated .bin file, causing `ValueError: all input arrays must have the same shape` in the dataloader. The stale `.index_cache` and `.blend_cache` files must also be deleted if this happens:
```bash
rm -f /raid/datasets/c4-8b/*.index_cache /raid/datasets/c4-8b/*.blend_cache
```

## Phase 4: wandb Login
```bash
wandb login
```
Enter API key from https://wandb.ai/authorize

Alternatively, pass the key directly:
```bash
wandb login <API_KEY>
```

## Phase 5: Run Training

Run training in tmux so it survives SSH disconnects:
```bash
tmux new-session -d -s train 'cd /root/tinygrad && COMGR_PATH=/opt/rocm/lib/libamd_comgr.so COMGR_3_PATH=/opt/rocm/lib/libamd_comgr.so CC=/opt/rocm/core-7.14/lib/llvm/bin/clang DEV=AMD:HIP ROCM_PATH=/opt/rocm WANDB=1 bash examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama31_8b/implementations/tinybox_8xMI350X/dev_run.sh 2>&1 | tee /root/train.log'
```
Attach with `tmux attach -t train`.

### 5.1 Smoke test (beam search, 2 layers, real data)
Always run beam first to validate the pipeline:
```bash
tmux new-session -d -s beam 'cd /root/tinygrad && COMGR_PATH=/opt/rocm/lib/libamd_comgr.so COMGR_3_PATH=/opt/rocm/lib/libamd_comgr.so CC=/opt/rocm/core-7.14/lib/llvm/bin/clang DEV=AMD:HIP ROCM_PATH=/opt/rocm bash examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama31_8b/implementations/tinybox_8xMI350X/dev_beam.sh 2>&1 | tee /root/beam.log'
```

The beam test runs 10 training steps with 2 layers. Expected results:
- ~0.29s per step after warmup
- ~700K GFLOPS, ~7% MFU (low because only 2 layers)
- ~380 GB VRAM used
- Loss stable at ~12.55 with random init

### 5.2 Full training run
```bash
tmux new-session -d -s train 'cd /root/tinygrad && COMGR_PATH=/opt/rocm/lib/libamd_comgr.so COMGR_3_PATH=/opt/rocm/lib/libamd_comgr.so CC=/opt/rocm/core-7.14/lib/llvm/bin/clang DEV=AMD:HIP ROCM_PATH=/opt/rocm WANDB=1 bash examples/mlperf/training_submission_v6.0/tinycorp/benchmarks/llama31_8b/implementations/tinybox_8xMI350X/dev_run.sh 2>&1 | tee /root/train.log'
```

## Environment Variable Reference

| Variable | Value | Why |
|---|---|---|
| `COMGR_PATH` | `/opt/rocm/lib/libamd_comgr.so` | tinygrad's DLL loader needs explicit path to find comgr 3.3 |
| `COMGR_3_PATH` | `/opt/rocm/lib/libamd_comgr.so` | comgr 3.x uses a separate `comgr_3` module with its own path var |
| `CC` | `/opt/rocm/core-7.14/lib/llvm/bin/clang` | System clang doesn't know gfx950; must use ROCm's bundled clang |
| `DEV` | `AMD:HIP` | Force HIPRenderer (comgr-based) over HIPCCRenderer (hipcc subprocess) |
| `ROCM_PATH` | `/opt/rocm` | Script defaults to `/opt/rocm-7.1.1` which doesn't exist |
| `WANDB` | `1` | Enable wandb logging (off by default) |

## Architecture

| Component | Source file |
|---|---|
| Model | `examples/mlperf/models/flat_llama.py` — FlatTransformer, FP8 MXFP4 weights, fused QKV, flash attention |
| Trainer | `examples/mlperf/model_train.py` → `train_llama3()` |
| Optimizer | `examples/mlperf/optim.py` — GradAccClipAdamW, master weights, FP8 re-quant |
| LR schedule | `examples/mlperf/lr_schedulers.py` — CosineAnnealingLRWithWarmup |
| Dataloader | `examples/mlperf/dataloader.py` — Megatron-LM indexed bin format |
| ASM GEMM | `extra/gemm/cdna_asm_gemm.py` — gfx950 MFMA assembly, MXFP4 |
| Flash attention | `extra/thunder/amd/fa.py` |
| Fused kernels | `extra/llama_kernels/` — rmsnorm, silu, quantize, fused_ce |
| GPU driver | `tinygrad/runtime/ops_amd.py` — HCQ, direct KFD ioctl |
| Renderer | `tinygrad/renderer/cstyle.py` — HIPRenderer for gfx950 |
| comgr compiler | `tinygrad/runtime/support/compiler_amd.py` — HIPCompiler using comgr 3.3 |

## Troubleshooting

### `'hip/hip_runtime.h' file not found`
Install `amdrocm-core-dev`:
```bash
apt-get install -y amdrocm-core-dev
```

### `'gfx950' is not a recognized processor` + LLVM crash
System clang doesn't know gfx950. Set `CC=/opt/rocm/core-7.14/lib/llvm/bin/clang`.

### `comgr not available: try setting COMGR_PATH?`
Add ROCm libs to ldconfig and set `COMGR_PATH` and `COMGR_3_PATH`:
```bash
# /etc/ld.so.conf.d/rocm.conf should contain /opt/rocm/lib paths
ldconfig
```

### `comgr not available: try setting COMGR_3_PATH?`
comgr 3.x uses a separate module. Set `COMGR_3_PATH=/opt/rocm/lib/libamd_comgr.so` too.

### `No such file or directory: 'clang'`
Install clang: `apt-get install -y clang` (for CPU compilation).
For gfx950 HIP compilation, comgr (not clang) is used — ensure the ROCm 7.14 comgr 3.3 is properly loaded via `COMGR_PATH` and `COMGR_3_PATH`.

## Appendix: KVM Virtualization Observations

### Virtualization detection
```
$ systemd-detect-virt
kvm
$ lspci -nn | grep AMD
83:00.0 ... Device [1002:75b0]
```
CPU flags include `hypervisor`. `dmesg` shows `Hypervisor detected: KVM`.

### Working path: amdgpu driver (KFDIface)
The amdgpu driver loads on boot and binds to all 8 GPUs, creating `/dev/kfd` and 64 renderD nodes (`/dev/dri/renderD128` through `/dev/dri/renderD191`). tinygrad's `KFDIface` enumerates GPUs through `/sys/devices/virtual/kfd/kfd/topology/nodes` and uses `/dev/kfd` for ioctl. No PCI device ID patching is needed — the KFD path does not use `PCIIface` or `AMDev._run_discovery()`.

This is the working configuration. No code changes to tinygrad are required.

### PCIIface path (does not work on this VM)
For reference, the `PCIIface` path was also explored but does not work in this KVM guest:

- `PCIIface` in `ops_amd.py` does not list device ID `0x75b0`. Adding it allows PCI detection but `AMDev._run_discovery()` fails because the VRAM BAR reads all `0xFF`.
- This was observed with the GPU unbound from any driver, after PCI reset, and with VFIO bound.
- VFIO binding (`vfio-pci` with `enable_unsafe_noiommu_mode=1`) succeeded but VRAM BAR still reads all `0xFF`.
- No IOMMU in guest — `dmesg` has no `AMD-Vi` entries, PCI devices have no `iommu_group` symlink.

### amdgpu driver behavior
On first boot, amdgpu loaded and bound to all 8 GPUs. On one boot it failed to initialize:
```
[  799.780369] amdgpu 0000:83:00.0: Failed to alloc msi vectors
[  799.781476] amdgpu 0000:83:00.0: sw_init of IP block <vega20_ih> failed -22
[  799.782724] amdgpu 0000:83:00.0: amdgpu_device_ip_init failed
[  799.793885] amdgpu 0000:83:00.0: Fatal error during GPU init
```
On a subsequent boot, amdgpu initialized successfully (SMU initialized, VRAM ready). After unbinding all 8 GPUs from amdgpu, `rmmod amdgpu` wedged the module (stuck in "Unloading" state in `/proc/modules`), requiring a full VM reboot.

### No fan control
No `fan*` or `pwm*` hwmon entries exist. Only `temp*`, `power*`, `freq*` are exposed. GPU temps read 56-63°C, power ~265W per GPU.
