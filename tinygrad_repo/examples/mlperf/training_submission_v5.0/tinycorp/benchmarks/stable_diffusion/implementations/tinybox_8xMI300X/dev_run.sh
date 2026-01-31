#!/usr/bin/env bash

DATETIME=${2:-$(date "+%m%d%H%M")}
LOGFILE="${HOME}/logs/sd_mi300x_${DATETIME}.log"
# UNET_CKPTDIR must be set: training saves checkpoints to this path, then a separate eval process scans this path to know which checkpoints to eval
export UNET_CKPTDIR="${HOME}/stable_diffusion/training_checkpoints/${DATETIME}"
mkdir -p "${HOME}/logs" "$UNET_CKPTDIR"

# run this script in isolation when using the --bg flag
if [[ "${1:-}" == "--bg" ]]; then
  echo "logging output to $LOGFILE"
  echo "saving UNet checkpoints to $UNET_CKPTDIR"
  script_path="$(readlink -f "${BASH_SOURCE[0]}")"
  nohup bash "$script_path" run "$DATETIME" >"$LOGFILE" 2>&1 & disown $!
  exit 0
fi

# venv management
if [[ -d .venv-sd-mlperf ]]; then
  . .venv-sd-mlperf/bin/activate
else
  python3 -m venv .venv-sd-mlperf && . .venv-sd-mlperf/bin/activate
  pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install tqdm numpy ftfy regex pillow scipy wandb webdataset
fi
pip list
apt list --installed | grep amdgpu
rocm-smi --version
modinfo amdgpu | grep version

export BEAM=2 BEAM_UOPS_MAX=8000 BEAM_UPCAST_MAX=256 BEAM_LOCAL_MAX=1024 BEAM_MIN_PROGRESS=5 IGNORE_JIT_FIRST_BEAM=1 HCQDEV_WAIT_TIMEOUT_MS=300000
export AMD_LLVM=0 # bf16 seems to require this
export DATADIR="/raid/datasets/stable_diffusion"
export CKPTDIR="/raid/weights/stable_diffusion"
export EVAL_CKPT_DIR=$UNET_CKPTDIR
export MODEL="stable_diffusion" PYTHONPATH="."
export GPUS=8 BS=304
export CONTEXT_BS=816 DENOISE_BS=600 DECODE_BS=384 INCEPTION_BS=560 CLIP_BS=240
export WANDB=1
export PARALLEL=4
export PYTHONUNBUFFERED=1
sudo rocm-smi -d 0 1 2 3 4 5 6 7 --setperfdeterminism 1500 || exit 1

# Retry BEAM search if script fails before BEAM COMPLETE is printed, but don't retry after that
run_retry(){ local try=0 max=5 code tmp py pgid kids
  while :; do
    tmp=$(mktemp)
    setsid bash -c 'exec env "$@"' _ "$@" > >(tee -a "$LOGFILE" | tee "$tmp") 2>&1 &
    py=$!; pgid=$(ps -o pgid= -p "$py" | tr -d ' ')
    wait "$py"; code=$?
    [[ -n "$pgid" ]] && { kill -TERM -"$pgid" 2>/dev/null; sleep 1; kill -KILL -"$pgid" 2>/dev/null; }
    kids=$(pgrep -P "$py" || true)
    while [[ -n "$kids" ]]; do
      kill -TERM $kids 2>/dev/null; sleep 0.5
      kids=$(for k in $kids; do pgrep -P "$k" || true; done)
    done
    grep -q 'BEAM COMPLETE' "$tmp" && { rm -f "$tmp"; return 1; }
    rm -f "$tmp"
    ((code==0)) && return 0
    ((try>=max)) && return 2
    ((try++)); sleep 90; echo "try = ${try}"
  done
}

# Power limiting to 400W is only needed if GPUs fall out of sync (causing 2.2x increased train time) at higher power, which has been observed at 450W
sudo rocm-smi -d 0 1 2 3 4 5 6 7 --setpoweroverdrive 750 && \
run_retry TOTAL_CKPTS=7 python3 examples/mlperf/model_train.py; (( $? == 2 )) && { echo "training failed before BEAM completion"; exit 2; }
sleep 90

run_retry EVAL_SAMPLES=600 python3 examples/mlperf/model_eval.py; (( $? == 2 )) && { echo "eval failed before BEAM completion"; exit 2; }
# Checkpoints will be evaluated in reverse chronological order, even if above training crashed early
# STOP_IF_CONVERGED=1: Stop the eval after the first time convergence is detected; no more checkpoints will be evaluated after that.
STOP_IF_CONVERGED=1 python3 examples/mlperf/model_eval.py
