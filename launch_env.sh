#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# models get lower priority than ui
# - ui is ~5ms
# - modeld is 20ms
# - DM is 10ms
# in order to run ui at 60fps (16.67ms), we need to allow
# it to preempt the model workloads. we have enough
# headroom for this until ui is moved to the CPU.
export QCOM_PRIORITY=12

if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="17.2"
fi

export STAGING_ROOT="/data/safe_staging"

if [ -f /ASIUS ]; then
  export BIG=1
  export DEV=CL
  export NOLOCALS=1
  export FLOAT16=1
  export RUSTICL_ENABLE=freedreno
  export DISABLE_FAST_IDIV=1
  export EMULATED_DTYPES=long
  export SUM_DTYPE=float16
  export JIT_BATCH_SIZE=0
  export BEAM=2
  export IGNORE_JIT_FIRST_BEAM=1
fi
