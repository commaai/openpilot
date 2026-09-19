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
  export AGNOS_VERSION="19.7"
fi

export STAGING_ROOT="/data/safe_staging"

# Pinball shares its CAN fingerprint with COMMA_BODY.
export FINGERPRINT=COMMA_PINBALL

# Opt-in camera mode survives reboot without changing the normal onroad lifecycle.
if [ -f /data/camera720p60 ] || [ -f /data/camera720p120 ]; then
  export CAMERA_720P60="${CAMERA_720P60:-${CAMERA_720P120:-1}}"
fi
