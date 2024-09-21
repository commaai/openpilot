#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="10.1"
fi

export STAGING_ROOT="/data/safe_staging"

# FIXME: revert once UDS/VIN fingerprinting in place
export FINGERPRINT="ALFA_ROMEO_STELVIO_1ST_GEN"
