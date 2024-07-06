#!/usr/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="10.1"
fi

export FINGERPRINT="GWM_HAVAL_H6_PHEV_3RD_GEN"
# export FINGERPRINT="TOYOTA_COROLLA_TSS2" # TODO clean-after-port
# export SKIP_FW_QUERY=1
export STAGING_ROOT="/data/safe_staging"
