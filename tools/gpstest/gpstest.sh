#!/bin/bash

LimeGPS_BIN=LimeGPS/LimeGPS
if test -f "$LimeGPS_BIN"; then
  LD_PRELOAD=LimeSuite/builddir/src/libLimeSuite.so $LimeGPS_BIN $@
else
  echo "LimeGPS binary not found, run 'setup.sh' first"
fi
