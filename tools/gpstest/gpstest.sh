#!/bin/bash

LimeGPS_BIN=LimeGPS/LimeGPS
if test -f "$LimeGPS_BIN"; then
    LD_PRELOAD=lib/libLimeSuite.so $LimeGPS_BIN $@
else
    echo "LimeGPS binary not found, run build.sh first"
fi
