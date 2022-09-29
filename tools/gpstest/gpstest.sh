#!/bin/bash

LimeGPS_BIN=bin/LimeGPS
if test -f "$LimeGPS_BIN"; then
    LD_PRELOAD=lib/libLimeSuite.so $LimeGPS_BIN $@
else
    echo "Build LimeGPS first using 'scons'"
fi
