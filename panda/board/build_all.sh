#!/usr/bin/env sh
set -e

scons -u -j$(nproc)
PEDAL=1 scons -u -j$(nproc)
PEDAL=1 PEDAL_USB=1 scons -u -j$(nproc)
