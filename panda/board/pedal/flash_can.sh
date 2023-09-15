#!/usr/bin/env sh
set -e

cd ..
scons -u -j$(nproc)
cd pedal

../../tests/pedal/enter_canloader.py obj/pedal.bin.signed
