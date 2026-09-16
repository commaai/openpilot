#!/usr/bin/env bash
set -e

sudo python3 openpilot/system/hardware/chestnut/flash.py

TARGET=openpilot/selfdrive/modeld/models/big_driving_tinygrad.pkl.chunkmanifest
rm -f "$TARGET"
SCONSFLAGS="-j4" ./openpilot/system/manager/build.py
test -s "$TARGET"
