#!/usr/bin/env bash
set -e

TARGET=openpilot/selfdrive/modeld/models/big_driving_tinygrad.pkl.chunkmanifest
rm -f "$TARGET"
SCONSFLAGS="-j2 --cache-disable" ./openpilot/system/manager/build.py
test -s "$TARGET"
