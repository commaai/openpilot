#!/usr/bin/env bash
set -e

TARGET=openpilot/selfdrive/modeld/models/big_driving_tinygrad.pkl.chunkmanifest
rm -f "$TARGET"
scons --cache-disable "$TARGET"
test -s "$TARGET"
