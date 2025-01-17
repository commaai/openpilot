#!/usr/bin/env bash

# Install user SecOCKey to params
cp /cache/params/SecOCKey /data/params/d/SecOCKey || true

exec ./launch_chffrplus.sh
