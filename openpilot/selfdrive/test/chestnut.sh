#!/usr/bin/env bash
set -e

sudo python3 openpilot/system/hardware/chestnut/flash.py

SCONSFLAGS="-j4" ./openpilot/system/manager/build.py
