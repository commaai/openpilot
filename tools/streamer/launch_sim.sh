#!/bin/bash
echo "Starting openpilot in simulation mode"
export DELAY="0"
export PASSIVE="0"
export PYOPENCL_CTX="0"
export NOBOARD="1"
export NOSENSOR="1"
export SIMULATION="1"
export METADRIVE="1"
export SKIP_FW_QUERY="1"
export FINGERPRINT="SIMULATOR"
export BLOCK="camerad,loggerd,encoderd,micd"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd selfdrive/manager && exec ./manager.py
