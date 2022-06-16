#!/bin/bash

export PASSIVE="0"
export NOBOARD="1"
export SIMULATION="1"
export FINGERPRINT="HONDA CIVIC 2016"

export BLOCK="camerad,loggerd,encoderd"
if [[ "$CI" ]]; then
  # TODO: offscreen UI should work
  export BLOCK="${BLOCK},ui"
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd ../../selfdrive/manager && exec ./manager.py
