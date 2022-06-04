#!/bin/bash

export PASSIVE="0"
export NOBOARD="1"
export SIMULATION="1"
export FINGERPRINT="HONDA CIVIC 2016"

if [[ "$CI" ]]
then
  export QT_QPA_PLATFORM=offscreen
fi

export BLOCK="camerad,loggerd,encoderd"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd ../../selfdrive/manager && exec ./manager.py
