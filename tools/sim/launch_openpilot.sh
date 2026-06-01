#!/usr/bin/env bash

export PASSIVE="0"
export NOBOARD="1"
export SIMULATION="1"
export SKIP_FW_QUERY="1"
export FINGERPRINT="HONDA_CIVIC_2022"

export BLOCK="${BLOCK},camerad,micd,logmessaged,manage_athenad"
if [[ "$CI" ]]; then
  pulseaudio --check 2>/dev/null || pulseaudio --start --daemonize --exit-idle-time=-1 2>/dev/null || true
  pactl load-module module-null-sink sink_name=dummy 2>/dev/null || true
  export PULSE_SINK=dummy
fi

python3 -c "from openpilot.selfdrive.test.helpers import set_params_enabled; set_params_enabled()"

SCRIPT_DIR=$(dirname "$0")
OPENPILOT_DIR=$SCRIPT_DIR/../../

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $OPENPILOT_DIR/system/manager && exec ./manager.py
