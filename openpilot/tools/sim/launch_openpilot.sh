#!/usr/bin/env bash

export PASSIVE="0"
export NOBOARD="1"
export SIMULATION="1"
export SKIP_FW_QUERY="1"
export FINGERPRINT="HONDA_CIVIC_2022"

export BLOCK="${BLOCK},camerad,micd,logmessaged,manage_athenad"
if [[ -z "$SIM_LOGS" ]]; then
  # set SIM_LOGS=1 to record qlog/rlog and camera files during the sim run
  export BLOCK="${BLOCK},loggerd,encoderd"
fi
if [[ "$CI" ]]; then
  # TODO: offscreen UI should work
  # soundd needs an audio device, which CI runners don't have
  # journald needs systemd, which CI runners don't have
  export BLOCK="${BLOCK},ui,soundd,journald"
fi

python3 -c "from openpilot.selfdrive.test.helpers import set_params_enabled; set_params_enabled()"

SCRIPT_DIR=$(dirname "$0")
OPENPILOT_DIR=$SCRIPT_DIR/../../

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $OPENPILOT_DIR/system/manager && exec ./manager.py
