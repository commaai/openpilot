#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

cd ../../

export PASSIVE="0"
export NOBOARD="1"
export SIMULATION="1"

tmux new -d -s htop
tmux send-keys "./launch_openpilot.sh" ENTER
tmux neww
tmux send-keys "PYTHONPATH=$(pwd) python tools/sim/bridge.py" ENTER
tmux a

