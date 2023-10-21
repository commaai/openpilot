#!/bin/bash
tmux new -d -s carla-sim
tmux send-keys "./launch_openpilot.sh" ENTER
tmux neww
tmux send-keys "./run_bridge.py $*" ENTER
tmux a -t carla-sim
