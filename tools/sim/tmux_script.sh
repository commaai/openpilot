#!/bin/bash
tmux new -d -s carla-sim
tmux send-keys "./launch_openpilot.sh" ENTER
tmux neww
tmux send-keys "./bridge.py --high_quality --dual_camera $*" ENTER
tmux a -t carla-sim
