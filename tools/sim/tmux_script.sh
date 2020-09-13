#!/bin/bash
tmux new -d -s htop
tmux send-keys "./launch_openpilot.sh" ENTER
tmux neww
tmux send-keys "./bridge.py" ENTER
tmux a