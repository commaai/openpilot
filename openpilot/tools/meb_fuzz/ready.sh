#!/bin/sh
# Chime on the comma screen: the maneuver alert plays the prompt sound when line 1 contains "Active".
S=/private/tmp/claude-501/-Users-shane-openpilot/277d3f73-02c6-42db-9783-48a8a7db1d8f/scratchpad
python3 $S/fuzz/ctl.py 172.20.10.3 alert "READY (Active): $1" "$2" > /dev/null
sleep 2.5
python3 $S/fuzz/ctl.py 172.20.10.3 alert "" "" > /dev/null
