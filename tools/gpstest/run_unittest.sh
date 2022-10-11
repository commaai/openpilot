#!/bin/bash

# NOTE: can only run inside the limeGPS test box!

# run limeGPS with static location
./run_static_gps_signal.py  37.1337 137.1337 >/dev/null &
gps_PID=$?

echo "starting limeGPS..."
sleep 2

# run unit tests depending on module available
python -m unittest test_gps.py
python -m unittest test_gps_qcom.py

kill $gps_PID
