#!/bin/bash

# NOTE: can only run inside the limeGPS test box!

# run limeGPS with static location
./run_static_gps_signal.py >/dev/null &
gps_PID=$?

echo "starting limeGPS..."
sleep 3

# run unit tests (skipped when module not present)
python -m unittest test_gps.py
python -m unittest test_gps_qcom.py

kill $gps_PID
