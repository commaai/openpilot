#!/bin/bash

# NOTE: can only run inside limeGPS test box!

# run limeGPS with random static location
timeout 300 ./simulate_gps_signal.py 32.7518 -117.1962 &
gps_PID=$(ps -aux | grep -m 1 "timeout 300" | awk '{print $2}')

echo "starting limeGPS..."
sleep 10

# run unit tests (skipped when module not present)
python -m unittest test_gps.py
python -m unittest test_gps_qcom.py

kill $gps_PID
