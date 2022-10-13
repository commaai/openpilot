#!/bin/bash

# NOTE: can only run inside limeGPS test box!

# run limeGPS with random static location
timeout 300 ./simulate_gps_signal.py &
gps_PID=$?

echo "starting limeGPS..."
sleep 10

# run unit tests (skipped when module not present)
python -m unittest test_gps.py
python -m unittest test_gps_qcom.py

kill $gps_PID
