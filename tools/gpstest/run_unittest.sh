#!/bin/bash

# NOTE: can only run inside the limeGPS test box!

# run limeGPS with static location
./run_static_gps_signal.py  37.1337 137.1337 >/dev/null &
gps_PID=$?

echo "starting limeGPS..."
sleep 2


# run unit tests depending on module available

./check_gps.py
has_ublox=$?
if [ $has_ublox ]
then
  echo "running ublox tests"
  python -m unittest test_gps.py
else
  echo "running quectel tests"
  python -m unittest test_gps_qcom.py
fi

kill $gps_PID
