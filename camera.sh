#!/usr/bin/env bash
sudo systemctl stop power_monitor

sudo chgrp gpu /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0
sudo chmod 660 /dev/adsprpc-smd /dev/ion /dev/kgsl-3d0

pkill -9 -f 'camerad'
pkill -9 -f 'bridge'
pkill -9 -f 'encoderd'

while true; do
  cd /data/openpilot/cereal/messaging/
  ./bridge &
  BRIDGE_PID=$!

  cd /data/openpilot/system/camerad/
  ./camerad &
  CAMERAD_PID=$!

  cd /data/openpilot/system/loggerd/
  ./encoderd &
  ENCODERD_PID=$!

  pids=("$BRIDGE_PID" "$CAMERAD_PID" "$ENCODERD_PID")

  set +e
  wait -n "${pids[@]}"
  ec=$?
  set -e

  if [ "$ec" -eq 137 ]; then
    kill -9 "${pids[@]}" 2>/dev/null || true
    exit 0
  fi

  kill -9 "${pids[@]}" 2>/dev/null || true

  sleep 1
done
