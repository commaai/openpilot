#!/usr/bin/env bash

# Kill any old instances
pkill -9 -f 'camerad'  || true
pkill -9 -f 'bridge'   || true
pkill -9 -f 'encoderd' || true

# Start bridge once
(
  cd /data/openpilot/cereal/messaging/ || exit 1
  ./bridge
) &

# Restart loop for camerad
(
  while true; do
    cd /data/openpilot/system/camerad/ || exit 1
    echo "[camerad] starting..."
    ./camerad
    echo "[camerad] exited with code $? – restarting in 1s"
    sleep 1
  done
) &

# Restart loop for encoderd
(
  while true; do
    cd /data/openpilot/system/loggerd/ || exit 1
    echo "[encoderd] starting..."
    ./encoderd
    echo "[encoderd] exited with code $? – restarting in 1s"
    sleep 1
  done
) &
