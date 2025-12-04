#!/usr/bin/env bash

pkill -f watch3
pkill -f compressed_vipc

BIG=1 /home/batman/openpilot/selfdrive/ui/watch3.py &

while true; do
  echo "Waiting for device..."
  adb wait-for-device

  adb shell 'su - comma -c "source /etc/profile && tmux kill-server"'

  PORTS="51336 57332 42305"
  for p in $PORTS; do
    adb forward tcp:$p tcp:$p
  done

  /home/batman/openpilot/tools/camerastream/compressed_vipc.py 127.0.0.1 &

  adb push camera.sh /data
  adb shell 'su - comma -c "source /etc/profile && sudo chown comma: /data/camera.sh && chmod +x /data/camera.sh"'

  adb push continue.sh /data
  adb shell 'su - comma -c "source /etc/profile && sudo chown comma: /data/continue.sh && chmod +x /data/continue.sh"'

  adb shell 'su - comma -c "source /etc/profile && /data/camera.sh"'
  touch /tmp/new_cam
  pkill -f compressed_vipc
  echo -e "\n\nDevice disconnected..."
  sleep 1
done
