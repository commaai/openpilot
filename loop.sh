#!/usr/bin/env bash

pkill -f watch3
pkill -f compressed_vipc

BIG=1 /home/batman/openpilot/selfdrive/ui/watch3.py &

while true; do
  echo "Waiting for device..."
  adb wait-for-device

  adb shell 'su - comma -c "source /etc/profile && tmux kill-server"'

  sleep 1

  ip=$(adb shell "ifconfig wlan0" | awk '/inet / {print $2}' | sed 's/addr://')
  /home/batman/openpilot/tools/camerastream/compressed_vipc.py "$ip" &

  adb push camera.sh /data
  adb shell 'su - comma -c "source /etc/profile && sudo chown comma: /data/camera.sh && chmod +x /data/camera.sh"'

  adb shell 'su - comma -c "source /etc/profile && /data/camera.sh"'
  pkill -f compressed_vipc
  echo -e "\n\nDevice disconnected..."
  sleep 1
done
