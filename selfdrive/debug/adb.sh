#!/usr/bin/bash
set -e

PORT=5555

setprop service.adb.tcp.port $PORT
if [ -f /EON ]; then
  stop adbd
  start adbd
else
  sudo systemctl start adbd
fi

IP=$(echo $SSH_CONNECTION | awk '{ print $3}')
echo "then, connect on your computer:"
echo "adb connect $IP:$PORT"
