#!/usr/bin/bash
set -e

PORT=5555

setprop service.adb.tcp.port $PORT
sudo systemctl start adbd

IP=$(echo $SSH_CONNECTION | awk '{ print $3}')
echo "then, connect on your computer:"
echo "adb connect $IP:$PORT"
