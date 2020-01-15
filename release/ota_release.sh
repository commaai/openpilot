#!/bin/bash
mkdir -p /tmp/panda_firmware
unzip -o $1 -d /tmp/panda_firmware

curl http://192.168.0.10/

echo "flashing user1"
curl http://192.168.0.10/espupdate1 --upload-file /tmp/panda_firmware/user1.bin
echo "flashing user2"
curl http://192.168.0.10/espupdate2 --upload-file /tmp/panda_firmware/user2.bin
echo "waiting 10s for reboot"
sleep 10
echo "flashing st"
curl http://192.168.0.10/stupdate --upload-file /tmp/panda_firmware/panda.bin
sleep 2
curl http://192.168.0.10/
echo "done"

