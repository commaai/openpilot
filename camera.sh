#!/usr/bin/env bash
pkill -9 -f 'camerad'
pkill -9 -f 'bridge'
pkill -9 -f 'encoderd'

sleep 1

cd /data/openpilot/cereal/messaging/
./bridge &

cd /data/openpilot/system/camerad/
./camerad &

cd /data/openpilot/system/loggerd/
./encoderd &
