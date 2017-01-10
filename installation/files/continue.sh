#!/usr/bin/bash

# enable wifi access point for debugging only!
#service call wifi 37 i32 0 i32 1 # WifiService.setWifiApEnabled(null, true)

# check out the openpilot repo
if [ ! -d /data/openpilot ]; then
  cd /tmp
  git clone https://github.com/commaai/openpilot.git -b release
  mv /tmp/openpilot /data/openpilot
fi

# enter openpilot directory
cd /data/openpilot

# automatic update
git pull

# start manager
cd selfdrive
mkdir -p /sdcard/realdata
PYTHONPATH=/data/openpilot ./manager.py

# if broken, keep on screen error
while true; do sleep 1; done

