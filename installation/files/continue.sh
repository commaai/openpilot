#!/usr/bin/bash

# enable wifi access point for debugging only!
#service call wifi 37 i32 0 i32 1 # WifiService.setWifiApEnabled(null, true)

# use the openpilot ro key
export GIT_SSH_COMMAND="ssh -i /data/data/com.termux/files/id_rsa_openpilot_ro"

# check out the openpilot repo
if [ ! -d /data/openpilot ]; then
  cd /tmp
  git clone git@github.com:commaai/openpilot.git -b release
  mv /tmp/openpilot /data/openpilot
fi

# enter openpilot directory
cd /data/openpilot

# removed automatic update from openpilot
#git pull

# start manager
cd selfdrive
mkdir -p /sdcard/realdata
PYTHONPATH=/data/openpilot ./manager.py

# if broken, keep on screen error
while true; do sleep 1; done

