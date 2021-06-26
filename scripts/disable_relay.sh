#!/usr/bin/bash

if [ $1 -eq 1 ]; then
  printf %s "1" > /data/params/d/dp_disable_relay
fi
if [ $1 -eq 0 ]; then
  printf %s "0" > /data/params/d/dp_disable_relay
fi

rm -rf /data/openpilot/selfdrive/boardd/boardd && reboot