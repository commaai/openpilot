#!/usr/bin/bash

if [ $1 -eq 1 ]; then
  printf %s "1" > /data/params/d/dp_vw
fi
if [ $1 -eq 0 ]; then
  printf %s "0" > /data/params/d/dp_vw
fi
rm /data/openpilot/panda/board/obj/panda.bin