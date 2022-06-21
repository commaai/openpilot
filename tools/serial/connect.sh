#!/bin/bash

while true; do
  if ls /dev/ttyUSB* 2> /dev/null; then
    sudo screen /dev/ttyUSB* 115200
  fi
  sleep 0.005
done
