#!/usr/bin/env bash

while true; do
  if ls /dev/serial/by-id/usb-FTDI_FT230X* 2> /dev/null; then
    sudo screen /dev/serial/by-id/usb-FTDI_FT230X* 115200
  fi
  sleep 0.005
done
