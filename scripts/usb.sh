#!/usr/bin/env bash

# testing the GPU box

export XDG_CACHE_HOME=/data/tinycache
mkdir -p $XDG_CACHE_HOME

cd /data/openpilot/tinygrad_repo/examples
while true; do
  AMD=1 AMD_IFACE=usb python ./beautiful_cartpole.py
  sleep 1
done
