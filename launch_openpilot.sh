#!/usr/bin/env bash

export XDG_CACHE_HOME=/data/tinycache
mkdir -p $XDG_CACHE_HOME
export USBGPU=1
exec ./launch_chffrplus.sh
