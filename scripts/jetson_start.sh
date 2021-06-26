#!/usr/bin/env sh

sudo echo 5000 > /sys/devices/c250000.i2c/i2c-7/7-0040/iio:device0/crit_current_limit_0
sudo nvpmodel -m 2 && sudo jetson_clocks
export PASSIVE=0
export NOSENSOR=1
export USE_MIPI=1
cd ../selfdrive/manager && ./manager.py