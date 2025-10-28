#!/bin/bash

rocm-smi --setprofile compute
rocm-smi --setmclk 3
rocm-smi --setperflevel high

# power cap to 350W
# echo "350000000" | sudo tee /sys/class/drm/card{1..6}/device/hwmon/hwmon*/power1_cap
