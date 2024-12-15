#!/usr/bin/env bash
set -e

#echo 4294967295 | sudo tee /sys/module/cam_debug_util/parameters/debug_mdl

# no CCI and UTIL, very spammy
echo 0xfffdbfff | sudo tee /sys/module/cam_debug_util/parameters/debug_mdl
echo 0 | sudo tee /sys/module/cam_debug_util/parameters/debug_mdl

sudo dmesg -C
scons -u -j8 --minimal .
export DEBUG_FRAMES=1
export DISABLE_ROAD=1 DISABLE_WIDE_ROAD=1
#export DISABLE_DRIVER=1
export LOGPRINT=debug
./camerad
