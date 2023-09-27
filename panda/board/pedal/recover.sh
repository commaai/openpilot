#!/usr/bin/env sh
set -e

DFU_UTIL="dfu-util"

cd ..
PEDAL=1 scons -u -j$(nproc)
cd pedal

$DFU_UTIL -d 0483:df11 -a 0 -s 0x08004000 -D ../obj/pedal.bin.signed
$DFU_UTIL -d 0483:df11 -a 0 -s 0x08000000:leave -D ../obj/bootstub.pedal.bin
