#!/usr/bin/env sh
set -e

DFU_UTIL="dfu-util"

scons -u

PYTHONPATH=.. python3 -c "from python import Panda; Panda().reset(enter_bootstub=True); Panda().reset(enter_bootloader=True)" || true
sleep 1
$DFU_UTIL -d 0483:df11 -a 0 -s 0x08004000 -D obj/panda.bin.signed
$DFU_UTIL -d 0483:df11 -a 0 -s 0x08000000:leave -D obj/bootstub.panda.bin
