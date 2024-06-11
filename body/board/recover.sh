#!/usr/bin/env sh
set -e

FLASH_UTIL="openocd"

scons -u

$FLASH_UTIL -f interface/stlink.cfg -c "set CPUTAPID 0" -f target/stm32f4x.cfg -c init -c "reset halt" -c "flash write_image erase obj/body.bin.signed 0x08004000" -c "flash write_image erase obj/bootstub.body.bin 0x08000000" -c "reset run" -c "shutdown"
