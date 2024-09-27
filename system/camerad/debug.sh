#!/usr/bin/bash

scons -u -j8 --minimal . && sudo dmesg -C && DISABLE_ROAD=1 DISABLE_WIDE_ROAD=1 DEBUG_FRAMES=1 LOGPRINT=debug ./camerad
