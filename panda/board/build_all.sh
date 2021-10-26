#!/usr/bin/env sh
set -e

scons -u
PEDAL=1 scons -u
PEDAL=1 PEDAL_USB=1 scons -u
