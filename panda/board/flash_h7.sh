#!/usr/bin/env sh
set -e

PANDA_GEN3=1 scons -u
PYTHONPATH=.. python3 -c "from python import Panda; Panda().flash('obj/panda_h7.bin.signed')"
