#!/usr/bin/env sh
set -e

scons -u
PYTHONPATH=.. python3 -c "from python import Panda; Panda().flash('obj/panda.bin.signed')"
