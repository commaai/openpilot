#!/usr/bin/env sh
set -e

scons -u -j$(nproc)
# Recovers panda from DFU mode only, use flash.sh after power cycling panda
printf %b 'from python import Panda\nfor serial in Panda.list(): Panda(serial).reset(enter_bootstub=True); Panda(serial).reset(enter_bootloader=True)' | PYTHONPATH=.. python3
sleep 1
printf %b 'from python import PandaDFU\nfor serial in PandaDFU.list(): PandaDFU(serial).recover()' | PYTHONPATH=.. python3
