#!/usr/bin/env python3
import os
import time
import subprocess

from panda import Panda, PandaDFU

board_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
  subprocess.check_call(f"scons -C {board_path}/.. -j$(nproc) {board_path}", shell=True)

  for s in Panda.list():
    with Panda(serial=s) as p:
      print(f"putting {p.get_usb_serial()} in DFU mode")
      p.reset(enter_bootstub=True)
      p.reset(enter_bootloader=True)

  # wait for reset pandas to come back up
  time.sleep(1)

  dfu_serials = PandaDFU.list()
  print(f"found {len(dfu_serials)} panda(s) in DFU - {dfu_serials}")
  for s in dfu_serials:
    print("flashing", s)
    PandaDFU(s).recover()
  exit(1 if len(dfu_serials) == 0 else 0)
