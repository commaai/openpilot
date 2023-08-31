#!/usr/bin/env python3
import os
import time
import subprocess

from panda import PandaJungle, PandaJungleDFU

board_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
  subprocess.check_call(f"scons -C {board_path}/.. -u -j$(nproc) {board_path}", shell=True)

  for s in PandaJungle.list():
    print("putting", s, "in DFU mode")
    with PandaJungle(serial=s) as p:
      p.reset(enter_bootstub=True)
      p.reset(enter_bootloader=True)

  # wait for reset panda jungles to come back up
  time.sleep(1)

  dfu_serials = PandaJungleDFU.list()
  print(f"found {len(dfu_serials)} panda jungle(s) in DFU - {dfu_serials}")
  for s in dfu_serials:
    print("flashing", s)
    PandaJungleDFU(s).recover()
