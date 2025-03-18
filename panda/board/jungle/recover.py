#!/usr/bin/env python3
import os
import time
import subprocess
import argparse

from panda import PandaJungle, PandaJungleDFU

board_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--all", action="store_true", help="Recover all panda jungle devices")
  args = parser.parse_args()

  subprocess.check_call(f"scons -C {board_path}/.. -u -j$(nproc) {board_path}", shell=True)

  serials = PandaJungle.list() if args.all else [None]
  for s in serials:
    with PandaJungle(serial=s) as p:
      print(f"putting {p.get_usb_serial()} in DFU mode")
      p.reset(enter_bootstub=True)
      p.reset(enter_bootloader=True)

  # wait for reset panda jungles to come back up
  time.sleep(1)

  dfu_serials = PandaJungleDFU.list()
  print(f"found {len(dfu_serials)} panda jungle(s) in DFU - {dfu_serials}")
  for s in dfu_serials:
    print("flashing", s)
    PandaJungleDFU(s).recover()
  exit(1 if len(dfu_serials) == 0 else 0)
