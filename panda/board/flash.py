#!/usr/bin/env python3
import os
import subprocess
import argparse

from panda import Panda

board_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--all", action="store_true", help="Recover all Panda devices")
  args = parser.parse_args()

  subprocess.check_call(f"scons -C {board_path}/.. -j$(nproc) {board_path}", shell=True)

  if args.all:
    serials = Panda.list()
    print(f"found {len(serials)} panda(s) - {serials}")
  else:
    serials = [None]

  for s in serials:
    with Panda(serial=s) as p:
      print("flashing", p.get_usb_serial())
      p.flash()
  exit(1 if len(serials) == 0 else 0)
