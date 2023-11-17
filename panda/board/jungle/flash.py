#!/usr/bin/env python3
import os
import subprocess

from panda import PandaJungle

board_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
  subprocess.check_call(f"scons -C {board_path}/.. -u -j$(nproc) {board_path}", shell=True)

  serials = PandaJungle.list()
  print(f"found {len(serials)} panda jungle(s) - {serials}")
  for s in serials:
    print("flashing", s)
    with PandaJungle(serial=s) as p:
      p.flash()
