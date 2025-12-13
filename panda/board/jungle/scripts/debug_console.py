#!/usr/bin/env python3
import os
import sys
import time

from panda import PandaJungle

setcolor = ["\033[1;32;40m", "\033[1;31;40m"]
unsetcolor = "\033[00m"

if __name__ == "__main__":
  while True:
    try:
      claim = os.getenv("CLAIM") is not None

      serials = PandaJungle.list()
      if os.getenv("SERIAL"):
        serials = [x for x in serials if x==os.getenv("SERIAL")]

      panda_jungles = [PandaJungle(x, claim=claim) for x in serials]

      if not len(panda_jungles):
        sys.exit("no panda jungles found")

      while True:
        for i, panda_jungle in enumerate(panda_jungles):
          while True:
            ret = panda_jungle.debug_read()
            if len(ret) > 0:
              sys.stdout.write(setcolor[i] + ret.decode('ascii') + unsetcolor)
              sys.stdout.flush()
            else:
              break
          time.sleep(0.01)
    except Exception as e:
      print(e)
      print("panda jungle disconnected!")
      time.sleep(0.5)
