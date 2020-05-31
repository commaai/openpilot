#!/usr/bin/env python3

import os
import sys
import time
import select

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from panda import Panda

setcolor = ["\033[1;32;40m", "\033[1;31;40m"]
unsetcolor = "\033[00m"

if __name__ == "__main__":
  while True:
    try:
      port_number = int(os.getenv("PORT", 0))
      claim = os.getenv("CLAIM") is not None

      serials = Panda.list()
      if os.getenv("SERIAL"):
        serials = [x for x in serials if x==os.getenv("SERIAL")]

      pandas = list([Panda(x, claim=claim) for x in serials])

      if not len(pandas):
        sys.exit("no pandas found")

      if os.getenv("BAUD") is not None:
        for panda in pandas:
          panda.set_uart_baud(port_number, int(os.getenv("BAUD"))) # type: ignore

      while True:
        for i, panda in enumerate(pandas):
          while True:
            ret = panda.serial_read(port_number)
            if len(ret) > 0:
              sys.stdout.write(setcolor[i] + ret.decode('ascii') + unsetcolor)
              sys.stdout.flush()
            else:
              break
          if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            ln = sys.stdin.readline()
            if claim:
              panda.serial_write(port_number, ln)
          time.sleep(0.01)
    except Exception:
      print("panda disconnected!")
      time.sleep(0.5);
