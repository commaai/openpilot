#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import time
import select

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from panda import Panda

setcolor = ["\033[1;32;40m", "\033[1;31;40m"]
unsetcolor = "\033[00m"

if __name__ == "__main__":
  port_number = int(os.getenv("PORT", 0))
  claim = os.getenv("CLAIM") is not None

  serials = Panda.list()
  if os.getenv("SERIAL"):
    serials = filter(lambda x: x==os.getenv("SERIAL"), serials)

  pandas = list(map(lambda x: Panda(x, claim=claim), serials))

  if os.getenv("BAUD") is not None:
    for panda in pandas:
      panda.set_uart_baud(port_number, int(os.getenv("BAUD")))

  while True:
    for i, panda in enumerate(pandas):
      while True:
        ret = panda.serial_read(port_number)
        if len(ret) > 0:
          sys.stdout.write(setcolor[i] + str(ret) + unsetcolor)
          sys.stdout.flush()
        else:
          break
      if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        ln = sys.stdin.readline()
        if claim:
          panda.serial_write(port_number, ln)
      time.sleep(0.01)
