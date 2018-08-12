#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import time
from collections import defaultdict
import binascii

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from panda import Panda

# fake
def sec_since_boot():
  return time.time()

def can_printer():
  p = Panda()
  p.set_safety_mode(0x1337)

  start = sec_since_boot()
  lp = sec_since_boot()
  msgs = defaultdict(list)
  canbus = int(os.getenv("CAN", 0))
  while True:
    can_recv = p.can_recv()
    for address, _, dat, src  in can_recv:
      if src == canbus:
        msgs[address].append(dat)

    if sec_since_boot() - lp > 0.1:
      dd = chr(27) + "[2J"
      dd += "%5.2f\n" % (sec_since_boot() - start)
      for k,v in sorted(zip(msgs.keys(), map(lambda x: binascii.hexlify(x[-1]), msgs.values()))):
        dd += "%s(%6d) %s\n" % ("%04X(%4d)" % (k,k),len(msgs[k]), v)
      print(dd)
      lp = sec_since_boot()

if __name__ == "__main__":
  can_printer()
