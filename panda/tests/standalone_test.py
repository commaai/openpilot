#!/usr/bin/env python3
import os
import sys
import struct
import time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from panda import Panda

if __name__ == "__main__":
  if os.getenv("WIFI") is not None:
    p = Panda("WIFI")
  else:
    p = Panda()
  print(p.get_serial())
  print(p.health())

  t1 = time.time()
  for i in range(100):
    p.get_serial()
  t2 = time.time()
  print("100 requests took %.2f ms" % ((t2-t1)*1000))

  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  a = 0
  while True:
    # flood
    msg = b"\xaa"*4 + struct.pack("I", a)
    p.can_send(0xaa, msg, 0)
    p.can_send(0xaa, msg, 1)
    p.can_send(0xaa, msg, 4)
    time.sleep(0.01)

    dat = p.can_recv()
    if len(dat) > 0:
      print(dat)
    a += 1
