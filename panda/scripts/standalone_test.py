#!/usr/bin/env python3
import struct
import time

from opendbc.car.structs import CarParams
from panda import Panda

if __name__ == "__main__":
  p = Panda()
  print(p.get_serial())
  print(p.health())

  t1 = time.time()
  for _ in range(100):
    p.get_serial()
  t2 = time.time()
  print("100 requests took %.2f ms" % ((t2 - t1) * 1000))

  p.set_safety_mode(CarParams.SafetyModel.allOutput)

  a = 0
  while True:
    # flood
    msg = b"\xaa" * 4 + struct.pack("I", a)
    p.can_send(0xaa, msg, 0)
    p.can_send(0xaa, msg, 1)
    p.can_send(0xaa, msg, 4)
    time.sleep(0.01)

    dat = p.can_recv()
    if len(dat) > 0:
      print(dat)
    a += 1
