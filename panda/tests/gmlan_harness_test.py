#!/usr/bin/env python3

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from panda import Panda  # noqa: E402

WHITE_GMLAN_BUS = 3
OTHER_GMLAN_BUS = 1

def set_gmlan(p):
  if p.is_white():
    p.set_gmlan(2)
  else:
    p.set_obd(True)

def set_speed_kbps(p, speed):
  if p.is_white():
    p.set_can_speed_kbps(WHITE_GMLAN_BUS, speed)
  else:
    p.set_can_speed_kbps(OTHER_GMLAN_BUS, speed)

def send(p, id_, msg):
  if p.is_white():
    p.can_send(id_, msg, WHITE_GMLAN_BUS)
  else:
    p.can_send(id_, msg, OTHER_GMLAN_BUS)

if __name__ == "__main__":
  pl = Panda.list()
  assert(len(pl) == 2)
  p0 = Panda(pl[1])
  p1 = Panda(pl[0])

  p0.set_safety_mode(Panda.SAFETY_ALLOUTPUT)
  p1.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  print("0: ", p0.get_type())
  print("1: ", p1.get_type())

  set_gmlan(p0)
  set_gmlan(p1)

  p0.can_clear(0xFFFF)
  p1.can_clear(0xFFFF)

  try:
    loops = 0
    while True:
      for speed in [33.3, 83.3]:
        set_speed_kbps(p0, speed)
        set_speed_kbps(p1, speed)
        p0.can_clear(0xFFFF)
        p1.can_clear(0xFFFF)

        print(f"Speed: {speed}")
        time.sleep(0.1)

        print("Send 1 -> 0")
        send(p1, 1, b"1to0:" + bytes(str(loops%100), "utf-8"))
        time.sleep(0.05)
        rx = list(filter(lambda x: x[3] < 128, p0.can_recv()))
        print(rx)
        assert(len(rx) == 1)

        print("Send 0 -> 1")
        send(p0, 1, b"0to1:" + bytes(str(loops%100), "utf-8"))
        time.sleep(0.05)
        rx = list(filter(lambda x: x[3] < 128, p1.can_recv()))
        print(rx)
        assert(len(rx) == 1)

        time.sleep(0.5)


      loops += 1
      print(f"Completed {loops} loops")
  except Exception:
    print("Test failed somehow. Did you power the black panda using the GMLAN harness?")
