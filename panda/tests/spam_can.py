#!/usr/bin/env python3
import os
import random

from panda import Panda

def get_test_string():
  return b"test" + os.urandom(10)

if __name__ == "__main__":
  p = Panda()
  p.set_safety_mode(Panda.SAFETY_ALLOUTPUT)

  print("Spamming all buses...")
  while True:
    at = random.randint(1, 2000)
    st = get_test_string()[0:8]
    bus = random.randint(0, 2)
    p.can_send(at, st, bus)
    # print("Sent message on bus: ", bus)
