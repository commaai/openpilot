#!/usr/bin/env python3
import time
from pprint import pprint

from panda import PandaJungle

if __name__ == "__main__":
  i = 0
  pi = 0

  pj = PandaJungle()
  while True:
    st = time.monotonic()
    while time.monotonic() - st < 1:
      pj.health()
      pj.can_health(0)
      i += 1
    pprint(pj.health())
    print(f"Speed: {i - pi}Hz")
    pi = i

