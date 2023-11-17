#!/usr/bin/env python3

import time
from panda import PandaJungle

if __name__ == "__main__":
  jungle = PandaJungle()

  while True:
    for bus in range(3):
      print(bus, jungle.can_health(bus))
    print()
    time.sleep(1)
