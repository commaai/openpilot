#!/usr/bin/env python3
import random
import time

from .boardd_old import can_health
from .boardd_old import can_init
from .boardd_old import can_recv
from .boardd_old import can_send_many

if __name__ == "__main__":
  can_init()
  while 1:
    c = random.randint(0, 3)
    if c == 0:
      print(can_recv())
    elif c == 1:
      print(can_health())
    elif c == 2:
      many = [[0x123, 0, "abcdef", 0]] * random.randint(1, 10)
      can_send_many(many)
    elif c == 3:
      time.sleep(random.randint(0, 100) / 1000.0)
