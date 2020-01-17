#!/usr/bin/env python3

import sys
import time
from selfdrive.thermald import setup_eon_fan, set_eon_fan

if __name__ == "__main__":
  val = 0
  setup_eon_fan()

  if len(sys.argv) > 1:
    set_eon_fan(int(sys.argv[1]))
    exit(0)

  while True:
    sys.stderr.write("setting fan to %d\n" % val)
    set_eon_fan(val)
    time.sleep(2)
    val += 1
    val %= 4


