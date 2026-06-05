#!/usr/bin/env python
import sys

from panda import PandaJungle

if __name__ == "__main__":
  jungle = PandaJungle()

  # If first argument specified, it sets the ignition (0 or 1)
  if len(sys.argv) > 1:
    if sys.argv[1] == '1':
      jungle.set_ignition(True)
    else:
      jungle.set_ignition(False)
  jungle.set_harness_orientation(1)
  jungle.set_panda_power(True)


