#!/usr/bin/env python3
from panda.tests.libs.resetter import Resetter


# * port 1: unused jungles-under-test
# * port 2: USB hubs
# * port 3: HITL pandas and their jungles
if __name__ == "__main__":
  with Resetter() as r:
    r.enable_power(1, False)
    r.cycle_power(ports=[2, 3])
