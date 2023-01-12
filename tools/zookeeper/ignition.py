#!/usr/bin/env python3

import sys
from openpilot.tools.zookeeper import Zookeeper


if __name__ == "__main__":
  z = Zookeeper()
  z.set_device_ignition(1 if int(sys.argv[1]) > 0 else 0)

