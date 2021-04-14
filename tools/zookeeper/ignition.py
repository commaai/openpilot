#!/usr/bin/env python

import sys
from tools.zookeeper import Zookeeper

z = Zookeeper()
z.set_device_ignition(1 if int(sys.argv[1]) > 0 else 0)

