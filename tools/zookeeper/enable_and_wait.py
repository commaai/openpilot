#!/usr/bin/env python

import os
import sys
import time
from tools.zookeeper import Zookeeper

z = Zookeeper()
z.set_device_power(True)

def is_online(ip):
  return (os.system(f"ping -c 1 {ip} > /dev/null") == 0)

ip = str(sys.argv[1])
while not is_online(ip):
  print(f"{ip} not online yet!")
  time.sleep(1)

