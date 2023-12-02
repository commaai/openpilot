#!/usr/bin/env python3
import os
import sys
import time
from socket import gethostbyname, gaierror
from openpilot.tools.zookeeper import Zookeeper

def is_online(ip):
  try:
    addr = gethostbyname(ip)
    return (os.system(f"ping -c 1 {addr} > /dev/null") == 0)
  except gaierror:
    return False

if __name__ == "__main__":
  z = Zookeeper()
  z.set_device_power(True)


  ip = str(sys.argv[1])
  timeout = int(sys.argv[2])
  start_time = time.time()
  while not is_online(ip):
    print(f"{ip} not online yet!")

    if time.time() - start_time > timeout:
      print("Timed out!")
      raise TimeoutError()

    time.sleep(1)

