#!/usr/bin/env python3

import os
import time
from common.params import Params
params = Params()
import cereal
import cereal.messaging as messaging


def main(gctx=None):

  shutdown_count = 0
  auto_shutdown_at = get_shutdown_val()
  frame = 0
  last_shutdown_val = auto_shutdown_at
  thermal_sock = messaging.sub_sock('thermal')
  started = False

  while 1:
    if frame % 5 == 0:
      msg = messaging.recv_sock(thermal_sock, wait=True)
      started = msg.thermal.started
      with open("/sys/class/power_supply/usb/present") as f:
        usb_online = bool(int(f.read()))
      auto_shutdown_at = get_shutdown_val()
      if not last_shutdown_val == auto_shutdown_at:
        shutdown_count = 0
        last_shutdown_val = auto_shutdown_at

    if not started and not usb_online:
      shutdown_count += 1
    else:
      shutdown_count = 0

    if auto_shutdown_at is None:
      auto_shutdown_at = get_shutdown_val()
    else:
      if shutdown_count >= auto_shutdown_at > 0:
        os.system('LD_LIBRARY_PATH="" svc power shutdown')

    time.sleep(1)

def get_shutdown_val():

  return int(1500)


if __name__ == "__main__":
  main()