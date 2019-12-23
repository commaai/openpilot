#!/usr/bin/env python2.7

import os
import time
from common.params import Params
params = Params()

def main(gctx=None):

  shutdown_count = 0
  auto_shutdown_at = get_shutdown_val()
  frame = 0
  last_shutdown_val = get_shutdown_val()

  while 1:
    with open("/sys/class/power_supply/usb/present") as f:
      usb_online = bool(int(f.read()))

    if not usb_online:
      # we update the value every 5 seconds in case of user updates it
      if frame % 5 == 0:
        auto_shutdown_at = get_shutdown_val()
      shutdown_count += 1
    else:
      shutdown_count = 0

    if not last_shutdown_val == auto_shutdown_at:
      shutdown_count = 0
      last_shutdown_val = auto_shutdown_at

    if auto_shutdown_at is None:
      auto_shutdown_at = get_shutdown_val()
    else:
      if shutdown_count >= auto_shutdown_at > 0:
        os.system('LD_LIBRARY_PATH="" svc power shutdown')

    time.sleep(1)

def get_shutdown_val():
  val = params.get("DragonAutoShutdownAt", encoding='utf8')
  if val is None:
    return None
  else:
    return int(val)*60 # convert to seconds


if __name__ == "__main__":
  main()