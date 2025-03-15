#!/usr/bin/env python3
import os
import time

from opendbc.car.structs import CarParams
from panda import Panda


if __name__ == "__main__":
  flag_set = False
  while True:
    try:
      with Panda(disable_checks=False) as p:
        if not flag_set:
          p.set_heartbeat_disabled()
          p.set_safety_mode(CarParams.SafetyModel.elm327, 30)
          flag_set = True

        # shutdown when told
        ch = p.can_health(0)
        if ch['can_data_speed'] == 1000:
          os.system("sudo poweroff")
    except Exception as e:
      print(str(e))
    time.sleep(0.5)
