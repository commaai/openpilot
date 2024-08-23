#!/usr/bin/env python3
import time
import cereal.messaging as messaging
from opendbc.car.disable_ecu import disable_ecu
from openpilot.selfdrive.car.card import can_comm_callbacks

if __name__ == "__main__":
  sendcan = messaging.pub_sock('sendcan')
  logcan = messaging.sub_sock('can')
  can_callbacks = can_comm_callbacks(logcan, sendcan)
  time.sleep(1)

  # honda bosch radar disable
  disabled = disable_ecu(*can_callbacks, bus=1, addr=0x18DAB0F1, com_cont_req=b'\x28\x83\x03', timeout=0.5, debug=False)
  print(f"disabled: {disabled}")
