#!/usr/bin/env python3
import argparse
import time
import cereal.messaging as messaging
from opendbc.car.carlog import carlog
from opendbc.car.ecu_addrs import get_all_ecu_addrs
from openpilot.common.params import Params
from openpilot.selfdrive.car.card import can_comm_callbacks, obd_callback


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Get addresses of all ECUs')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--bus', type=int, default=1)
  parser.add_argument('--no-obd', action='store_true')
  parser.add_argument('--timeout', type=float, default=1.0)
  args = parser.parse_args()

  if args.debug:
    carlog.setLevel('DEBUG')

  logcan = messaging.sub_sock('can')
  sendcan = messaging.pub_sock('sendcan')
  can_callbacks = can_comm_callbacks(logcan, sendcan)

  # Set up params for pandad
  params = Params()
  params.remove("FirmwareQueryDone")
  params.put_bool("IsOnroad", False)
  time.sleep(0.2)  # thread is 10 Hz
  params.put_bool("IsOnroad", True)

  obd_callback(params)(not args.no_obd)

  print("Getting ECU addresses ...")
  ecu_addrs = get_all_ecu_addrs(*can_callbacks, args.bus, args.timeout)

  print()
  print("Found ECUs on rx addresses:")
  for addr, subaddr, _ in ecu_addrs:
    msg = f"  {hex(addr)}"
    if subaddr is not None:
      msg += f" (sub-address: {hex(subaddr)})"
    print(msg)
