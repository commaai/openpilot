#!/usr/bin/env python3
import argparse
import time
import cereal.messaging as messaging
from opendbc.car.vin import get_vin
from openpilot.selfdrive.car.card import can_comm_callbacks

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Get VIN of the car')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--bus', type=int, default=1)
  parser.add_argument('--timeout', type=float, default=0.1)
  parser.add_argument('--retry', type=int, default=5)
  args = parser.parse_args()

  sendcan = messaging.pub_sock('sendcan')
  logcan = messaging.sub_sock('can')
  can_callbacks = can_comm_callbacks(logcan, sendcan)
  time.sleep(1)

  vin_rx_addr, vin_rx_bus, vin = get_vin(*can_callbacks, (args.bus,), args.timeout, args.retry, debug=args.debug)
  print(f'RX: {hex(vin_rx_addr)}, BUS: {vin_rx_bus}, VIN: {vin}')
