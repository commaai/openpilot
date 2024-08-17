#!/usr/bin/env python3
import time
import argparse
import cereal.messaging as messaging
from cereal import car
from opendbc.car.fw_versions import get_fw_versions, match_fw_to_car
from opendbc.car.vin import get_vin
from openpilot.common.params import Params
from openpilot.selfdrive.car.card import can_comm_callbacks, obd_callback
from typing import Any

Ecu = car.CarParams.Ecu

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Get firmware version of ECUs')
  parser.add_argument('--scan', action='store_true')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--brand', help='Only query addresses/with requests for this brand')
  args = parser.parse_args()

  logcan = messaging.sub_sock('can')
  pandaStates_sock = messaging.sub_sock('pandaStates')
  sendcan = messaging.pub_sock('sendcan')
  can_callbacks = can_comm_callbacks(logcan, sendcan)

  # Set up params for pandad
  params = Params()
  params.remove("FirmwareQueryDone")
  params.put_bool("IsOnroad", False)
  time.sleep(0.2)  # thread is 10 Hz
  params.put_bool("IsOnroad", True)
  set_obd_multiplexing = obd_callback(params)

  extra: Any = None
  if args.scan:
    extra = {}
    # Honda
    for i in range(256):
      extra[(Ecu.unknown, 0x18da00f1 + (i << 8), None)] = []
      extra[(Ecu.unknown, 0x700 + i, None)] = []
      extra[(Ecu.unknown, 0x750, i)] = []
    extra = {"any": {"debug": extra}}

  num_pandas = len(messaging.recv_one_retry(pandaStates_sock).pandaStates)

  t = time.time()
  print("Getting vin...")
  set_obd_multiplexing(True)
  vin_rx_addr, vin_rx_bus, vin = get_vin(*can_callbacks, (0, 1), debug=args.debug)
  print(f'RX: {hex(vin_rx_addr)}, BUS: {vin_rx_bus}, VIN: {vin}')
  print(f"Getting VIN took {time.time() - t:.3f} s")
  print()

  t = time.time()
  fw_vers = get_fw_versions(*can_callbacks, set_obd_multiplexing, query_brand=args.brand, extra=extra, num_pandas=num_pandas, debug=args.debug, progress=True)
  _, candidates = match_fw_to_car(fw_vers, vin)

  print()
  print("Found FW versions")
  print("{")
  padding = max([len(fw.brand) for fw in fw_vers] or [0])
  for version in fw_vers:
    subaddr = None if version.subAddress == 0 else hex(version.subAddress)
    print(f"  Brand: {version.brand:{padding}}, bus: {version.bus}, OBD: {version.obdMultiplexing} - " +
          f"(Ecu.{version.ecu}, {hex(version.address)}, {subaddr}): [{version.fwVersion!r}]")
  print("}")

  print()
  print("Possible matches:", candidates)
  print(f"Getting fw took {time.time() - t:.3f} s")
