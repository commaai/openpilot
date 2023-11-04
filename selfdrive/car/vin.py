#!/usr/bin/env python3
import re

import cereal.messaging as messaging
from panda.python.uds import get_rx_addr_for_tx_addr, FUNCTIONAL_ADDRS
from selfdrive.car.isotp_parallel_query import IsoTpParallelQuery
from selfdrive.car.fw_query_definitions import StdQueries
from system.swaglog import cloudlog

VIN_UNKNOWN = "0" * 17
VIN_RE = "[A-HJ-NPR-Z0-9]{17}"


def is_valid_vin(vin: str):
  return re.fullmatch(VIN_RE, vin) is not None


def get_vin(logcan, sendcan, bus, timeout=0.1, retry=5, debug=False):
  addrs = list(range(0x7e0, 0x7e8)) + list(range(0x18DA00F1, 0x18DB00F1, 0x100))  # addrs to process/wait for
  valid_vin_addrs = [0x7e0, 0x7e2, 0x18da10f1, 0x18da0ef1]  # engine, VMCU, 29-bit engine, PGM-FI
  for i in range(retry):
    for request, response in ((StdQueries.UDS_VIN_REQUEST, StdQueries.UDS_VIN_RESPONSE), (StdQueries.OBD_VIN_REQUEST, StdQueries.OBD_VIN_RESPONSE)):
      try:
        query = IsoTpParallelQuery(sendcan, logcan, bus, addrs, [request, ], [response, ], functional_addrs=FUNCTIONAL_ADDRS, debug=debug)
        results = query.get_data(timeout)

        for addr in valid_vin_addrs:
          vin = results.get((addr, None))
          if vin is not None:
            # Ford pads with null bytes
            if len(vin) == 24:
              vin = re.sub(b'\x00*$', b'', vin)

            # Honda Bosch response starts with a length, trim to correct length
            if vin.startswith(b'\x11'):
              vin = vin[1:18]

            return get_rx_addr_for_tx_addr(addr), vin.decode()

        cloudlog.error(f"vin query retry ({i+1}) ...")
      except Exception:
        cloudlog.exception("VIN query exception")

  return 0, VIN_UNKNOWN


if __name__ == "__main__":
  import argparse
  import time

  parser = argparse.ArgumentParser(description='Get VIN of the car')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--bus', type=int, default=1)
  parser.add_argument('--timeout', type=float, default=0.1)
  parser.add_argument('--retry', type=int, default=5)
  args = parser.parse_args()

  sendcan = messaging.pub_sock('sendcan')
  logcan = messaging.sub_sock('can')
  time.sleep(1)

  vin_rx_addr, vin = get_vin(logcan, sendcan, args.bus, args.timeout, args.retry, debug=args.debug)
  print(f'RX: {hex(vin_rx_addr)}, VIN: {vin}')
