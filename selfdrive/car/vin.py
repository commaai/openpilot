#!/usr/bin/env python3
import re
import traceback

import cereal.messaging as messaging
from selfdrive.car.isotp_parallel_query import IsoTpParallelQuery
from selfdrive.car.fw_query_definitions import StdQueries
from system.swaglog import cloudlog

VIN_UNKNOWN = "0" * 17
VIN_RE = "[A-HJ-NPR-Z0-9]{17}"


def is_valid_vin(vin: str):
  return re.fullmatch(VIN_RE, vin) is not None


def get_vin(logcan, sendcan, bus, timeout=0.1, retry=5, debug=False):
  addrs = [0x7e0, 0x7e2, 0x18da10f1, 0x18da0ef1]  # engine, VMCU, 29-bit engine, PGM-FI
  for i in range(retry):
    for request, response in ((StdQueries.UDS_VIN_REQUEST, StdQueries.UDS_VIN_RESPONSE), (StdQueries.OBD_VIN_REQUEST, StdQueries.OBD_VIN_RESPONSE)):
      try:
        query = IsoTpParallelQuery(sendcan, logcan, bus, addrs, [request, ], [response, ], debug=debug)
        for (addr, rx_addr), vin in query.get_data(timeout).items():

          # Honda Bosch response starts with a length, trim to correct length
          if vin.startswith(b'\x11'):
            vin = vin[1:18]

          return addr[0], rx_addr, vin.decode()
        print(f"vin query retry ({i+1}) ...")
      except Exception:
        cloudlog.warning(f"VIN query exception: {traceback.format_exc()}")

  return 0, 0, VIN_UNKNOWN


if __name__ == "__main__":
  import time
  sendcan = messaging.pub_sock('sendcan')
  logcan = messaging.sub_sock('can')
  time.sleep(1)
  addr, vin_rx_addr, vin = get_vin(logcan, sendcan, 1, debug=False)
  print(f'TX: {hex(addr)}, RX: {hex(vin_rx_addr)}, VIN: {vin}')
