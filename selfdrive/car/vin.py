#!/usr/bin/env python3
import re
import struct
import traceback

import cereal.messaging as messaging
import panda.python.uds as uds
from panda.python.uds import FUNCTIONAL_ADDRS
from selfdrive.car.isotp_parallel_query import IsoTpParallelQuery
from system.swaglog import cloudlog

OBD_VIN_REQUEST = b'\x09\x02'
OBD_VIN_RESPONSE = b'\x49\x02\x01'

UDS_VIN_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + struct.pack("!H", uds.DATA_IDENTIFIER_TYPE.VIN)
UDS_VIN_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + struct.pack("!H", uds.DATA_IDENTIFIER_TYPE.VIN)

VIN_UNKNOWN = "0" * 17
VIN_RE = "[A-HJ-NPR-Z0-9]{17}"


def is_valid_vin(vin: str):
  return re.fullmatch(VIN_RE, vin) is not None


def get_vin(logcan, sendcan, bus, timeout=0.1, retry=5, debug=False):
  for i in range(retry):
    for request, response in ((UDS_VIN_REQUEST, UDS_VIN_RESPONSE), (OBD_VIN_REQUEST, OBD_VIN_RESPONSE)):
      try:
        query = IsoTpParallelQuery(sendcan, logcan, bus, FUNCTIONAL_ADDRS, [request, ], [response, ], functional_addr=True, debug=debug)
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
