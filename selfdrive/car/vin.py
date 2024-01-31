#!/usr/bin/env python3
import re

import cereal.messaging as messaging
from panda.python.uds import get_rx_addr_for_tx_addr, FUNCTIONAL_ADDRS
from openpilot.selfdrive.car.isotp_parallel_query import IsoTpParallelQuery
from openpilot.selfdrive.car.fw_query_definitions import STANDARD_VIN_ADDRS, StdQueries
from openpilot.common.swaglog import cloudlog

VIN_UNKNOWN = "0" * 17
VIN_RE = "[A-HJ-NPR-Z0-9]{17}"


def is_valid_vin(vin: str):
  return re.fullmatch(VIN_RE, vin) is not None


def get_vin(logcan, sendcan, buses, timeout=0.1, retry=3, debug=False):
  for i in range(retry):
    for bus in buses:
      for request, response, valid_buses, vin_addrs, functional_addrs, rx_offset in (
        (StdQueries.UDS_VIN_REQUEST, StdQueries.UDS_VIN_RESPONSE, (0, 1), STANDARD_VIN_ADDRS, FUNCTIONAL_ADDRS, 0x8),
        (StdQueries.OBD_VIN_REQUEST, StdQueries.OBD_VIN_RESPONSE, (0, 1), STANDARD_VIN_ADDRS, FUNCTIONAL_ADDRS, 0x8),
        (StdQueries.GM_VIN_REQUEST, StdQueries.GM_VIN_RESPONSE, (0,), [0x24b], None, 0x400),  # Bolt fwdCamera
      ):
        if bus not in valid_buses:
          continue

        try:
          query = IsoTpParallelQuery(sendcan, logcan, bus, vin_addrs, [request, ], [response, ], response_offset=rx_offset,
                                     functional_addrs=functional_addrs, debug=debug)
          results = query.get_data(timeout)

          for addr in vin_addrs:
            vin = results.get((addr, None))
            if vin is not None:
              # Ford pads with null bytes
              if len(vin) == 24:
                vin = re.sub(b'\x00*$', b'', vin)

              # Honda Bosch response starts with a length, trim to correct length
              if vin.startswith(b'\x11'):
                vin = vin[1:18]

              cloudlog.warning(f"got vin with {request=}")
              return get_rx_addr_for_tx_addr(addr), bus, vin.decode()
        except Exception:
          cloudlog.exception("VIN query exception")

    cloudlog.error(f"vin query retry ({i+1}) ...")

  return -1, -1, VIN_UNKNOWN


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

  vin_rx_addr, vin_rx_bus, vin = get_vin(logcan, sendcan, (args.bus,), args.timeout, args.retry, debug=args.debug)
  print(f'RX: {hex(vin_rx_addr)}, BUS: {vin_rx_bus}, VIN: {vin}')
