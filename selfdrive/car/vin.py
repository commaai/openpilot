#!/usr/bin/env python3
from collections import defaultdict
import re

import cereal.messaging as messaging
from panda.python.uds import get_rx_addr_for_tx_addr, FUNCTIONAL_ADDRS
from openpilot.selfdrive.car.interfaces import get_interface_attr
from openpilot.selfdrive.car.isotp_parallel_query import IsoTpParallelQuery
from openpilot.selfdrive.car.fw_query_definitions import STANDARD_VIN_ADDRS, FwQueryConfig, VinRequest, StdQueries
from openpilot.common.swaglog import cloudlog

FW_QUERY_CONFIGS: dict[str, FwQueryConfig] = get_interface_attr('FW_QUERY_CONFIG', ignore_none=True)

VIN_UNKNOWN = "0" * 17
VIN_RE = "[A-HJ-NPR-Z0-9]{17}"

STD_VIN_QUERIES = [
  VinRequest(request=StdQueries.UDS_VIN_REQUEST,
             response=StdQueries.UDS_VIN_RESPONSE,
             addrs=STANDARD_VIN_ADDRS,
             functional=True),
  VinRequest(request=StdQueries.OBD_VIN_REQUEST,
             response=StdQueries.OBD_VIN_RESPONSE,
             addrs=STANDARD_VIN_ADDRS,
             functional=True),
]


def is_valid_vin(vin: str):
  return re.fullmatch(VIN_RE, vin) is not None


def get_vin(logcan, sendcan, buses, timeout=0.1, retry=3, debug=False):
  # build queries
  queries = defaultdict(set)
  # queries = [
  #   (StdQueries.UDS_VIN_REQUEST, StdQueries.UDS_VIN_RESPONSE, (0, 1), STANDARD_VIN_ADDRS, FUNCTIONAL_ADDRS, 0x8),
  #   (StdQueries.OBD_VIN_REQUEST, StdQueries.OBD_VIN_RESPONSE, (0, 1), STANDARD_VIN_ADDRS, FUNCTIONAL_ADDRS, 0x8),
  # ]
  for vin_request in [config.vin_request for config in FW_QUERY_CONFIGS.values() if
                      config.vin_request is not None] + STD_VIN_QUERIES:
    for bus in vin_request.buses:
      queries[(bus, tuple(vin_request.addrs))].add((vin_request.request, vin_request.response,
                        None, vin_request.rx_offset))

  # print(queries)
  # raise Exception

  for bus, bus_queries in queries.items():
    print('Queries for bus={}:'.format(bus))
    for query in bus_queries:
      print(bus, query[0], query[1])
    print()

  for i in range(retry):
    for (bus, vin_addrs), bus_queries in queries.items():
      for request, response, functional_addrs, rx_offset in bus_queries:
        try:
          query = IsoTpParallelQuery(sendcan, logcan, bus, tx_addrs, [request, ], [response, ], response_offset=rx_offset,
                                     functional_addrs=functional_addrs, debug=debug)
          results = query.get_data(timeout)

          for addr in vin_addrs:
            vin = results.get((addr, None))
            if vin is not None:
              # Ford and Nissan pads with null bytes
              if len(vin) in (19, 24):
                vin = re.sub(b'\x00*$', b'', vin)

              # Honda Bosch response starts with a length, trim to correct length
              if vin.startswith(b'\x11'):
                vin = vin[1:18]

              cloudlog.error(f"got vin with {request=}")
              return get_rx_addr_for_tx_addr(addr, rx_offset=rx_offset), bus, vin.decode()
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
