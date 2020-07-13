#!/usr/bin/env python3
import struct
import traceback
from typing import Any

from tqdm import tqdm

import panda.python.uds as uds
from cereal import car
from selfdrive.car.fingerprints import get_attr_from_cars
from selfdrive.car.isotp_parallel_query import IsoTpParallelQuery
from selfdrive.swaglog import cloudlog

Ecu = car.CarParams.Ecu


def p16(val):
  return struct.pack("!H", val)


TESTER_PRESENT_REQUEST = bytes([uds.SERVICE_TYPE.TESTER_PRESENT, 0x0])
TESTER_PRESENT_RESPONSE = bytes([uds.SERVICE_TYPE.TESTER_PRESENT + 0x40, 0x0])

SHORT_TESTER_PRESENT_REQUEST = bytes([uds.SERVICE_TYPE.TESTER_PRESENT])
SHORT_TESTER_PRESENT_RESPONSE = bytes([uds.SERVICE_TYPE.TESTER_PRESENT + 0x40])

DEFAULT_DIAGNOSTIC_REQUEST = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL,
                                    uds.SESSION_TYPE.DEFAULT])
DEFAULT_DIAGNOSTIC_RESPONSE = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL + 0x40,
                                    uds.SESSION_TYPE.DEFAULT, 0x0, 0x32, 0x1, 0xf4])

EXTENDED_DIAGNOSTIC_REQUEST = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL,
                                     uds.SESSION_TYPE.EXTENDED_DIAGNOSTIC])
EXTENDED_DIAGNOSTIC_RESPONSE = bytes([uds.SERVICE_TYPE.DIAGNOSTIC_SESSION_CONTROL + 0x40,
                                      uds.SESSION_TYPE.EXTENDED_DIAGNOSTIC, 0x0, 0x32, 0x1, 0xf4])

UDS_VERSION_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.APPLICATION_SOFTWARE_IDENTIFICATION)
UDS_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.APPLICATION_SOFTWARE_IDENTIFICATION)


SUBARU_ECU_REQUEST = b'\x22\xf1\x97'
SUBARU_ECU_RESPONSE = b'\x62\xf1\x97'


# supports subaddressing, request, response
REQUESTS = [
  # Subaru
  (
    "subaru",
    [TESTER_PRESENT_REQUEST, SUBARU_ECU_REQUEST],
    [TESTER_PRESENT_RESPONSE, SUBARU_ECU_RESPONSE],
  ),
]


def chunks(l, n=128):
  for i in range(0, len(l), n):
    yield l[i:i + n]


def get_ecu_list(logcan, sendcan, bus, extra=None, timeout=0.1, debug=False, progress=False):
  ecu_types = {}

  # Extract ECU addresses to query from fingerprints
  # ECUs using a subadress need be queried one by one, the rest can be done in parallel
  addrs = []
  parallel_addrs = []

  versions = get_attr_from_cars('FW_VERSIONS', combine_brands=False)
  if extra is not None:
    versions.update(extra)

  for brand, brand_versions in versions.items():
    for c in brand_versions.values():
      for ecu_type, addr, sub_addr in c.keys():
        a = (brand, addr, sub_addr)
        if a not in ecu_types:
          ecu_types[(addr, sub_addr)] = ecu_type

        if sub_addr is None:
          if a not in parallel_addrs:
            parallel_addrs.append(a)
        else:
          if [a] not in addrs:
            addrs.append([a])

  addrs.insert(0, parallel_addrs)

  ecu_list = {}
  for i, addr in enumerate(tqdm(addrs, disable=not progress)):
    for addr_chunk in chunks(addr):
      for brand, request, response in REQUESTS:
        try:
          addrs = [(a, s) for (b, a, s) in addr_chunk if b in (brand, 'any')]

          if addrs:
            query = IsoTpParallelQuery(sendcan, logcan, bus, addrs, request, response, debug=debug)
            t = 2 * timeout if i == 0 else timeout
            ecu_list.update(query.get_data(t))
        except Exception:
          cloudlog.warning(f"ECU query exception: {traceback.format_exc()}")

  # Build capnp list to put into CarParams
  car_fw = []
  for addr, version in ecu_list.items():
    f = car.CarParams.CarFw.new_message()

    f.ecu = ecu_types[addr]
    f.fwVersion = version
    f.address = addr[0]

    if addr[1] is not None:
      f.subAddress = addr[1]

    car_fw.append(f)

  return car_fw


if __name__ == "__main__":
  import time
  import argparse
  import cereal.messaging as messaging
  from selfdrive.car.vin import get_vin

  parser = argparse.ArgumentParser(description='Get firmware version of ECUs')
  parser.add_argument('--scan', action='store_true')
  parser.add_argument('--debug', action='store_true')
  args = parser.parse_args()

  logcan = messaging.sub_sock('can')
  sendcan = messaging.pub_sock('sendcan')

  extra: Any = None
  if args.scan:
    extra = {}
    # Honda
    for i in range(256):
      extra[(Ecu.unknown, 0x18da00f1 + (i << 8), None)] = []
      extra[(Ecu.unknown, 0x700 + i, None)] = []
      extra[(Ecu.unknown, 0x750, i)] = []
    extra = {"any": {"debug": extra}}

  time.sleep(10.)

  t = time.time()
  print("Getting vin...")
  addr, vin = get_vin(logcan, sendcan, 1, retry=10, debug=args.debug)
  print(f"VIN: {vin}")
  print("Getting VIN took %.3f s" % (time.time() - t))
  print()

  t = time.time()
  ecu_list = get_ecu_list(logcan, sendcan, 1, extra=extra, debug=args.debug, progress=True)

  print()
  print("Found ECU descriptions")
  print("{")
  for desc in ecu_list:
    subaddr = None if desc.subAddress == 0 else hex(desc.subAddress)
    print(f"  (Ecu.{desc.ecu}, {hex(desc.address)}, {subaddr}): [{desc.fwVersion}]")
  print("}")

  print()
  print("Getting fw took %.3f s" % (time.time() - t))
