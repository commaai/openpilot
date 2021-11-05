#!/usr/bin/env python3
import struct
import traceback
from typing import Any
from collections import defaultdict

from tqdm import tqdm

import panda.python.uds as uds
from cereal import car
from selfdrive.car.fingerprints import FW_VERSIONS, get_attr_from_cars
from selfdrive.car.isotp_parallel_query import IsoTpParallelQuery
from selfdrive.car.toyota.values import CAR as TOYOTA
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


HYUNDAI_VERSION_REQUEST_LONG = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(0xf100)  # Long description
HYUNDAI_VERSION_REQUEST_MULTI = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_SPARE_PART_NUMBER) + \
  p16(uds.DATA_IDENTIFIER_TYPE.APPLICATION_SOFTWARE_IDENTIFICATION) + \
  p16(0xf100) 
HYUNDAI_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40])


TOYOTA_VERSION_REQUEST = b'\x1a\x88\x01'
TOYOTA_VERSION_RESPONSE = b'\x5a\x88\x01'

VOLKSWAGEN_VERSION_REQUEST_MULTI = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_SPARE_PART_NUMBER) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_ECU_SOFTWARE_VERSION_NUMBER) + \
  p16(uds.DATA_IDENTIFIER_TYPE.APPLICATION_DATA_IDENTIFICATION)
VOLKSWAGEN_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40])

OBD_VERSION_REQUEST = b'\x09\x04'
OBD_VERSION_RESPONSE = b'\x49\x04'

DEFAULT_RX_OFFSET = 0x8
VOLKSWAGEN_RX_OFFSET = 0x6a

MAZDA_VERSION_REQUEST = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_ECU_SOFTWARE_NUMBER)
MAZDA_VERSION_RESPONSE =  bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_ECU_SOFTWARE_NUMBER)

NISSAN_VERSION_REQUEST_KWP = b'\x21\x83'
NISSAN_VERSION_RESPONSE_KWP = b'\x61\x83'

NISSAN_VERSION_REQUEST_STANDARD = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_ECU_SOFTWARE_NUMBER)
NISSAN_VERSION_RESPONSE_STANDARD =  bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_ECU_SOFTWARE_NUMBER)

NISSAN_RX_OFFSET = 0x20

# brand, request, response, response offset
REQUESTS = [
  # Hyundai
  (
    "hyundai",
    [HYUNDAI_VERSION_REQUEST_LONG],
    [HYUNDAI_VERSION_RESPONSE],
    DEFAULT_RX_OFFSET,
  ),
  (
    "hyundai",
    [HYUNDAI_VERSION_REQUEST_MULTI],
    [HYUNDAI_VERSION_RESPONSE],
    DEFAULT_RX_OFFSET,
  ),
  # Honda
  (
    "honda",
    [UDS_VERSION_REQUEST],
    [UDS_VERSION_RESPONSE],
    DEFAULT_RX_OFFSET,
  ),
  # Toyota
  (
    "toyota",
    [SHORT_TESTER_PRESENT_REQUEST, TOYOTA_VERSION_REQUEST],
    [SHORT_TESTER_PRESENT_RESPONSE, TOYOTA_VERSION_RESPONSE],
    DEFAULT_RX_OFFSET,
  ),
  (
    "toyota",
    [SHORT_TESTER_PRESENT_REQUEST, OBD_VERSION_REQUEST],
    [SHORT_TESTER_PRESENT_RESPONSE, OBD_VERSION_RESPONSE],
    DEFAULT_RX_OFFSET,
  ),
  (
    "toyota",
    [TESTER_PRESENT_REQUEST, DEFAULT_DIAGNOSTIC_REQUEST, EXTENDED_DIAGNOSTIC_REQUEST, UDS_VERSION_REQUEST],
    [TESTER_PRESENT_RESPONSE, DEFAULT_DIAGNOSTIC_RESPONSE, EXTENDED_DIAGNOSTIC_RESPONSE, UDS_VERSION_RESPONSE],
    DEFAULT_RX_OFFSET,
  ),
  # Volkswagen
  (
    "volkswagen",
    [VOLKSWAGEN_VERSION_REQUEST_MULTI],
    [VOLKSWAGEN_VERSION_RESPONSE],
    VOLKSWAGEN_RX_OFFSET,
  ),
  (
    "volkswagen",
    [VOLKSWAGEN_VERSION_REQUEST_MULTI],
    [VOLKSWAGEN_VERSION_RESPONSE],
    DEFAULT_RX_OFFSET,
  ),
  # Mazda
  (
    "mazda",
    [MAZDA_VERSION_REQUEST],
    [MAZDA_VERSION_RESPONSE],
    DEFAULT_RX_OFFSET,
  ),
  # Nissan
  (
    "nissan",
    [NISSAN_VERSION_REQUEST_KWP],
    [NISSAN_VERSION_RESPONSE_KWP],
    DEFAULT_RX_OFFSET,
  ),
  (
    "nissan",
    [NISSAN_VERSION_REQUEST_KWP],
    [NISSAN_VERSION_RESPONSE_KWP],
    NISSAN_RX_OFFSET,
  ),
  (
    "nissan",
    [NISSAN_VERSION_REQUEST_STANDARD],
    [NISSAN_VERSION_RESPONSE_STANDARD],
    NISSAN_RX_OFFSET,
  ),
]


def chunks(l, n=128):
  for i in range(0, len(l), n):
    yield l[i:i + n]


def build_fw_dict(fw_versions):
  fw_versions_dict = {}
  for fw in fw_versions:
    addr = fw.address
    sub_addr = fw.subAddress if fw.subAddress != 0 else None
    fw_versions_dict[(addr, sub_addr)] = fw.fwVersion
  return fw_versions_dict


def match_fw_to_car_fuzzy(fw_versions_dict, log=True, exclude=None):
  """Do a fuzzy FW match. This function will return a match, and the number of firmware version
  that were matched uniquely to that specific car. If multiple ECUs uniquely match to different cars
  the match is rejected."""

  # These ECUs are known to be shared between models (EPS only between hybrid/ICE version)
  # Getting this exactly right isn't crucial, but excluding camera and radar makes it almost
  # impossible to get 3 matching versions, even if two models with shared parts are released at the same
  # time and only one is in our database.
  exclude_types = [Ecu.fwdCamera, Ecu.fwdRadar, Ecu.eps]

  # Build lookup table from (addr, subaddr, fw) to list of candidate cars
  all_fw_versions = defaultdict(list)
  for candidate, fw_by_addr in FW_VERSIONS.items():
    if candidate == exclude:
      continue

    for addr, fws in fw_by_addr.items():
      if addr[0] in exclude_types:
        continue
      for f in fws:
        all_fw_versions[(addr[1], addr[2], f)].append(candidate)

  match_count = 0
  candidate = None
  for addr, version in fw_versions_dict.items():
    # All cars that have this FW response on the specified address
    candidates = all_fw_versions[(addr[0], addr[1], version)]

    if len(candidates) == 1:
      match_count += 1
      if candidate is None:
        candidate = candidates[0]
      # We uniquely matched two different cars. No fuzzy match possible
      elif candidate != candidates[0]:
        return set()

  if match_count >= 2:
    if log:
      cloudlog.error(f"Fingerprinted {candidate} using fuzzy match. {match_count} matching ECUs")
    return set([candidate])
  else:
    return set()


def match_fw_to_car_exact(fw_versions_dict):
  """Do an exact FW match. Returns all cars that match the given
  FW versions for a list of "essential" ECUs. If an ECU is not considered
  essential the FW version can be missing to get a fingerprint, but if it's present it
  needs to match the database."""
  invalid = []
  candidates = FW_VERSIONS

  for candidate, fws in candidates.items():
    for ecu, expected_versions in fws.items():
      ecu_type = ecu[0]
      addr = ecu[1:]
      found_version = fw_versions_dict.get(addr, None)
      ESSENTIAL_ECUS = [Ecu.engine, Ecu.eps, Ecu.esp, Ecu.fwdRadar, Ecu.fwdCamera, Ecu.vsa]
      if ecu_type == Ecu.esp and candidate in [TOYOTA.RAV4, TOYOTA.COROLLA, TOYOTA.HIGHLANDER] and found_version is None:
        continue

      # On some Toyota models, the engine can show on two different addresses
      if ecu_type == Ecu.engine and candidate in [TOYOTA.CAMRY, TOYOTA.COROLLA_TSS2, TOYOTA.CHR, TOYOTA.LEXUS_IS] and found_version is None:
        continue

      # Ignore non essential ecus
      if ecu_type not in ESSENTIAL_ECUS and found_version is None:
        continue

      if found_version not in expected_versions:
        invalid.append(candidate)
        break

  return set(candidates.keys()) - set(invalid)


def match_fw_to_car(fw_versions, allow_fuzzy=True):
  fw_versions_dict = build_fw_dict(fw_versions)
  matches = match_fw_to_car_exact(fw_versions_dict)

  exact_match = True
  if allow_fuzzy and len(matches) == 0:
    matches = match_fw_to_car_fuzzy(fw_versions_dict)

    # Fuzzy match found
    if len(matches) == 1:
      exact_match = False

  return exact_match, matches


def get_fw_versions(logcan, sendcan, bus, extra=None, timeout=0.1, debug=False, progress=False):
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

  fw_versions = {}
  for i, addr in enumerate(tqdm(addrs, disable=not progress)):
    for addr_chunk in chunks(addr):
      for brand, request, response, response_offset in REQUESTS:
        try:
          addrs = [(a, s) for (b, a, s) in addr_chunk if b in (brand, 'any')]

          if addrs:
            query = IsoTpParallelQuery(sendcan, logcan, bus, addrs, request, response, response_offset, debug=debug)
            t = 2 * timeout if i == 0 else timeout
            fw_versions.update(query.get_data(t))
        except Exception:
          cloudlog.warning(f"FW query exception: {traceback.format_exc()}")

  # Build capnp list to put into CarParams
  car_fw = []
  for addr, version in fw_versions.items():
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

  time.sleep(1.)

  t = time.time()
  print("Getting vin...")
  addr, vin = get_vin(logcan, sendcan, 1, retry=10, debug=args.debug)
  print(f"VIN: {vin}")
  print("Getting VIN took %.3f s" % (time.time() - t))
  print()

  t = time.time()
  fw_vers = get_fw_versions(logcan, sendcan, 1, extra=extra, debug=args.debug, progress=True)
  _, candidates = match_fw_to_car(fw_vers)

  print()
  print("Found FW versions")
  print("{")
  for version in fw_vers:
    subaddr = None if version.subAddress == 0 else hex(version.subAddress)
    print(f"  (Ecu.{version.ecu}, {hex(version.address)}, {subaddr}): [{version.fwVersion}]")
  print("}")

  print()
  print("Possible matches:", candidates)
  print("Getting fw took %.3f s" % (time.time() - t))
