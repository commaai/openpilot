#!/usr/bin/env python3
import traceback
from collections import defaultdict
from typing import Optional, Set, Tuple

import panda.python.uds as uds
from cereal import car
from selfdrive.car.ecu_addrs import get_ecu_addrs
from selfdrive.car.interfaces import get_interface_attr
from selfdrive.car.fingerprints import FW_VERSIONS
from selfdrive.car.fw_query_definitions import FwQueryConfig
from selfdrive.car.isotp_parallel_query import IsoTpParallelQuery
from system.swaglog import cloudlog

Ecu = car.CarParams.Ecu
ESSENTIAL_ECUS = [Ecu.engine, Ecu.eps, Ecu.abs, Ecu.fwdRadar, Ecu.fwdCamera, Ecu.vsa]

FW_QUERY_CONFIGS = get_interface_attr('FW_QUERY_CONFIG', ignore_none=True)
VERSIONS = get_interface_attr('FW_VERSIONS', ignore_none=True)

MODEL_TO_BRAND = {c: b for b, e in VERSIONS.items() for c in e}
REQUESTS = [(brand, r) for brand, config in FW_QUERY_CONFIGS.items() for r in config.requests]


def chunks(l, n=128):
  for i in range(0, len(l), n):
    yield l[i:i + n]


def build_fw_dict(fw_versions, filter_brand=None):
  fw_versions_dict = defaultdict(set)
  for fw in fw_versions:
    if filter_brand is None or fw.brand == filter_brand:
      fw_versions_dict[fw.ecu.raw].add(fw.fwVersion)
  return dict(fw_versions_dict)


def match_fw_to_car_fuzzy(fw_versions_dict, log=True, exclude=None):
  """Do a fuzzy FW match. This function will return a match, and the number of firmware version
  that were matched uniquely to that specific car. If multiple ECUs uniquely match to different cars
  the match is rejected."""

  # These ECUs are known to be shared between models (EPS only between hybrid/ICE version)
  # Getting this exactly right isn't crucial, but excluding camera and radar makes it almost
  # impossible to get 3 matching versions, even if two models with shared parts are released at the same
  # time and only one is in our database.
  exclude_types = [Ecu.fwdCamera, Ecu.fwdRadar, Ecu.eps, Ecu.debug]

  # Build lookup table from (ecu, fw) to list of candidate cars
  all_fw_versions = defaultdict(list)
  for candidate, fw_by_ecu in FW_VERSIONS.items():
    if candidate == exclude:
      continue

    for ecu_type, fws in fw_by_ecu.items():
      if ecu_type in exclude_types:
        continue
      for f in fws:
        all_fw_versions[(ecu_type, f)].append(candidate)

  match_count = 0
  candidate = None
  for ecu_type, versions in fw_versions_dict.items():
    for version in versions:
      # All cars that have this FW response on the specified address
      candidates = all_fw_versions[(ecu_type, version)]

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
    return {candidate}
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
    for ecu_type, expected_versions in fws.items():
      config = FW_QUERY_CONFIGS[MODEL_TO_BRAND[candidate]]
      found_versions = fw_versions_dict.get(ecu_type, set())
      if not len(found_versions):
        # Some models can sometimes miss an ecu, or show on two different addresses
        if candidate in config.non_essential_ecus.get(ecu_type, []):
          continue

        # Ignore non essential ecus
        if ecu_type not in ESSENTIAL_ECUS:
          continue

      # Virtual debug ecu doesn't need to match the database
      if ecu_type == Ecu.debug:
        continue

      if not any([found_version in expected_versions for found_version in found_versions]):
        invalid.append(candidate)
        break

  return set(candidates.keys()) - set(invalid)


def match_fw_to_car(fw_versions, allow_exact=True, allow_fuzzy=True):
  # Try exact matching first
  exact_matches = []
  if allow_exact:
    exact_matches = [(True, match_fw_to_car_exact)]
  if allow_fuzzy:
    exact_matches.append((False, match_fw_to_car_fuzzy))

  for exact_match, match_func in exact_matches:
    # For each brand, attempt to fingerprint using all FW returned from its queries
    matches = set()
    for brand in VERSIONS.keys():
      fw_versions_dict = build_fw_dict(fw_versions, filter_brand=brand)
      matches |= match_func(fw_versions_dict)

    if len(matches):
      return exact_match, matches

  return True, set()


def get_present_ecus(logcan, sendcan):
  queries = list()
  parallel_queries = list()
  responses = set()

  # TODO: common function for generating addrs with proper parallel/serial addrs
  for config in FW_QUERY_CONFIGS.values():
    for r in config.requests:
      for (addr, sub_addr), ecu_type in config.ecus.items():
        # Only query ecus in whitelist if whitelist is not empty
        if len(r.whitelist_ecus) == 0 or ecu_type in r.whitelist_ecus:
          a = (addr, sub_addr, r.bus)
          # Build set of queries
          if sub_addr is None:
            if a not in parallel_queries:
              parallel_queries.append(a)
          else:  # subaddresses must be queried one by one
            if [a] not in queries:
              queries.append([a])

          # Build set of expected responses to filter
          response_addr = uds.get_rx_addr_for_tx_addr(addr, r.rx_offset)
          responses.add((response_addr, sub_addr, r.bus))

  queries.insert(0, parallel_queries)

  ecu_responses: Set[Tuple[int, Optional[int], int]] = set()
  for query in queries:
    ecu_responses.update(get_ecu_addrs(logcan, sendcan, set(query), responses, timeout=0.1))
  return ecu_responses


def get_brand_ecu_matches(ecu_rx_addrs):
  """Returns dictionary of brands and matches with ECUs in their FW versions"""

  brand_matches = {brand: set() for brand, _ in REQUESTS}
  brand_rx_offsets = set((brand, r.rx_offset) for brand, r in REQUESTS)

  for addr, sub_addr, _ in ecu_rx_addrs:
    # Since we can't know what request an ecu responded to, add matches for all possible rx offsets
    for brand, rx_offset in brand_rx_offsets:
      a = (uds.get_rx_addr_for_tx_addr(addr, -rx_offset), sub_addr)
      if a in FW_QUERY_CONFIGS[brand].ecus:
        brand_matches[brand].add(a)

  return brand_matches


def get_fw_versions_ordered(logcan, sendcan, ecu_rx_addrs, timeout=0.1, debug=False, progress=False):
  """Queries for FW versions ordering brands by likelihood, breaks when exact match is found"""

  all_car_fw = []
  brand_matches = get_brand_ecu_matches(ecu_rx_addrs)

  for brand in sorted(brand_matches, key=lambda b: len(brand_matches[b]), reverse=True):
    car_fw = get_fw_versions(logcan, sendcan, query_brand=brand, timeout=timeout, debug=debug, progress=progress)
    all_car_fw.extend(car_fw)
    # Try to match using FW returned from this brand only
    matches = match_fw_to_car_exact(build_fw_dict(car_fw))
    if len(matches) == 1:
      break

  return all_car_fw


def get_fw_versions(logcan, sendcan, query_brand=None, extra_config=None, timeout=0.1, debug=False, progress=False):
  configs = FW_QUERY_CONFIGS.copy()
  if extra_config is not None:
    configs['debug'] = extra_config

  # Get versions and build capnp list to put into CarParams
  car_fw = []

  for brand, config in configs.items():
    if query_brand is not None and brand != query_brand:
      continue

    # Extract ECU addresses to query from fingerprints
    # ECUs using a subaddress need be queried one by one, the rest can be done in parallel
    addrs = []
    parallel_addrs = list({(addr, sub_addr) for addr, sub_addr in config.ecus if sub_addr is None})

    for addr, sub_addr in config.ecus.keys():
      a = (addr, sub_addr)
      if sub_addr is not None:
        if [a] not in addrs:
          addrs.append([a])

    addrs.insert(0, parallel_addrs)

    for r in config.requests:
      try:
        query_addrs = [(a, s) for (b, a, s) in addrs if (len(r.whitelist_ecus) == 0 or config.ecus[(a, s)] in r.whitelist_ecus)]
        query = IsoTpParallelQuery(sendcan, logcan, r.bus, query_addrs, r.request, r.response, r.rx_offset, debug=debug)
        for (addr, rx_addr), version in query.get_data(timeout).items():
          f = car.CarParams.CarFw.new_message()

          # TODO: verify it should always be in ecus dict
          f.ecu = config.ecus.get((addr[0], addr[1]), Ecu.unknown)
          f.fwVersion = version
          f.address = addr[0]
          f.responseAddress = rx_addr
          f.request = r.request
          f.brand = brand
          f.bus = r.bus

          if addr[1] is not None:
            f.subAddress = addr[1]

          car_fw.append(f)
      except Exception:
        cloudlog.warning(f"FW query exception: {traceback.format_exc()}")

  return car_fw


if __name__ == "__main__":
  import time
  import argparse
  import cereal.messaging as messaging
  from selfdrive.car.vin import get_vin

  parser = argparse.ArgumentParser(description='Get firmware version of ECUs')
  parser.add_argument('--scan', action='store_true')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--brand', help='Only query addresses/with requests for this brand')
  args = parser.parse_args()

  logcan = messaging.sub_sock('can')
  sendcan = messaging.pub_sock('sendcan')

  extra_config: Optional[FwQueryConfig] = None
  if args.scan:
    extra_config = FwQueryConfig(requests=[r for _, r in REQUESTS], ecus={})
    # Honda
    for i in range(256):
      extra_config.ecus[(0x18da00f1 + (i << 8), None)] = Ecu.unknown
      extra_config.ecus[(0x700 + i, None)] = Ecu.unknown
      extra_config.ecus[(0x750, i)] = Ecu.unknown

  time.sleep(1.)

  t = time.time()
  print("Getting vin...")
  addr, vin_rx_addr, vin = get_vin(logcan, sendcan, 1, retry=10, debug=args.debug)
  print(f'TX: {hex(addr)}, RX: {hex(vin_rx_addr)}, VIN: {vin}')
  print(f"Getting VIN took {time.time() - t:.3f} s")
  print()

  t = time.time()
  fw_vers = get_fw_versions(logcan, sendcan, query_brand=args.brand, extra_config=extra_config, debug=args.debug, progress=True)
  _, candidates = match_fw_to_car(fw_vers)

  print()
  print("Found FW versions")
  print("{")
  padding = max([len(fw.brand) for fw in fw_vers] or [0])
  for version in fw_vers:
    subaddr = None if version.subAddress == 0 else hex(version.subAddress)
    print(f"  Brand: {version.brand:{padding}}, bus: {version.bus} - (Ecu.{version.ecu}, {hex(version.address)}, {subaddr}): [{version.fwVersion}]")
  print("}")

  print()
  print("Possible matches:", candidates)
  print(f"Getting fw took {time.time() - t:.3f} s")
