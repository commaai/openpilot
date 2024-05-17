#!/usr/bin/env python3
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, Protocol, TypeVar

from tqdm import tqdm
import capnp

import panda.python.uds as uds
from cereal import car
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.car.ecu_addrs import get_ecu_addrs
from openpilot.selfdrive.car.fingerprints import FW_VERSIONS
from openpilot.selfdrive.car.fw_query_definitions import AddrType, EcuAddrBusType, FwQueryConfig, LiveFwVersions, OfflineFwVersions
from openpilot.selfdrive.car.interfaces import get_interface_attr
from openpilot.selfdrive.car.isotp_parallel_query import IsoTpParallelQuery

Ecu = car.CarParams.Ecu
ESSENTIAL_ECUS = [Ecu.engine, Ecu.eps, Ecu.abs, Ecu.fwdRadar, Ecu.fwdCamera, Ecu.vsa]
FUZZY_EXCLUDE_ECUS = [Ecu.fwdCamera, Ecu.fwdRadar, Ecu.eps, Ecu.debug]

FW_QUERY_CONFIGS: dict[str, FwQueryConfig] = get_interface_attr('FW_QUERY_CONFIG', ignore_none=True)
VERSIONS = get_interface_attr('FW_VERSIONS', ignore_none=True)

MODEL_TO_BRAND = {c: b for b, e in VERSIONS.items() for c in e}
REQUESTS = [(brand, config, r) for brand, config in FW_QUERY_CONFIGS.items() for r in config.requests]

T = TypeVar('T')


def chunks(l: list[T], n: int = 128) -> Iterator[list[T]]:
  for i in range(0, len(l), n):
    yield l[i:i + n]


def is_brand(brand: str, filter_brand: str | None) -> bool:
  """Returns if brand matches filter_brand or no brand filter is specified"""
  return filter_brand is None or brand == filter_brand


def build_fw_dict(fw_versions: list[capnp.lib.capnp._DynamicStructBuilder], filter_brand: str = None) -> dict[AddrType, set[bytes]]:
  fw_versions_dict: defaultdict[AddrType, set[bytes]] = defaultdict(set)
  for fw in fw_versions:
    if is_brand(fw.brand, filter_brand) and not fw.logging:
      sub_addr = fw.subAddress if fw.subAddress != 0 else None
      fw_versions_dict[(fw.address, sub_addr)].add(fw.fwVersion)
  return dict(fw_versions_dict)


class MatchFwToCar(Protocol):
  def __call__(self, live_fw_versions: LiveFwVersions, match_brand: str = None, log: bool = True) -> set[str]:
    ...


def match_fw_to_car_fuzzy(live_fw_versions: LiveFwVersions, match_brand: str = None, log: bool = True, exclude: str = None) -> set[str]:
  """Do a fuzzy FW match. This function will return a match, and the number of firmware version
  that were matched uniquely to that specific car. If multiple ECUs uniquely match to different cars
  the match is rejected."""

  # Build lookup table from (addr, sub_addr, fw) to list of candidate cars
  all_fw_versions = defaultdict(list)
  for candidate, fw_by_addr in FW_VERSIONS.items():
    if not is_brand(MODEL_TO_BRAND[candidate], match_brand):
      continue

    if candidate == exclude:
      continue

    for addr, fws in fw_by_addr.items():
      # These ECUs are known to be shared between models (EPS only between hybrid/ICE version)
      # Getting this exactly right isn't crucial, but excluding camera and radar makes it almost
      # impossible to get 3 matching versions, even if two models with shared parts are released at the same
      # time and only one is in our database.
      if addr[0] in FUZZY_EXCLUDE_ECUS:
        continue
      for f in fws:
        all_fw_versions[(addr[1], addr[2], f)].append(candidate)

  matched_ecus = set()
  match: str | None = None
  for addr, versions in live_fw_versions.items():
    ecu_key = (addr[0], addr[1])
    for version in versions:
      # All cars that have this FW response on the specified address
      candidates = all_fw_versions[(*ecu_key, version)]

      if len(candidates) == 1:
        matched_ecus.add(ecu_key)
        if match is None:
          match = candidates[0]
        # We uniquely matched two different cars. No fuzzy match possible
        elif match != candidates[0]:
          return set()

  # Note that it is possible to match to a candidate without all its ECUs being present
  # if there are enough matches. FIXME: parameterize this or require all ECUs to exist like exact matching
  if match and len(matched_ecus) >= 2:
    if log:
      cloudlog.error(f"Fingerprinted {match} using fuzzy match. {len(matched_ecus)} matching ECUs")
    return {match}
  else:
    return set()


def match_fw_to_car_exact(live_fw_versions: LiveFwVersions, match_brand: str = None, log: bool = True, extra_fw_versions: dict = None) -> set[str]:
  """Do an exact FW match. Returns all cars that match the given
  FW versions for a list of "essential" ECUs. If an ECU is not considered
  essential the FW version can be missing to get a fingerprint, but if it's present it
  needs to match the database."""
  if extra_fw_versions is None:
    extra_fw_versions = {}

  invalid = set()
  candidates = {c: f for c, f in FW_VERSIONS.items() if
                is_brand(MODEL_TO_BRAND[c], match_brand)}

  for candidate, fws in candidates.items():
    config = FW_QUERY_CONFIGS[MODEL_TO_BRAND[candidate]]
    for ecu, expected_versions in fws.items():
      expected_versions = expected_versions + extra_fw_versions.get(candidate, {}).get(ecu, [])
      ecu_type = ecu[0]
      addr = ecu[1:]

      found_versions = live_fw_versions.get(addr, set())
      if not len(found_versions):
        # Some models can sometimes miss an ecu, or show on two different addresses
        # FIXME: this logic can be improved to be more specific, should require one of the two addresses
        if candidate in config.non_essential_ecus.get(ecu_type, []):
          continue

        # Ignore non essential ecus
        if ecu_type not in ESSENTIAL_ECUS:
          continue

      # Virtual debug ecu doesn't need to match the database
      if ecu_type == Ecu.debug:
        continue

      if not any(found_version in expected_versions for found_version in found_versions):
        invalid.add(candidate)
        break

  return set(candidates.keys()) - invalid


def match_fw_to_car(fw_versions: list[capnp.lib.capnp._DynamicStructBuilder], vin: str,
                    allow_exact: bool = True, allow_fuzzy: bool = True, log: bool = True) -> tuple[bool, set[str]]:
  # Try exact matching first
  exact_matches: list[tuple[bool, MatchFwToCar]] = []
  if allow_exact:
    exact_matches = [(True, match_fw_to_car_exact)]
  if allow_fuzzy:
    exact_matches.append((False, match_fw_to_car_fuzzy))

  for exact_match, match_func in exact_matches:
    # For each brand, attempt to fingerprint using all FW returned from its queries
    matches: set[str] = set()
    for brand in VERSIONS.keys():
      fw_versions_dict = build_fw_dict(fw_versions, filter_brand=brand)
      matches |= match_func(fw_versions_dict, match_brand=brand, log=log)

      # If specified and no matches so far, fall back to brand's fuzzy fingerprinting function
      config = FW_QUERY_CONFIGS[brand]
      if not exact_match and not len(matches) and config.match_fw_to_car_fuzzy is not None:
        matches |= config.match_fw_to_car_fuzzy(fw_versions_dict, vin, VERSIONS[brand])

    if len(matches):
      return exact_match, matches

  return True, set()


def get_present_ecus(logcan, sendcan, num_pandas: int = 1) -> set[EcuAddrBusType]:
  params = Params()
  # queries are split by OBD multiplexing mode
  queries: dict[bool, list[list[EcuAddrBusType]]] = {True: [], False: []}
  parallel_queries: dict[bool, list[EcuAddrBusType]] = {True: [], False: []}
  responses: set[EcuAddrBusType] = set()

  for brand, config, r in REQUESTS:
    # Skip query if no panda available
    if r.bus > num_pandas * 4 - 1:
      continue

    for ecu_type, addr, sub_addr in config.get_all_ecus(VERSIONS[brand]):
      # Only query ecus in whitelist if whitelist is not empty
      if len(r.whitelist_ecus) == 0 or ecu_type in r.whitelist_ecus:
        a = (addr, sub_addr, r.bus)
        # Build set of queries
        if sub_addr is None:
          if a not in parallel_queries[r.obd_multiplexing]:
            parallel_queries[r.obd_multiplexing].append(a)
        else:  # subaddresses must be queried one by one
          if [a] not in queries[r.obd_multiplexing]:
            queries[r.obd_multiplexing].append([a])

        # Build set of expected responses to filter
        response_addr = uds.get_rx_addr_for_tx_addr(addr, r.rx_offset)
        responses.add((response_addr, sub_addr, r.bus))

  for obd_multiplexing in queries:
    queries[obd_multiplexing].insert(0, parallel_queries[obd_multiplexing])

  ecu_responses = set()
  for obd_multiplexing in queries:
    set_obd_multiplexing(params, obd_multiplexing)
    for query in queries[obd_multiplexing]:
      ecu_responses.update(get_ecu_addrs(logcan, sendcan, set(query), responses, timeout=0.1))
  return ecu_responses


def get_brand_ecu_matches(ecu_rx_addrs: set[EcuAddrBusType]) -> dict[str, set[AddrType]]:
  """Returns dictionary of brands and matches with ECUs in their FW versions"""

  brand_addrs = {brand: {(addr, subaddr) for _, addr, subaddr in config.get_all_ecus(VERSIONS[brand])} for
                 brand, config in FW_QUERY_CONFIGS.items()}
  brand_matches: dict[str, set[AddrType]] = {brand: set() for brand, _, _ in REQUESTS}

  brand_rx_offsets = {(brand, r.rx_offset) for brand, _, r in REQUESTS}
  for addr, sub_addr, _ in ecu_rx_addrs:
    # Since we can't know what request an ecu responded to, add matches for all possible rx offsets
    for brand, rx_offset in brand_rx_offsets:
      a = (uds.get_rx_addr_for_tx_addr(addr, -rx_offset), sub_addr)
      if a in brand_addrs[brand]:
        brand_matches[brand].add(a)

  return brand_matches


def set_obd_multiplexing(params: Params, obd_multiplexing: bool):
  if params.get_bool("ObdMultiplexingEnabled") != obd_multiplexing:
    cloudlog.warning(f"Setting OBD multiplexing to {obd_multiplexing}")
    params.remove("ObdMultiplexingChanged")
    params.put_bool("ObdMultiplexingEnabled", obd_multiplexing)
    params.get_bool("ObdMultiplexingChanged", block=True)
    cloudlog.warning("OBD multiplexing set successfully")


def get_fw_versions_ordered(logcan, sendcan, vin: str, ecu_rx_addrs: set[EcuAddrBusType], timeout: float = 0.1, num_pandas: int = 1,
                            debug: bool = False, progress: bool = False) -> list[capnp.lib.capnp._DynamicStructBuilder]:
  """Queries for FW versions ordering brands by likelihood, breaks when exact match is found"""

  all_car_fw = []
  brand_matches = get_brand_ecu_matches(ecu_rx_addrs)

  for brand in sorted(brand_matches, key=lambda b: len(brand_matches[b]), reverse=True):
    # Skip this brand if there are no matching present ECUs
    if not len(brand_matches[brand]):
      continue

    car_fw = get_fw_versions(logcan, sendcan, query_brand=brand, timeout=timeout, num_pandas=num_pandas, debug=debug, progress=progress)
    all_car_fw.extend(car_fw)

    # If there is a match using this brand's FW alone, finish querying early
    _, matches = match_fw_to_car(car_fw, vin, log=False)
    if len(matches) == 1:
      break

  return all_car_fw


def get_fw_versions(logcan, sendcan, query_brand: str = None, extra: OfflineFwVersions = None, timeout: float = 0.1, num_pandas: int = 1,
                    debug: bool = False, progress: bool = False) -> list[capnp.lib.capnp._DynamicStructBuilder]:
  versions = VERSIONS.copy()
  params = Params()

  if query_brand is not None:
    versions = {query_brand: versions[query_brand]}

  if extra is not None:
    versions.update(extra)

  # Extract ECU addresses to query from fingerprints
  # ECUs using a subaddress need be queried one by one, the rest can be done in parallel
  addrs = []
  parallel_addrs = []
  ecu_types = {}

  for brand, brand_versions in versions.items():
    config = FW_QUERY_CONFIGS[brand]
    for ecu_type, addr, sub_addr in config.get_all_ecus(brand_versions):
      a = (brand, addr, sub_addr)
      if a not in ecu_types:
        ecu_types[a] = ecu_type

      if sub_addr is None:
        if a not in parallel_addrs:
          parallel_addrs.append(a)
      else:
        if [a] not in addrs:
          addrs.append([a])

  addrs.insert(0, parallel_addrs)

  # Get versions and build capnp list to put into CarParams
  car_fw = []
  requests = [(brand, config, r) for brand, config, r in REQUESTS if is_brand(brand, query_brand)]
  for addr_group in tqdm(addrs, disable=not progress):  # split by subaddr, if any
    for addr_chunk in chunks(addr_group):
      for brand, config, r in requests:
        # Skip query if no panda available
        if r.bus > num_pandas * 4 - 1:
          continue

        # Toggle OBD multiplexing for each request
        if r.bus % 4 == 1:
          set_obd_multiplexing(params, r.obd_multiplexing)

        try:
          query_addrs = [(a, s) for (b, a, s) in addr_chunk if b in (brand, 'any') and
                         (len(r.whitelist_ecus) == 0 or ecu_types[(b, a, s)] in r.whitelist_ecus)]

          if query_addrs:
            query = IsoTpParallelQuery(sendcan, logcan, r.bus, query_addrs, r.request, r.response, r.rx_offset, debug=debug)
            for (tx_addr, sub_addr), version in query.get_data(timeout).items():
              f = car.CarParams.CarFw.new_message()

              f.ecu = ecu_types.get((brand, tx_addr, sub_addr), Ecu.unknown)
              f.fwVersion = version
              f.address = tx_addr
              f.responseAddress = uds.get_rx_addr_for_tx_addr(tx_addr, r.rx_offset)
              f.request = r.request
              f.brand = brand
              f.bus = r.bus
              f.logging = r.logging or (f.ecu, tx_addr, sub_addr) in config.extra_ecus
              f.obdMultiplexing = r.obd_multiplexing

              if sub_addr is not None:
                f.subAddress = sub_addr

              car_fw.append(f)
        except Exception:
          cloudlog.exception("FW query exception")

  return car_fw


if __name__ == "__main__":
  import time
  import argparse
  import cereal.messaging as messaging
  from openpilot.selfdrive.car.vin import get_vin

  parser = argparse.ArgumentParser(description='Get firmware version of ECUs')
  parser.add_argument('--scan', action='store_true')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--brand', help='Only query addresses/with requests for this brand')
  args = parser.parse_args()

  logcan = messaging.sub_sock('can')
  pandaStates_sock = messaging.sub_sock('pandaStates')
  sendcan = messaging.pub_sock('sendcan')

  # Set up params for boardd
  params = Params()
  params.remove("FirmwareQueryDone")
  params.put_bool("IsOnroad", False)
  time.sleep(0.2)  # thread is 10 Hz
  params.put_bool("IsOnroad", True)

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
  set_obd_multiplexing(params, True)
  vin_rx_addr, vin_rx_bus, vin = get_vin(logcan, sendcan, (0, 1), retry=10, debug=args.debug)
  print(f'RX: {hex(vin_rx_addr)}, BUS: {vin_rx_bus}, VIN: {vin}')
  print(f"Getting VIN took {time.time() - t:.3f} s")
  print()

  t = time.time()
  fw_vers = get_fw_versions(logcan, sendcan, query_brand=args.brand, extra=extra, num_pandas=num_pandas, debug=args.debug, progress=True)
  _, candidates = match_fw_to_car(fw_vers, vin)

  print()
  print("Found FW versions")
  print("{")
  padding = max([len(fw.brand) for fw in fw_vers] or [0])
  for version in fw_vers:
    subaddr = None if version.subAddress == 0 else hex(version.subAddress)
    print(f"  Brand: {version.brand:{padding}}, bus: {version.bus}, OBD: {version.obdMultiplexing} - " +
          f"(Ecu.{version.ecu}, {hex(version.address)}, {subaddr}): [{version.fwVersion}]")
  print("}")

  print()
  print("Possible matches:", candidates)
  print(f"Getting fw took {time.time() - t:.3f} s")
