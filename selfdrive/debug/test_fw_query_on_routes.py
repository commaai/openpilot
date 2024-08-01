#!/usr/bin/env python3
# type: ignore

from collections import defaultdict
import argparse
import os
import traceback
from tqdm import tqdm
from openpilot.tools.lib.logreader import LogReader, ReadMode
from openpilot.tools.lib.route import SegmentRange
from openpilot.selfdrive.car.car_helpers import interface_names
from openpilot.selfdrive.car.fingerprints import MIGRATION
from openpilot.selfdrive.car.fw_versions import VERSIONS, match_fw_to_car


NO_API = "NO_API" in os.environ
SUPPORTED_BRANDS = VERSIONS.keys()
SUPPORTED_CARS = [brand for brand in SUPPORTED_BRANDS for brand in interface_names[brand]]
UNKNOWN_BRAND = "unknown"

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run FW fingerprint on Qlog of route or list of routes')
  parser.add_argument('route', help='Route or file with list of routes')
  parser.add_argument('--car', help='Force comparison fingerprint to known car')
  args = parser.parse_args()

  if os.path.exists(args.route):
    routes = list(open(args.route))
  else:
    routes = [args.route]

  mismatches = defaultdict(list)

  not_fingerprinted = 0
  solved_by_fuzzy = 0

  good_exact = 0
  wrong_fuzzy = 0
  good_fuzzy = 0

  dongles = []
  for route in tqdm(routes):
    sr = SegmentRange(route)
    dongle_id = sr.dongle_id

    if dongle_id in dongles:
      continue

    if sr.slice == '' and sr.selector is None:
      route += '/0'

    lr = LogReader(route, default_mode=ReadMode.QLOG)

    try:
      dongles.append(dongle_id)

      CP = None
      for msg in lr:
        if msg.which() == "pandaStates":
          if msg.pandaStates[0].pandaType in ('unknown', 'whitePanda', 'greyPanda', 'pedal'):
            print("wrong panda type")
            break

        elif msg.which() == "carParams":
          CP = msg.carParams
          car_fw = [fw for fw in CP.carFw if not fw.logging]
          if len(car_fw) == 0:
            print("no fw")
            break

          live_fingerprint = CP.carFingerprint
          live_fingerprint = MIGRATION.get(live_fingerprint, live_fingerprint)

          if args.car is not None:
            live_fingerprint = args.car

          if live_fingerprint not in SUPPORTED_CARS:
            print("not in supported cars")
            break

          _, exact_matches = match_fw_to_car(car_fw, CP.carVin, allow_exact=True, allow_fuzzy=False)
          _, fuzzy_matches = match_fw_to_car(car_fw, CP.carVin, allow_exact=False, allow_fuzzy=True)

          if (len(exact_matches) == 1) and (list(exact_matches)[0] == live_fingerprint):
            good_exact += 1
            print(f"Correct! Live: {live_fingerprint} - Fuzzy: {fuzzy_matches}")

            # Check if fuzzy match was correct
            if len(fuzzy_matches) == 1:
              if list(fuzzy_matches)[0] != live_fingerprint:
                wrong_fuzzy += 1
                print("Fuzzy match wrong! Fuzzy:", fuzzy_matches, "Live:", live_fingerprint)
              else:
                good_fuzzy += 1
            break

          print("Old style:", live_fingerprint, "Vin", CP.carVin)
          print("New style (exact):", exact_matches)
          print("New style (fuzzy):", fuzzy_matches)

          padding = max([len(fw.brand or UNKNOWN_BRAND) for fw in car_fw])
          for version in sorted(car_fw, key=lambda fw: fw.brand):
            subaddr = None if version.subAddress == 0 else hex(version.subAddress)
            print(f"  Brand: {version.brand or UNKNOWN_BRAND:{padding}}, bus: {version.bus} - " +
                  f"(Ecu.{version.ecu}, {hex(version.address)}, {subaddr}): [{version.fwVersion}],")

          print("Mismatches")
          found = False
          for brand in SUPPORTED_BRANDS:
            car_fws = VERSIONS[brand]
            if live_fingerprint in car_fws:
              found = True
              expected = car_fws[live_fingerprint]
              for (_, expected_addr, expected_sub_addr), v in expected.items():
                for version in car_fw:
                  if version.brand != brand and len(version.brand):
                    continue
                  sub_addr = None if version.subAddress == 0 else version.subAddress
                  addr = version.address

                  if (addr, sub_addr) == (expected_addr, expected_sub_addr):
                    if version.fwVersion not in v:
                      print(f"({hex(addr)}, {'None' if sub_addr is None else hex(sub_addr)}) - {version.fwVersion}")

                      # Add to global list of mismatches
                      mismatch = (addr, sub_addr, version.fwVersion)
                      if mismatch not in mismatches[live_fingerprint]:
                        mismatches[live_fingerprint].append(mismatch)

          # No FW versions for this car yet, add them all to mismatch list
          if not found:
            for version in car_fw:
              sub_addr = None if version.subAddress == 0 else version.subAddress
              addr = version.address
              mismatch = (addr, sub_addr, version.fwVersion)
              if mismatch not in mismatches[live_fingerprint]:
                mismatches[live_fingerprint].append(mismatch)

          print()
          not_fingerprinted += 1

          if len(fuzzy_matches) == 1:
            if list(fuzzy_matches)[0] == live_fingerprint:
              solved_by_fuzzy += 1
            else:
              wrong_fuzzy += 1
              print("Fuzzy match wrong! Fuzzy:", fuzzy_matches, "Live:", live_fingerprint)

          break

      if CP is None:
        print("no CarParams in logs")
    except Exception:
      traceback.print_exc()
    except KeyboardInterrupt:
      break

  print()
  # Print FW versions that need to be added separated out by car and address
  for car, m in sorted(mismatches.items()):
    print(car)
    addrs = defaultdict(list)
    for (addr, sub_addr, version) in m:
      addrs[(addr, sub_addr)].append(version)

    for (addr, sub_addr), versions in addrs.items():
      print(f"  ({hex(addr)}, {'None' if sub_addr is None else hex(sub_addr)}): [")
      for v in versions:
        print(f"    {v},")
      print("  ]")
    print()

  print()
  print(f"Number of dongle ids checked: {len(dongles)}")
  print(f"Fingerprinted:                {good_exact}")
  print(f"Not fingerprinted:            {not_fingerprinted}")
  print(f"  of which had a fuzzy match: {solved_by_fuzzy}")

  print()
  print(f"Correct fuzzy matches:        {good_fuzzy}")
  print(f"Wrong fuzzy matches:          {wrong_fuzzy}")
  print()

