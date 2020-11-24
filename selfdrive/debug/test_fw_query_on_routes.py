#!/usr/bin/env python3
# type: ignore

from collections import defaultdict
import argparse
import os
import traceback
from tqdm import tqdm
from tools.lib.logreader import LogReader
from selfdrive.car.fw_versions import match_fw_to_car
from selfdrive.car.toyota.values import FW_VERSIONS as TOYOTA_FW_VERSIONS
from selfdrive.car.honda.values import FW_VERSIONS as HONDA_FW_VERSIONS
from selfdrive.car.hyundai.values import FW_VERSIONS as HYUNDAI_FW_VERSIONS

from selfdrive.car.toyota.values import FINGERPRINTS as TOYOTA_FINGERPRINTS
from selfdrive.car.honda.values import FINGERPRINTS as HONDA_FINGERPRINTS
from selfdrive.car.hyundai.values import FINGERPRINTS as HYUNDAI_FINGERPRINTS


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

  wrong = 0
  good = 0

  dongles = []
  for route in tqdm(routes):
    route = route.rstrip()
    dongle_id, time = route.split('|')
    qlog_path = f"cd:/{dongle_id}/{time}/0/qlog.bz2"

    if dongle_id in dongles:
      continue

    try:
      lr = LogReader(qlog_path)

      for msg in lr:
        if msg.which() == "health":
          if msg.health.hwType not in ['uno', 'blackPanda']:
            dongles.append(dongle_id)
            break

        elif msg.which() == "carParams":
          bts = msg.carParams.as_builder().to_bytes()

          car_fw = msg.carParams.carFw
          if len(car_fw) == 0:
            break

          dongles.append(dongle_id)
          live_fingerprint = msg.carParams.carFingerprint

          if args.car is not None:
            live_fingerprint = args.car

          if live_fingerprint not in list(TOYOTA_FINGERPRINTS.keys()) + list(HONDA_FINGERPRINTS.keys()) + list(HYUNDAI_FINGERPRINTS.keys()):
            break

          candidates = match_fw_to_car(car_fw)
          if (len(candidates) == 1) and (list(candidates)[0] == live_fingerprint):
            good += 1
            print("Correct", live_fingerprint, dongle_id)
            break

          print(f"{dongle_id}|{time}")
          print("Old style:", live_fingerprint, "Vin", msg.carParams.carVin)
          print("New style:", candidates)

          for version in car_fw:
            subaddr = None if version.subAddress == 0 else hex(version.subAddress)
            print(f"  (Ecu.{version.ecu}, {hex(version.address)}, {subaddr}): [{version.fwVersion}],")

          print("Mismatches")
          found = False
          for car_fws in [TOYOTA_FW_VERSIONS, HONDA_FW_VERSIONS, HYUNDAI_FW_VERSIONS]:
            if live_fingerprint in car_fws:
              found = True
              expected = car_fws[live_fingerprint]
              for (_, expected_addr, expected_sub_addr), v in expected.items():
                for version in car_fw:
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
          wrong += 1
          break
    except Exception:
      traceback.print_exc()
    except KeyboardInterrupt:
      break

  print(f"Fingerprinted: {good} - Not fingerprinted: {wrong}")
  print(f"Number of dongle ids checked: {len(dongles)}")
  print()

  # Print FW versions that need to be added seperated out by car and address
  for car, m in mismatches.items():
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
