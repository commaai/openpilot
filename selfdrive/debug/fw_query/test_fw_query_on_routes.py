#!/usr/bin/env python3
import traceback
import sys
from tqdm import tqdm
from tools.lib.logreader import LogReader
from selfdrive.car.fw_versions import match_fw_to_car
from selfdrive.car.toyota.values import FW_VERSIONS as TOYOTA_FW_VERSIONS
from selfdrive.car.honda.values import FW_VERSIONS as HONDA_FW_VERSIONS


def fw_versions_to_dict(car_fw):
  fw_versions = {}
  for f in car_fw:
    addr = f.address
    subaddr = f.subAddress
    if subaddr == 0:
      subaddr = None
    fw_versions[(addr, subaddr)] = f.fwVersion

  return fw_versions


if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: ./test_fw_query_on_routes.py <route_list>")
    sys.exit(1)

  i = 0
  dongles = []
  for route in tqdm(list(open(sys.argv[1]))):
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
          car_fw = msg.carParams.carFw
          if len(car_fw) == 0:
            break

          dongles.append(dongle_id)
          live_fingerprint = msg.carParams.carFingerprint

          if live_fingerprint not in list(TOYOTA_FW_VERSIONS.keys()) + list(HONDA_FW_VERSIONS.keys()):
            continue

          fw_versions = fw_versions_to_dict(car_fw)
          candidates = match_fw_to_car(fw_versions)
          if (len(candidates) == 1) and (list(candidates)[0] == live_fingerprint):
            print("Correct", live_fingerprint, dongle_id)
            break

          print(f"{dongle_id}|{time}")
          print("Old style:", live_fingerprint, "Vin", msg.carParams.carVin)
          print("New style:", candidates)

          for version in car_fw:
            subaddr = None if version.subAddress == 0 else hex(version.subAddress)
            print(f"  (Ecu.{version.ecu}, {hex(version.address)}, {subaddr}): [{version.fwVersion}],")

          print("Mismatches")
          for car_fws in [TOYOTA_FW_VERSIONS, HONDA_FW_VERSIONS]:
            if live_fingerprint in car_fws:

              expected = car_fws[live_fingerprint]
              for (_, expected_addr, expected_sub_addr), v in expected.items():
                for version in car_fw:
                  sub_addr = None if version.subAddress == 0 else version.subAddress
                  addr = version.address

                  if (addr, sub_addr) == (expected_addr, expected_sub_addr):
                    if version.fwVersion not in v:
                      print(f"({hex(addr)}, {'None' if sub_addr is None else hex(sub_addr)}) - {version.fwVersion}")

          print()
          i += 1
          break
    except Exception:
      traceback.print_exc()

  print(f"Unfingerprinted cars: {i}")
  print(f"Number of dongle ids checked: {len(dongles)}")
