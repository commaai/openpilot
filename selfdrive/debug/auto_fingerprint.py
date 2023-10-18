#!/usr/bin/env python3

import argparse
from openpilot.common.basedir import BASEDIR

from openpilot.tools.lib.logreader import MultiLogIterator
from openpilot.tools.lib.route import Route
from openpilot.selfdrive.car.fw_versions import match_fw_to_car
from openpilot.selfdrive.car.interfaces import get_interface_attr


ALL_FW_VERSIONS = get_interface_attr("FW_VERSIONS")
ALL_CARS = get_interface_attr("CAR")

PLATFORM_TO_PYTHON_CAR_NAME = {brand: {car.value: car.name for car in ALL_CARS[brand]} for brand in ALL_CARS}
BRAND_TO_PLATFORMS = {brand: [car.value for car in ALL_CARS[brand]] for brand in ALL_CARS}
PLATFORM_TO_BRAND = dict(sum([[(platform, brand) for platform in BRAND_TO_PLATFORMS[brand]] for brand in BRAND_TO_PLATFORMS], []))

def add_fw_versions(brand, platform, new_fw_versions):
  filename = f"{BASEDIR}/selfdrive/car/{brand}/values.py"
  with open(filename, "r") as f:
    values_py = f.read()

  for key in new_fw_versions.keys():
    ecu, addr, subAddr = key
    fw_version = new_fw_versions[key]

    platform_start = values_py.index(f"CAR.{PLATFORM_TO_PYTHON_CAR_NAME[brand][platform]}: {{")

    start = values_py.index(f"(Ecu.{ecu}, {hex(addr)}, {subAddr}): [", platform_start)

    try:
      end_str = "],\n"
      end = values_py.index(end_str, start)
    except ValueError:
      end_str = "]\n"
      end = values_py.index(end_str, start)

    values_py = values_py[:end] + f"  {fw_version},\n    " + values_py[end:]

  with open(filename, "w") as f:
    f.write(values_py)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Auto fingerprint from a route")
  parser.add_argument("route", help="The route name to use")
  parser.add_argument("platform", help="The platform, or leave empty to auto-determine using fuzzy", default=None, nargs='?')
  args = parser.parse_args()

  route = Route(args.route)
  lr = MultiLogIterator(route.qlog_paths())

  carFw = None
  carVin = None
  carPlatform = None

  for msg in lr:
    if msg.which() == "carParams":
      carFw = msg.carParams.carFw
      carVin = msg.carParams.carVin
      carPlatform = msg.carParams.carFingerprint
      break

  if carFw is None:
    raise Exception("No fw versions in the provided route...")

  if args.platform is None: # attempt to auto-determine platform with other fuzzy fingerprints
    _, possible_platforms = match_fw_to_car(carFw, log=False)

    if len(possible_platforms) != 1:
      print(f"Unable to auto-determine platform, possible platforms: {possible_platforms}")

      if carPlatform != "mock":
        print("Using platform from route")
        platform = carPlatform
      else:
        raise Exception("unable to determine platform, try manually specifying the fingerprint.")
    else:
      platform = list(possible_platforms)[0]

  else:
    platform = args.platform

  print("Attempting to add fw version for: ", platform)

  brand = PLATFORM_TO_BRAND[platform]

  new_fw_versions = {}

  for fw in carFw:
    if fw.brand == brand:
      addr = fw.address
      subAddr = None if fw.subAddress == 0 else fw.subAddress
      key = (fw.ecu.raw, addr, subAddr)

      if key in ALL_FW_VERSIONS[brand][platform]:
        fw_versions = set(ALL_FW_VERSIONS[brand][platform][key])
        if fw.fwVersion not in fw_versions:
          new_fw_versions[(fw.ecu, addr, subAddr)] = fw.fwVersion

  if not new_fw_versions:
    print("No new fw versions found...")

  add_fw_versions(brand, platform, new_fw_versions)