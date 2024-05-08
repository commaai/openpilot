#!/usr/bin/env python3

import argparse
from collections import defaultdict
from openpilot.selfdrive.debug.format_fingerprints import format_brand_fw_versions

from openpilot.selfdrive.car.fw_versions import MODEL_TO_BRAND, match_fw_to_car
from openpilot.tools.lib.logreader import LogReader, ReadMode

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Auto fingerprint from a route")
  parser.add_argument("route", help="The route name to use")
  parser.add_argument("platform", help="The platform, or leave empty to auto-determine using fuzzy", default=None, nargs='?')
  args = parser.parse_args()

  lr = LogReader(args.route, ReadMode.QLOG)

  platform: str | None = None

  CP = lr.first("carParams")

  if CP is None:
    raise Exception("No fw versions in the provided route...")

  carFw = CP.carFw
  carVin = CP.carVin
  carPlatform = CP.carFingerprint

  if args.platform is None:  # attempt to auto-determine platform with other fuzzy fingerprints
    _, possible_platforms = match_fw_to_car(carFw, carVin, log=False)

    if len(possible_platforms) != 1:
      print(f"Unable to auto-determine platform, possible platforms: {possible_platforms}")

      if carPlatform != "MOCK":
        print("Using platform from route")
        platform = carPlatform
      else:
        platform = None
    else:
      platform = list(possible_platforms)[0]
  else:
    platform = args.platform

  if platform is None:
    raise Exception("unable to determine platform, try manually specifying the fingerprint.")

  print("Attempting to add fw version for: ", platform)

  fw_versions: dict[str, dict[tuple, list[bytes]]] = defaultdict(lambda: defaultdict(list))
  brand = MODEL_TO_BRAND[platform]

  for fw in carFw:
    if fw.brand == brand and not fw.logging:
      addr = fw.address
      subAddr = None if fw.subAddress == 0 else fw.subAddress
      key = (fw.ecu.raw, addr, subAddr)

      fw_versions[platform][key].append(fw.fwVersion)

  format_brand_fw_versions(brand, fw_versions)
