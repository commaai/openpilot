#!/usr/bin/env python3

import argparse
from collections import defaultdict
from openpilot.selfdrive.debug.format_fingerprints import format_brand_fw_versions

from opendbc.car.fingerprints import MIGRATION
from opendbc.car.fw_versions import MODEL_TO_BRAND, match_fw_to_car
from openpilot.tools.lib.logreader import LogReader, ReadMode

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Auto fingerprint from a route")
  parser.add_argument("route", help="The route name to use")
  parser.add_argument("platform", help="The platform, or leave empty to auto-determine using fuzzy", default=None, nargs="?")
  args = parser.parse_args()

  lr = LogReader(args.route, ReadMode.QLOG)
  CP = lr.first("carParams")
  assert CP is not None, "No carParams in route"

  carPlatform = MIGRATION.get(CP.carFingerprint, CP.carFingerprint)

  if args.platform is not None:
    platform = args.platform
  elif carPlatform != "MOCK":
    platform = carPlatform
  else:
    _, matches = match_fw_to_car(CP.carFw, CP.carVin, log=False)
    assert len(matches) == 1, f"Unable to auto-determine platform, matches: {matches}"
    platform = list(matches)[0]

  print("Attempting to add fw version for:", platform)

  fw_versions: dict[str, dict[tuple, list[bytes]]] = defaultdict(lambda: defaultdict(list))
  brand = MODEL_TO_BRAND[platform]

  for fw in CP.carFw:
    if fw.brand == brand and not fw.logging:
      addr = fw.address
      subAddr = None if fw.subAddress == 0 else fw.subAddress
      key = (fw.ecu.raw, addr, subAddr)

      fw_versions[platform][key].append(fw.fwVersion)

  format_brand_fw_versions(brand, fw_versions)
