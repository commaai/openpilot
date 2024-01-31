#!/usr/bin/env python3
from collections import defaultdict

from cereal import car
from openpilot.selfdrive.car.ford.values import get_platform_codes
from openpilot.selfdrive.car.ford.fingerprints import FW_VERSIONS

Ecu = car.CarParams.Ecu
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}


if __name__ == "__main__":
  for car_model, ecus in FW_VERSIONS.items():
    print(car_model)
    for ecu in sorted(ecus, key=lambda x: int(x[0])):
      platform_codes = get_platform_codes(ecus[ecu])

      code_versions = defaultdict(set)
      for code, version in platform_codes:
        code_versions[code].add(version)

      print(f'  (Ecu.{ECU_NAME[ecu[0]]}, {hex(ecu[1])}, {ecu[2]}):')
      for code, versions in code_versions.items():
        print(f'    {code!r}: {sorted(versions)}')
    print()
