#!/usr/bin/env python3
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
      codes = {code for code, _ in platform_codes}
      versions = sorted({version for _, version in platform_codes if version is not None})
      min_version, max_version = min(versions), max(versions)
      print(f'  (Ecu.{ECU_NAME[ecu[0]]}, {hex(ecu[1])}, {ecu[2]}):')
      print(f'    Codes: {codes}')
      print(f'    Versions: {versions}')
    print()
