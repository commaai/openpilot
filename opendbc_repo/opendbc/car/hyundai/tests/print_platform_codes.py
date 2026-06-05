#!/usr/bin/env python3
from opendbc.car.structs import CarParams
from opendbc.car.hyundai.values import PLATFORM_CODE_ECUS, get_platform_codes
from opendbc.car.hyundai.fingerprints import FW_VERSIONS

Ecu = CarParams.Ecu

if __name__ == "__main__":
  for car_model, ecus in FW_VERSIONS.items():
    print()
    print(car_model)
    for ecu in sorted(ecus):
      if ecu[0] not in PLATFORM_CODE_ECUS:
        continue

      platform_codes = get_platform_codes(ecus[ecu])
      codes = {code for code, _ in platform_codes}
      dates = {date for _, date in platform_codes if date is not None}
      print(f'  (Ecu.{ecu[0]}, {hex(ecu[1])}, {ecu[2]}):')
      print(f'    Codes: {codes}')
      print(f'    Dates: {dates}')
