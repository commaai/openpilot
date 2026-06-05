#!/usr/bin/env python3
from collections import defaultdict
from opendbc.car.toyota.values import PLATFORM_CODE_ECUS, get_platform_codes
from opendbc.car.toyota.fingerprints import FW_VERSIONS

if __name__ == "__main__":
  parts_for_ecu: dict = defaultdict(set)
  cars_for_code: dict = defaultdict(lambda: defaultdict(set))
  for car_model, ecus in FW_VERSIONS.items():
    print()
    print(car_model)
    for ecu in sorted(ecus):
      if ecu[0] not in PLATFORM_CODE_ECUS:
        continue

      platform_codes = get_platform_codes(ecus[ecu])
      parts_for_ecu[ecu] |= {code.split(b'-')[0] for code in platform_codes if code.count(b'-') > 1}
      for code in platform_codes:
        cars_for_code[ecu][b'-'.join(code.split(b'-')[:2])] |= {car_model}
      print(f'  (Ecu.{ecu[0]}, {hex(ecu[1])}, {ecu[2]}):')
      print(f'    Codes: {platform_codes}')

  print('\nECU parts:')
  for ecu, parts in parts_for_ecu.items():
    print(f'  (Ecu.{ecu[0]}, {hex(ecu[1])}, {ecu[2]}): {parts}')

  print('\nCar models vs. platform codes (no major versions):')
  for ecu, codes in cars_for_code.items():
    print(f' (Ecu.{ecu[0]}, {hex(ecu[1])}, {ecu[2]}):')
    for code, cars in codes.items():
      print(f'    {code!r}: {sorted(cars)}')
