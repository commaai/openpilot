#!/usr/bin/env python3
from collections import defaultdict

from opendbc.car.structs import CarParams
from opendbc.car.ford.values import get_platform_codes
from opendbc.car.ford.fingerprints import FW_VERSIONS

Ecu = CarParams.Ecu

if __name__ == "__main__":
  cars_for_code: defaultdict = defaultdict(lambda: defaultdict(set))

  for car_model, ecus in FW_VERSIONS.items():
    print(car_model)
    for ecu in sorted(ecus):
      platform_codes = get_platform_codes(ecus[ecu])
      for code in platform_codes:
        cars_for_code[ecu][code].add(car_model)

      print(f'  (Ecu.{ecu[0]}, {hex(ecu[1])}, {ecu[2]}):')
      print(f'    Codes: {sorted(platform_codes)}')
    print()

  print('\nCar models vs. platform codes:')
  for ecu, codes in cars_for_code.items():
    print(f'  (Ecu.{ecu[0]}, {hex(ecu[1])}, {ecu[2]}):')
    for code, cars in codes.items():
      print(f'    {code!r}: {sorted(map(str, cars))}')
