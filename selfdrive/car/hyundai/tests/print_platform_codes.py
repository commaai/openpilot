#!/usr/bin/env python3
from cereal import car
from selfdrive.car.hyundai.values import FW_VERSIONS, PLATFORM_CODE_ECUS, get_platform_codes

Ecu = car.CarParams.Ecu
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}

if __name__ == "__main__":
  for car_model, ecus in FW_VERSIONS.items():
    print()
    print(car_model)
    for ecu in sorted(ecus, key=lambda x: int(x[0])):
      if ecu[0] not in PLATFORM_CODE_ECUS:
        continue

      codes = set()
      dates = set()
      for fw in ecus[ecu]:
        code = list(get_platform_codes([fw]))[0]
        codes.add(code.split(b"-")[0])
        if b"-" in code:
          dates.add(code.split(b"-")[1])

      print(f'  (Ecu.{ECU_NAME[ecu[0]]}, {hex(ecu[1])}, {ecu[2]}):')
      print(f'    Codes: {codes}')
      print(f'    Dates: {dates}')
