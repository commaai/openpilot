#!/usr/bin/env python3
import re
import random

from cereal import car
from selfdrive.car.hyundai.values import FW_VERSIONS, match_fw_to_hyundai_fuzzy

Ecu = car.CarParams.Ecu

platform_codes = set()

# these ecus are available on all cars (even CAN FD with no OBD fingerprinting)
fw_keys = [(Ecu.fwdCamera, 0x7c4, None), (Ecu.fwdRadar, 0x7d0, None)]

RADAR_REGEX = br'([A-Z]+[A-Z0-9]*)'


def get_platform_codes(car_fw):
  radar_codes = set()
  camera_codes = set()
  if (Ecu.fwdRadar, 0x7d0, None) in car_fw:
    for fw in car_fw[(Ecu.fwdRadar, 0x7d0, None)]:
      start_idx = fw.index(b'\xf1\x00')
      fw = fw[start_idx + 2:][:4]

      # code = re.match(RADAR_REGEX, fw).group(0)  # TODO: check NONE, or have a test
      # radar_code_variant = fw[len(code):4].replace(b'_', b'').replace(b' ', b'')

      radar_code = fw[:4].replace(b" ", b"").replace(b"_", b"")

      radar_codes.add(radar_code)
      # codes.add((code, radar_code_variant))

  for fw in car_fw[(Ecu.fwdCamera, 0x7c4, None)]:
    start_idx = fw.index(b'\xf1\x00')
    fw = fw[start_idx + 2:][:4]

    # code = re.match(RADAR_REGEX, fw).group(0)  # TODO: check NONE, or have a test
    # radar_code_variant = fw[len(code):4].replace(b'_', b'').replace(b' ', b'')

    radar_code = fw[:4].replace(b" ", b"").replace(b"_", b"")

    camera_codes.add(radar_code)
    # codes.add((code, radar_code_variant))

  return radar_codes, camera_codes
  # return codes


for car, car_fw in FW_VERSIONS.items():
  radar_codes, camera_codes = get_platform_codes(car_fw)

  print(f"{car:36}: radar: {radar_codes}, camera: {camera_codes}")


issue = set()
for i in range(200):
  for car, versions in FW_VERSIONS.items():
    if 'HYUNDAI GENESIS' in car:
      continue
    versions = {(ecu[1], ecu[2]): [random.choice(fws)] for ecu, fws in versions.items()}
    # print(versions)

    ret = match_fw_to_hyundai_fuzzy(versions)
    # is_exact, ret = match_fw_to_car(versions, allow_exact=False)
    # ret = match_fw_to_car_fuzzy(versions, config=None)
    if len(ret) != 1:
      print(f'Not one match! {car=}: {ret=}')
      issue.add(car)
    elif list(ret)[0] != car:
      print(f'Matched to wrong car! real: {car}, match: {ret}')
      issue.add(car)

print()
print('bad:', issue)
print()
print('good:', set(FW_VERSIONS) - issue)
# raise Exception
