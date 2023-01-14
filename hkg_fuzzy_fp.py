#!/usr/bin/env python3
import re

from cereal import car
from selfdrive.car.hyundai.values import FW_VERSIONS

Ecu = car.CarParams.Ecu

platform_codes = set()

# these ecus are available on all cars (even CAN FD with no OBD fingerprinting)
fw_keys = [(Ecu.fwdCamera, 0x7c4, None), (Ecu.fwdRadar, 0x7d0, None)]

RADAR_REGEX = br'([A-Z]+[A-Z0-9]*)'


def get_platform_codes(car_fw):
  radar_codes = set()
  camera_codes = set()
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
