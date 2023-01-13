#!/usr/bin/env python3
import re

from cereal import car
from selfdrive.car.hyundai.values import FW_VERSIONS

Ecu = car.CarParams.Ecu

ice = []
plugin_hybrid = []
hybrid = []
ev = []

platform_codes = set()

# these ecus are available on all cars (even CAN FD with no OBD fingerprinting)
fw_keys = [(Ecu.fwdCamera, 0x7c4, None), (Ecu.fwdRadar, 0x7d0, None)]

# RADAR_REGEX = br'[A-Z]+|\S+'
# RADAR_REGEX = br'(?=[A-Z][a-z]|_)'
RADAR_REGEX = br'([A-Z]+[A-Z0-9]*)'


def get_platform_codes(car_fw):
  codes = set()
  for fw in car_fw[(Ecu.fwdRadar, 0x7d0, None)]:
    start_idx = fw.index(b'\xf1\x00')
    fw = fw[start_idx + 2:][:4]
    # fw = fw.decode('utf-8', 'ignore')

    # code, radar_code_variant = re.findall(RADAR_REGEX, fw)
    code = re.match(RADAR_REGEX, fw).group(0)
    radar_code_variant = fw[len(code):4].replace(b'_', b'').replace(b' ', b'')
    # code, radar_code_variant = fw[:2], fw[2:4].replace(b"_", b"")

    print(f"{code=}, {radar_code_variant=}, {fw=}")
    codes.add((code, radar_code_variant))
    # continue
    #
    # # code = ''
    # # radar_code_variant = ''
    #
    # end_of_platform = False
    # for idx, char in enumerate(fw):
    #   if char.islower():
    #     end_of_platform = True
    #   elif char in ['_']:  # end of platform code
    #     break
    #
    #   if not end_of_platform:
    #     code += char
    #   else:
    #     if char == ' ':
    #       break
    #     radar_code_variant += char
    #
    # print(f"{code=}, {radar_code_variant=}, {fw=}")
  return codes


for car, car_fw in FW_VERSIONS.items():
  print()
  print(car)

  # print(car, car_fw)
  codes = get_platform_codes(car_fw)
  print(codes)
  if car == 'HYUNDAI PALISADE 2020':
    print('skipping')
    continue
  assert len(codes) == 1

  continue

  for ecu, fws in FW_VERSIONS[car].items():
    if ecu not in fw_keys:
      continue

    # print(ecu)
    for fw in fws:
      start_idx = fw.index(b'\xf1\x00')
      fw = fw[start_idx+2:]
      fw = fw.split()[0]
      fw = fw.decode('utf-8', 'ignore')
      # print(fw)
      if (Ecu.fwdRadar, 0x7d0, None) == ecu:
        # if 'he' in fw:
        if fw.endswith('he'):
          hybrid.append(car)
        # elif fw.endswith('P'):
        #   print('interesting:', car)
        elif fw.endswith('ev'):
          ev.append(car)
        else:
          ice.append(car)

print('\nHybrid:', list(set(hybrid)), '\n')
print('Electric:', list(set(ev)), '\n')
print('ICE:', list(set(ice)), '\n')
