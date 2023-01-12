#!/usr/bin/env python3
from cereal import car
from selfdrive.car.hyundai.values import FW_VERSIONS

Ecu = car.CarParams.Ecu

ice = []
plugin_hybrid = []
hybrid = []
ev = []

fw_keys = [(Ecu.fwdCamera, 0x7c4, None), (Ecu.fwdRadar, 0x7d0, None)]

for car in FW_VERSIONS.keys():
  # print()
  # print(car)
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
