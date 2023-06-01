#!/usr/bin/env python3
from collections import defaultdict
from selfdrive.car.toyota.values import FW_VERSIONS, EV_HYBRID_CAR

platform_prefixes = defaultdict(lambda: defaultdict(set))
hybrid_prefixes = defaultdict(lambda: defaultdict(set))
all_prefixes = defaultdict(set)

PREFIX_LEN = 5

for car, versions in FW_VERSIONS.items():
  for ecu_type, fws in versions.items():
    for fw in fws:
      fw = fw[1:] if fw[0] <= 3 else fw
      # fw = fw.split(b'\x00')[0]
      all_prefixes[ecu_type].add(fw[:PREFIX_LEN])
      hybrid_prefixes[car in EV_HYBRID_CAR][ecu_type].add(fw[:PREFIX_LEN])
      platform_prefixes[car][ecu_type].add(fw[PREFIX_LEN:PREFIX_LEN+4])
    # print(car, ecu_type)

print('all prefixes:')
for ecu_type, pfxs in all_prefixes.items():
  print(ecu_type, pfxs)

print('\nplatform prefixes:')
for platform in platform_prefixes:
  print(platform)
  for ecu_type, pfxs in platform_prefixes[platform].items():
    print(ecu_type, pfxs)
  print()
