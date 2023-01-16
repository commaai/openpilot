#!/usr/bin/env python3
import random
from collections import defaultdict
from selfdrive.car.toyota.values import FW_VERSIONS
from selfdrive.car.toyota.values import match_fw_to_toyota_fuzzy
from cereal import car

Ecu = car.CarParams.Ecu

issue = set()

for i in range(200):
  for car, versions in FW_VERSIONS.items():
    versions = {(ecu[1], ecu[2]): [random.choice(fws)] for ecu, fws in versions.items()}
    # print(versions)

    ret = match_fw_to_toyota_fuzzy(versions)
    if len(ret) != 1:
      print(f'Not one match! {car=}: {ret=}')
      issue.add(car)

    # print(ret)
print()
print(issue)
print()
print(set(FW_VERSIONS) - issue)
exit(0)


def get_common_prefix(l):
  s1 = min(l)
  s2 = max(l)
  for i, c in enumerate(s1):
    if c != s2[i]:
      return s1[:i]
  return s1


prefixes = defaultdict(dict)

for car, fw_versions in FW_VERSIONS.items():
  # print(car)  # , fw_versions)
  for ecu, fws in fw_versions.items():
    print(car, ecu, len(get_common_prefix(fws)), get_common_prefix(fws))
    prefixes[car][ecu] = get_common_prefix(fws)
    # print((ecu[0], hex(ecu[1]), ecu[2]), get_common_prefix(fws), len(fws))
  # print()
  print()
exit()

prefixes = dict(prefixes)
for prefix_car, fw_prefixes in prefixes.items():
  if prefix_car != 'TOYOTA CAMRY 2021':
    continue
  print(f'{prefix_car=}')
  for fw_car, fw_versions in FW_VERSIONS.items():
    correct_prefix = []
    for prefix_ecu, fw_prefix in fw_prefixes.items():
      correct_prefix.append(
        prefix_ecu in fw_versions and all([fw[:len(fw_prefix)] == fw_prefix for fw in fw_versions[prefix_ecu]]))
    print(all(correct_prefix), fw_car, correct_prefix)
  break

print()
print()
print(prefixes['TOYOTA CAMRY 2021'])
print(prefixes['LEXUS ES 2019'])
