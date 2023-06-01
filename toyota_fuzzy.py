#!/usr/bin/env python3
import random
from collections import defaultdict
from selfdrive.car.toyota.values import FW_VERSIONS
from selfdrive.car.toyota.values import match_fw_to_toyota_fuzzy
from selfdrive.car.fw_versions import match_fw_to_car_fuzzy, match_fw_to_car
from cereal import car

Ecu = car.CarParams.Ecu
ECU_NAME = {v: k for k, v in Ecu.schema.enumerants.items()}


def get_common_prefix(l):
  s1 = min(l)
  s2 = max(l)
  for i, c in enumerate(s1):
    if c != s2[i]:
      return s1[:i]
  return s1


# # get all fw for each ecu to find common prefixes
# all_fw_by_ecu = defaultdict(list)
# for car, fw_by_ecu in FW_VERSIONS.items():
#   for ecu, fws in fw_by_ecu.items():
#     all_fw_by_ecu[ecu].extend(fws)

# # find common prefixes
# for ecu, fws in all_fw_by_ecu.items():
#   fws_unicode = [fw.decode('utf-8', 'ignore').translate(dict.fromkeys(range(32))) for fw in fws]
#   print('  *', f'{str((ECU_NAME[ecu[0]], hex(ecu[1]))) + ":":24}', get_common_prefix(fws_unicode), f'(ONLY ONE FW)' if len(fws) == 1 else f'({len(fws)} FW)')

# raise Exception

# PREFIXES_BY_ECU = {
#   (Ecu.abs, 0x7b0, None): "F1526",
#   (Ecu.engine, 0x700, None): "89663",
#   (Ecu.dsu, 0x791, None): "80",
#   (Ecu.fwdRadar, 0x750, 0xf): "8821F",
#   (Ecu.fwdCamera, 0x750, 0x6d): "8646F",
#   (Ecu.eps, 0x7a1, None): "8965B",
# }
#
# # find car codes (ABS only for now)
# for car, fw_by_ecu in FW_VERSIONS.items():
#   print()
#   print(' *', car)
#   for ecu, fws in sorted(fw_by_ecu.items(), key=lambda e: e[0]):
#     fws_no_len = [fw.replace(b'\x01', b'').replace(b'\x02', b'').replace(b'\x03', b'') for fw in fws]
#     if ecu in PREFIXES_BY_ECU:
#       pfx = PREFIXES_BY_ECU[ecu]
#       car_ecu_codes = set([fw[len(pfx):len(pfx) + 4] for fw in fws_no_len])
#       print('  *', f'{str((ECU_NAME[ecu[0]], hex(ecu[1]))) + ":":24}', car_ecu_codes, f'(ONLY ONE FW)' if len(fws) == 1 else f'({len(fws)} FW)')
#
# raise Exception


# # find common prefixes for fw for each car
# for car, fw_by_ecu in FW_VERSIONS.items():
#   print(' *', car)
#   for ecu, fws in sorted(fw_by_ecu.items(), key=lambda e: e[0]):
#     fws_unicode = [fw.decode('utf-8', 'ignore').translate(dict.fromkeys(range(32))) for fw in fws]
#     print('  *', f'{ECU_NAME[ecu[0]] + ":":24}', get_common_prefix(fws_unicode), f'(ONLY ONE FW)' if len(fws) == 1 else f'({len(fws)} FW)')
#
# raise Exception


issue = set()

for i in range(200):
  for car, versions in FW_VERSIONS.items():
    print('testing', car)
    versions = {(ecu[1], ecu[2]): [random.choice(fws)] for ecu, fws in versions.items()}
    # versions = {k: v for k, v in random.sample(list(versions.items()), len(versions) - 1)}
    rand_ecu = random.choice(list(versions.keys()))
    versions[rand_ecu] = [versions[rand_ecu][0] + b'random stuff not good']
    print(versions)

    ret = match_fw_to_toyota_fuzzy(versions)
    print('ret', ret)
    raise Exception

    # is_exact, ret = match_fw_to_car(versions, allow_exact=False)
    # ret = match_fw_to_car_fuzzy(versions, config=None)
    if len(ret) != 1:
      print(f'Not one match! {car=}: {ret=}')
      issue.add(car)
    elif list(ret)[0] != car:
      print(f'Matched to wrong car! real: {car}, match: {ret}')
      issue.add(car)

    # print(ret)
print()
print('bad:', issue)
print()
print('good:', set(FW_VERSIONS) - issue)
raise Exception


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
