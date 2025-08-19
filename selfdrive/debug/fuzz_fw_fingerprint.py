#!/usr/bin/env python3
# type: ignore
import random
from collections import defaultdict

from tqdm import tqdm

from opendbc.car.fw_versions import match_fw_to_car_fuzzy
from opendbc.car.toyota.values import FW_VERSIONS as TOYOTA_FW_VERSIONS
from opendbc.car.honda.values import FW_VERSIONS as HONDA_FW_VERSIONS
from opendbc.car.hyundai.values import FW_VERSIONS as HYUNDAI_FW_VERSIONS
from opendbc.car.volkswagen.values import FW_VERSIONS as VW_FW_VERSIONS


FWS = {}
FWS.update(TOYOTA_FW_VERSIONS)
FWS.update(HONDA_FW_VERSIONS)
FWS.update(HYUNDAI_FW_VERSIONS)
FWS.update(VW_FW_VERSIONS)

if __name__ == "__main__":
  total = 0
  match = 0
  wrong_match = 0
  confusions = defaultdict(set)

  for _ in tqdm(range(1000)):
    for candidate, fws in FWS.items():
      fw_dict = {}
      for (_, addr, subaddr), fw_list in fws.items():
        fw_dict[(addr, subaddr)] = [random.choice(fw_list)]

      matches = match_fw_to_car_fuzzy(fw_dict, log=False, exclude=candidate)

      total += 1
      if len(matches) == 1:
        if list(matches)[0] == candidate:
          match += 1
        else:
          confusions[candidate] |= matches
          wrong_match += 1

  print()
  for candidate, wrong_matches in sorted(confusions.items()):
    print(candidate, wrong_matches)

  print()
  print(f"Total fuzz cases: {total}")
  print(f"Correct matches:  {match}")
  print(f"Wrong matches:    {wrong_match}")


