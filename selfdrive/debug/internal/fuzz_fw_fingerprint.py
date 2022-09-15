#!/usr/bin/env python3
# type: ignore
import random
from collections import defaultdict

from tqdm import tqdm

from selfdrive.car.fw_versions import match_fw_to_car_fuzzy
from selfdrive.car.fingerprints import FW_VERSIONS

if __name__ == "__main__":
  total = 0
  match = 0
  wrong_match = 0
  confusions = defaultdict(set)

  for _ in tqdm(range(1000)):
    for candidate, fws in FW_VERSIONS.items():
      fw_dict = {}
      for tp, fw_list in fws.items():
        if not len(fw_list):
          continue
        fw_dict[tp] = random.choice(fw_list)

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
