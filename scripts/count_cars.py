#!/usr/bin/env python3
import os
from collections import Counter
from pprint import pprint

from common.basedir import BASEDIR

with open(os.path.join(BASEDIR, "docs/CARS.md")) as f:
  lines = f.readlines()
  cars = [l for l in lines if l.strip().startswith("|") and l.strip().endswith("|") and
                              "Make" not in l and any(c.isalpha() for c in l)]

  make_count = Counter(l.split('|')[1].split('|')[0].strip() for l in cars)
  print("\n", "*"*20, len(cars), "total", "*"*20, "\n")
  pprint(make_count)
