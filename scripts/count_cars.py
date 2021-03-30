#!/usr/bin/env python3
import os
from common.basedir import BASEDIR

with open(os.path.join(BASEDIR, "README.md")) as f:
  lines = f.readlines()
  cars = [l for l in lines if l.strip().startswith("|") and l.strip().endswith("|") and
                              "Make" not in l and any(c.isalpha() for c in l)]
  print(''.join(cars))
  print(len(cars))
